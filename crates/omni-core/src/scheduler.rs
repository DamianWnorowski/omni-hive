//! Adaptive Scheduler - Ultra-resource efficient polling

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

use crate::config::SchedulerConfig;

/// Adaptive scheduler that adjusts polling based on activity
pub struct AdaptiveScheduler {
    /// Idle interval (1 hour when sleeping)
    idle_ms: AtomicU64,

    /// Active interval (30 sec when user active)
    active_ms: AtomicU64,

    /// Burst interval (5 sec right after action)
    burst_ms: AtomicU64,

    /// Last activity timestamp
    last_activity: RwLock<Instant>,

    /// Is actively polling
    active: AtomicBool,

    /// Burst mode duration
    burst_duration: Duration,

    /// Active mode duration
    active_duration: Duration,

    /// Current backoff multiplier
    backoff_multiplier: AtomicU64,
}

impl AdaptiveScheduler {
    pub fn new(config: &SchedulerConfig) -> Self {
        Self {
            idle_ms: AtomicU64::new(config.idle_interval_ms),
            active_ms: AtomicU64::new(config.active_interval_ms),
            burst_ms: AtomicU64::new(config.burst_interval_ms),
            last_activity: RwLock::new(Instant::now()),
            active: AtomicBool::new(false),
            burst_duration: Duration::from_secs(30),
            active_duration: Duration::from_secs(300),
            backoff_multiplier: AtomicU64::new(1),
        }
    }

    /// Get current polling interval based on activity state
    pub async fn get_interval(&self) -> Duration {
        let last = *self.last_activity.read().await;
        let elapsed = last.elapsed();
        let multiplier = self.backoff_multiplier.load(Ordering::Relaxed);

        let ms = if elapsed < self.burst_duration {
            // Burst mode: user just did something
            self.burst_ms.load(Ordering::Relaxed)
        } else if elapsed < self.active_duration {
            // Active mode: user was recently active
            self.active_ms.load(Ordering::Relaxed)
        } else {
            // Idle mode: user is away
            self.idle_ms.load(Ordering::Relaxed)
        };

        Duration::from_millis(ms * multiplier)
    }

    /// Increase backoff due to failure
    pub async fn backoff(&self) {
        let current = self.backoff_multiplier.load(Ordering::Relaxed);
        if current < 32 { // Max 32x backoff
            self.backoff_multiplier.store(current * 2, Ordering::Relaxed);
            tracing::warn!("Scheduler: Backoff increased to {}x due to failures.", current * 2);
        }
    }

    /// Reset backoff after success
    pub async fn reset_backoff(&self) {
        self.backoff_multiplier.store(1, Ordering::Relaxed);
    }

    /// Record user activity (triggers burst mode)
    pub async fn record_activity(&self) {
        let mut last = self.last_activity.write().await;
        *last = Instant::now();
        self.active.store(true, Ordering::Relaxed);
    }

    /// Check if scheduler is in active mode
    pub async fn is_active(&self) -> bool {
        let last = *self.last_activity.read().await;
        last.elapsed() < self.active_duration
    }

    /// Force wake up (e.g., external trigger)
    pub fn wake_up(&self) {
        self.active.store(true, Ordering::Relaxed);
    }

    /// Put to sleep
    pub fn sleep(&self) {
        self.active.store(false, Ordering::Relaxed);
    }
}
