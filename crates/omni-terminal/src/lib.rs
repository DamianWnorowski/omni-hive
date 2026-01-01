//! OMNI-TERMINAL: Physics-based terminal visualization
//!
//! Integration layer for neuroflux-style rendering

use anyhow::Result;
use crossterm::{
    cursor,
    style::{Color, Print, SetBackgroundColor, SetForegroundColor},
    terminal::{self, ClearType},
    ExecutableCommand, QueueableCommand,
};
use std::io::{stdout, Write};

/// RGB color
#[derive(Clone, Copy, Debug, Default)]
pub struct Rgb {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Rgb {
    pub const BLACK: Rgb = Rgb { r: 0, g: 0, b: 0 };
    pub const WHITE: Rgb = Rgb { r: 255, g: 255, b: 255 };
    pub const RED: Rgb = Rgb { r: 255, g: 0, b: 0 };
    pub const GREEN: Rgb = Rgb { r: 0, g: 255, b: 0 };
    pub const BLUE: Rgb = Rgb { r: 0, g: 0, b: 255 };
    pub const CYAN: Rgb = Rgb { r: 0, g: 255, b: 255 };
    pub const MAGENTA: Rgb = Rgb { r: 255, g: 0, b: 255 };
    pub const YELLOW: Rgb = Rgb { r: 255, g: 255, b: 0 };

    pub fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    pub fn to_crossterm(&self) -> Color {
        Color::Rgb { r: self.r, g: self.g, b: self.b }
    }
}

/// Cell in the terminal grid
#[derive(Clone, Debug)]
pub struct Cell {
    pub glyph: char,
    pub fg: Rgb,
    pub bg: Rgb,
    pub heat: f32,
}

impl Default for Cell {
    fn default() -> Self {
        Self {
            glyph: ' ',
            fg: Rgb::WHITE,
            bg: Rgb::BLACK,
            heat: 0.0,
        }
    }
}

/// Terminal renderer
pub struct Renderer {
    width: u16,
    height: u16,
    cells: Vec<Vec<Cell>>,
    in_alternate: bool,
}

impl Renderer {
    pub fn new() -> Result<Self> {
        let (width, height) = terminal::size()?;
        let cells = vec![vec![Cell::default(); width as usize]; height as usize];

        Ok(Self {
            width,
            height,
            cells,
            in_alternate: false,
        })
    }

    /// Enter alternate screen
    pub fn enter_alternate_screen(&mut self) -> Result<()> {
        stdout().execute(terminal::EnterAlternateScreen)?;
        stdout().execute(cursor::Hide)?;
        terminal::enable_raw_mode()?;
        self.in_alternate = true;
        Ok(())
    }

    /// Leave alternate screen
    pub fn leave_alternate_screen(&mut self) -> Result<()> {
        terminal::disable_raw_mode()?;
        stdout().execute(cursor::Show)?;
        stdout().execute(terminal::LeaveAlternateScreen)?;
        self.in_alternate = false;
        Ok(())
    }

    /// Clear screen
    pub fn clear(&mut self) -> Result<()> {
        stdout().execute(terminal::Clear(ClearType::All))?;
        for row in &mut self.cells {
            for cell in row {
                *cell = Cell::default();
            }
        }
        Ok(())
    }

    /// Set a cell
    pub fn set_cell(&mut self, x: u16, y: u16, cell: Cell) {
        if (x as usize) < self.width as usize && (y as usize) < self.height as usize {
            self.cells[y as usize][x as usize] = cell;
        }
    }

    /// Write text at position
    pub fn write_text(&mut self, x: u16, y: u16, text: &str, fg: Rgb, bg: Rgb) {
        for (i, ch) in text.chars().enumerate() {
            self.set_cell(x + i as u16, y, Cell {
                glyph: ch,
                fg,
                bg,
                heat: 0.0,
            });
        }
    }

    /// Write heated text (with importance highlighting)
    pub fn write_heated(&mut self, x: u16, y: u16, text: &str, fg: Rgb, bg: Rgb, heat: f32) {
        for (i, ch) in text.chars().enumerate() {
            self.set_cell(x + i as u16, y, Cell {
                glyph: ch,
                fg,
                bg,
                heat,
            });
        }
    }

    /// Draw progress bar
    pub fn draw_progress(&mut self, x: u16, y: u16, width: u16, progress: f32, fg: Rgb, bg: Rgb) {
        let filled = (width as f32 * progress.clamp(0.0, 1.0)) as u16;

        for i in 0..width {
            let (glyph, cell_fg, cell_bg) = if i < filled {
                ('█', fg, bg)
            } else {
                ('░', Rgb::new(60, 60, 60), bg)
            };

            self.set_cell(x + i, y, Cell {
                glyph,
                fg: cell_fg,
                bg: cell_bg,
                heat: if i < filled { progress } else { 0.0 },
            });
        }
    }

    /// Draw box
    pub fn draw_box(&mut self, x: u16, y: u16, width: u16, height: u16, fg: Rgb, title: Option<&str>) {
        // Corners
        self.set_cell(x, y, Cell { glyph: '┌', fg, bg: Rgb::BLACK, heat: 0.0 });
        self.set_cell(x + width - 1, y, Cell { glyph: '┐', fg, bg: Rgb::BLACK, heat: 0.0 });
        self.set_cell(x, y + height - 1, Cell { glyph: '└', fg, bg: Rgb::BLACK, heat: 0.0 });
        self.set_cell(x + width - 1, y + height - 1, Cell { glyph: '┘', fg, bg: Rgb::BLACK, heat: 0.0 });

        // Horizontal lines
        for i in 1..width - 1 {
            self.set_cell(x + i, y, Cell { glyph: '─', fg, bg: Rgb::BLACK, heat: 0.0 });
            self.set_cell(x + i, y + height - 1, Cell { glyph: '─', fg, bg: Rgb::BLACK, heat: 0.0 });
        }

        // Vertical lines
        for i in 1..height - 1 {
            self.set_cell(x, y + i, Cell { glyph: '│', fg, bg: Rgb::BLACK, heat: 0.0 });
            self.set_cell(x + width - 1, y + i, Cell { glyph: '│', fg, bg: Rgb::BLACK, heat: 0.0 });
        }

        // Title
        if let Some(title) = title {
            let title_x = x + 2;
            self.write_text(title_x, y, &format!(" {} ", title), fg, Rgb::BLACK);
        }
    }

    /// Render to terminal
    pub fn render(&self) -> Result<()> {
        let mut stdout = stdout();

        for (y, row) in self.cells.iter().enumerate() {
            stdout.queue(cursor::MoveTo(0, y as u16))?;

            for cell in row {
                // Apply heat to color
                let fg = if cell.heat > 0.0 {
                    Rgb::new(
                        (cell.fg.r as f32 + (255.0 - cell.fg.r as f32) * cell.heat * 0.5) as u8,
                        cell.fg.g,
                        cell.fg.b,
                    )
                } else {
                    cell.fg
                };

                stdout
                    .queue(SetForegroundColor(fg.to_crossterm()))?
                    .queue(SetBackgroundColor(cell.bg.to_crossterm()))?
                    .queue(Print(cell.glyph))?;
            }
        }

        stdout.flush()?;
        Ok(())
    }

    pub fn width(&self) -> u16 {
        self.width
    }

    pub fn height(&self) -> u16 {
        self.height
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        if self.in_alternate {
            let _ = self.leave_alternate_screen();
        }
    }
}

/// Simple dashboard for OMNI-HIVE status
pub struct Dashboard {
    renderer: Renderer,
}

impl Dashboard {
    pub fn new() -> Result<Self> {
        Ok(Self {
            renderer: Renderer::new()?,
        })
    }

    /// Show OMNI-HIVE status
    pub fn show_status(
        &mut self,
        fitness: f32,
        agents: usize,
        tokens: usize,
        generation: u64,
    ) -> Result<()> {
        self.renderer.enter_alternate_screen()?;
        self.renderer.clear()?;

        // Title
        self.renderer.draw_box(0, 0, 60, 15, Rgb::CYAN, Some("OMNI-HIVE STATUS"));

        // Fitness
        self.renderer.write_text(2, 2, "FITNESS:", Rgb::WHITE, Rgb::BLACK);
        self.renderer.write_heated(
            12, 2,
            &format!("{:.1}%", fitness * 100.0),
            if fitness > 0.8 { Rgb::GREEN } else { Rgb::YELLOW },
            Rgb::BLACK,
            fitness,
        );
        self.renderer.draw_progress(2, 3, 50, fitness, Rgb::GREEN, Rgb::BLACK);

        // Stats
        self.renderer.write_text(2, 5, &format!("Agents:     {}", agents), Rgb::WHITE, Rgb::BLACK);
        self.renderer.write_text(2, 6, &format!("Tokens:     {}", tokens), Rgb::WHITE, Rgb::BLACK);
        self.renderer.write_text(2, 7, &format!("Generation: {}", generation), Rgb::WHITE, Rgb::BLACK);

        // Footer
        self.renderer.write_text(2, 13, "Press any key to exit...", Rgb::new(100, 100, 100), Rgb::BLACK);

        self.renderer.render()?;

        // Wait for key
        crossterm::event::read()?;

        self.renderer.leave_alternate_screen()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb_colors() {
        let color = Rgb::new(128, 64, 32);
        assert_eq!(color.r, 128);
        assert_eq!(color.g, 64);
        assert_eq!(color.b, 32);
    }
}
