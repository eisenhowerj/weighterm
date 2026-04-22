//! Terminal grid state and VTE escape-sequence performer.
//!
//! `TerminalState` is the primary data structure that both the PTY reader
//! thread and the renderer share.  The PTY thread holds a `vte::Parser` and
//! calls `Performer::advance` for every byte it receives; the renderer reads
//! the grid to build vertex buffers each frame.

use std::collections::VecDeque;
use bitflags::bitflags;
use vte::{Params, Perform};

// ── Cell flags ───────────────────────────────────────────────────────────────

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub struct CellFlags: u8 {
        const BOLD      = 0b0000_0001;
        const DIM       = 0b0000_0010;
        const ITALIC    = 0b0000_0100;
        const UNDERLINE = 0b0000_1000;
        const BLINK     = 0b0001_0000;
        const REVERSE   = 0b0010_0000;
        const HIDDEN    = 0b0100_0000;
        const STRIKE    = 0b1000_0000;
    }
}

// ── Colour representation ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TermColor {
    /// Default terminal foreground / background
    Default,
    /// ANSI 0-15 palette index
    Named(u8),
    /// xterm 256-colour palette index
    Indexed(u8),
    /// 24-bit true colour
    Rgb(u8, u8, u8),
}

impl Default for TermColor {
    fn default() -> Self { Self::Default }
}

// ── Cell ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Cell {
    pub c:     char,
    pub fg:    TermColor,
    pub bg:    TermColor,
    pub flags: CellFlags,
    /// Column-span for wide (e.g. CJK) characters (1 or 2)
    pub width: u8,
}

impl Default for Cell {
    fn default() -> Self {
        Self {
            c:     ' ',
            fg:    TermColor::Default,
            bg:    TermColor::Default,
            flags: CellFlags::empty(),
            width: 1,
        }
    }
}

// ── Cursor ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct Cursor {
    pub col: usize,
    pub row: usize,
    /// Saved cursor position (ESC 7 / ESC 8)
    pub saved_col: usize,
    pub saved_row: usize,
}

// ── TerminalState ────────────────────────────────────────────────────────────

pub struct TerminalState {
    pub cols: usize,
    pub rows: usize,

    /// Active screen grid, row-major: `grid[row * cols + col]`
    pub grid: Vec<Cell>,

    /// Scrollback lines (oldest first)
    pub scrollback: VecDeque<Vec<Cell>>,
    pub scrollback_limit: usize,

    pub cursor: Cursor,

    /// Current SGR pen colours
    pub current_fg: TermColor,
    pub current_bg: TermColor,
    pub current_flags: CellFlags,

    /// Alternate screen active
    pub alt_screen: bool,
    /// Primary screen saved when alt is active
    alt_grid: Vec<Cell>,
    alt_cursor: Cursor,

    /// Scroll region (inclusive row indices)
    scroll_top: usize,
    scroll_bot: usize,

    /// Set when the terminal title is updated
    pub title: Option<String>,

    /// Set to request a PTY resize notification
    pub pending_resize: Option<(u16, u16)>,
}

impl TerminalState {
    pub fn new(cols: usize, rows: usize, scrollback_limit: usize) -> Self {
        let grid = vec![Cell::default(); cols * rows];
        Self {
            cols,
            rows,
            grid,
            scrollback: VecDeque::new(),
            scrollback_limit,
            cursor: Cursor::default(),
            current_fg: TermColor::Default,
            current_bg: TermColor::Default,
            current_flags: CellFlags::empty(),
            alt_screen: false,
            alt_grid: Vec::new(),
            alt_cursor: Cursor::default(),
            scroll_top: 0,
            scroll_bot: rows.saturating_sub(1),
            title: None,
            pending_resize: None,
        }
    }

    // ── Grid accessors ───────────────────────────────────────────────────────

    #[inline]
    pub fn cell(&self, col: usize, row: usize) -> &Cell {
        &self.grid[row * self.cols + col]
    }

    #[inline]
    #[allow(dead_code)]
    fn cell_mut(&mut self, col: usize, row: usize) -> &mut Cell {
        &mut self.grid[row * self.cols + col]
    }

    /// Write `c` at the current cursor, then advance.
    fn put_char(&mut self, c: char) {
        use unicode_width::UnicodeWidthChar;
        let w = c.width().unwrap_or(1).max(1) as u8;

        // For wide characters that would overflow the current line (no room for
        // the continuation cell in the next column), place a blank in the last
        // column and wrap first.  This matches xterm / VTE behaviour.
        if w == 2 && self.cursor.col + 1 >= self.cols {
            let col = self.cursor.col;
            let row = self.cursor.row;
            if col < self.cols && row < self.rows {
                let bg = self.current_bg;
                let idx = row * self.cols + col;
                self.grid[idx] = Cell { bg, ..Default::default() };
            }
            self.cursor.col = 0;
            self.linefeed();
        }

        let (col, row) = (self.cursor.col, self.cursor.row);
        // Copy pen state BEFORE indexing the grid (avoids simultaneous borrows)
        let fg    = self.current_fg;
        let bg    = self.current_bg;
        let flags = self.current_flags;

        if col < self.cols && row < self.rows {
            let idx = row * self.cols + col;
            self.grid[idx].c     = c;
            self.grid[idx].fg    = fg;
            self.grid[idx].bg    = bg;
            self.grid[idx].flags = flags;
            self.grid[idx].width = w;
            // If wide, mark continuation cell
            if w == 2 && col + 1 < self.cols {
                let idx2 = row * self.cols + col + 1;
                self.grid[idx2].c     = '\0';
                self.grid[idx2].fg    = fg;
                self.grid[idx2].bg    = bg;
                self.grid[idx2].flags = flags;
                self.grid[idx2].width = 0;
            }
        }

        self.cursor.col += w as usize;
        if self.cursor.col >= self.cols {
            // Auto-wrap
            self.cursor.col = 0;
            self.linefeed();
        }
    }

    // ── Scrolling ────────────────────────────────────────────────────────────

    fn linefeed(&mut self) {
        if self.cursor.row == self.scroll_bot {
            self.scroll_up(1);
        } else if self.cursor.row + 1 < self.rows {
            self.cursor.row += 1;
        }
    }

    pub fn scroll_up(&mut self, n: usize) {
        let top  = self.scroll_top;
        let bot  = self.scroll_bot;
        let cols = self.cols;
        let height = bot - top + 1;
        if height == 0 { return; }
        let steps = n.min(height);

        // Save scrolled-off lines to scrollback (only on main screen and top == 0)
        if !self.alt_screen && top == 0 {
            for row in top..top + steps {
                let start = row * cols;
                self.scrollback.push_back(self.grid[start..start + cols].to_vec());
                if self.scrollback.len() > self.scrollback_limit {
                    self.scrollback.pop_front();
                }
            }
        }

        // Shift the scroll region up using a rotate (O(n) single move, no per-cell
        // clone loops).
        let start = top * cols;
        let end   = (bot + 1) * cols;
        self.grid[start..end].rotate_left(steps * cols);

        // Clear the newly revealed rows at the bottom of the scroll region,
        // preserving the current background color (Background Color Erase).
        let clear_bg = self.current_bg;
        for row in (bot + 1 - steps)..=bot {
            let s = row * cols;
            for cell in &mut self.grid[s..s + cols] {
                *cell = Cell { bg: clear_bg, ..Default::default() };
            }
        }
    }

    pub fn scroll_down(&mut self, n: usize) {
        let top  = self.scroll_top;
        let bot  = self.scroll_bot;
        let cols = self.cols;
        let height = bot - top + 1;
        if height == 0 { return; }
        let steps = n.min(height);

        let start = top * cols;
        let end   = (bot + 1) * cols;
        self.grid[start..end].rotate_right(steps * cols);

        // Clear the newly revealed rows at the top of the scroll region.
        let clear_bg = self.current_bg;
        for row in top..top + steps {
            let s = row * cols;
            for cell in &mut self.grid[s..s + cols] {
                *cell = Cell { bg: clear_bg, ..Default::default() };
            }
        }
    }

    // ── Erase helpers ────────────────────────────────────────────────────────

    fn erase_display(&mut self, mode: usize) {
        let (cols, rows) = (self.cols, self.rows);
        let (col, row) = (self.cursor.col, self.cursor.row);
        // Use current background colour for erased cells (Background Color Erase).
        let blank = Cell { bg: self.current_bg, ..Default::default() };
        match mode {
            0 => {
                // Erase from cursor to end of screen
                for c in col..cols {
                    self.grid[row * cols + c] = blank.clone();
                }
                for r in (row + 1)..rows {
                    for c in 0..cols {
                        self.grid[r * cols + c] = blank.clone();
                    }
                }
            }
            1 => {
                // Erase from start of screen to cursor
                for r in 0..row {
                    for c in 0..cols {
                        self.grid[r * cols + c] = blank.clone();
                    }
                }
                for c in 0..=col {
                    self.grid[row * cols + c] = blank.clone();
                }
            }
            2 | 3 => {
                // Erase entire screen
                for cell in &mut self.grid {
                    *cell = blank.clone();
                }
                self.cursor.col = 0;
                self.cursor.row = 0;
            }
            _ => {}
        }
    }

    fn erase_line(&mut self, mode: usize) {
        let cols = self.cols;
        let (col, row) = (self.cursor.col, self.cursor.row);
        // Use current background colour for erased cells (Background Color Erase).
        let blank = Cell { bg: self.current_bg, ..Default::default() };
        match mode {
            0 => {
                for c in col..cols {
                    self.grid[row * cols + c] = blank.clone();
                }
            }
            1 => {
                for c in 0..=col {
                    self.grid[row * cols + c] = blank.clone();
                }
            }
            2 => {
                for c in 0..cols {
                    self.grid[row * cols + c] = blank.clone();
                }
            }
            _ => {}
        }
    }

    // ── Resize ───────────────────────────────────────────────────────────────

    pub fn resize(&mut self, new_cols: usize, new_rows: usize) {
        if new_cols == self.cols && new_rows == self.rows {
            return;
        }
        let mut new_grid = vec![Cell::default(); new_cols * new_rows];
        let copy_cols = self.cols.min(new_cols);
        let copy_rows = self.rows.min(new_rows);
        for r in 0..copy_rows {
            for c in 0..copy_cols {
                new_grid[r * new_cols + c] = self.grid[r * self.cols + c].clone();
            }
        }
        self.grid = new_grid;
        self.cols = new_cols;
        self.rows = new_rows;
        self.scroll_top = 0;
        self.scroll_bot = new_rows.saturating_sub(1);
        self.cursor.col = self.cursor.col.min(new_cols.saturating_sub(1));
        self.cursor.row = self.cursor.row.min(new_rows.saturating_sub(1));
        self.pending_resize = Some((new_cols as u16, new_rows as u16));
    }

    // ── Alternate screen ─────────────────────────────────────────────────────

    fn enter_alt_screen(&mut self) {
        if !self.alt_screen {
            self.alt_grid   = self.grid.clone();
            self.alt_cursor = self.cursor.clone();
            for cell in &mut self.grid {
                *cell = Cell::default();
            }
            self.cursor = Cursor::default();
            self.alt_screen = true;
        }
    }

    fn leave_alt_screen(&mut self) {
        if self.alt_screen {
            self.grid   = self.alt_grid.clone();
            self.cursor = self.alt_cursor.clone();
            self.alt_screen = false;
        }
    }

    // ── DEC private-mode handling ────────────────────────────────────────────

    pub fn handle_dec_mode(&mut self, mode: usize, enable: bool) {
        match mode {
            25 => { /* show/hide cursor – TODO: expose field */ }
            47 | 1047 | 1049 => {
                if enable { self.enter_alt_screen(); } else { self.leave_alt_screen(); }
            }
            _ => {}
        }
    }

    // ── SGR (Select Graphic Rendition) ───────────────────────────────────────

    fn apply_sgr(&mut self, params: &Params) {
        let mut iter = params.iter();
        loop {
            let sub = match iter.next() {
                Some(s) => s,
                None    => break,
            };
            let p = sub.first().copied().unwrap_or(0);
            match p {
                0  => {
                    self.current_fg    = TermColor::Default;
                    self.current_bg    = TermColor::Default;
                    self.current_flags = CellFlags::empty();
                }
                1  => { self.current_flags.insert(CellFlags::BOLD); }
                2  => { self.current_flags.insert(CellFlags::DIM); }
                3  => { self.current_flags.insert(CellFlags::ITALIC); }
                4  => { self.current_flags.insert(CellFlags::UNDERLINE); }
                5 | 6 => { self.current_flags.insert(CellFlags::BLINK); }
                7  => { self.current_flags.insert(CellFlags::REVERSE); }
                8  => { self.current_flags.insert(CellFlags::HIDDEN); }
                9  => { self.current_flags.insert(CellFlags::STRIKE); }
                22 => { self.current_flags.remove(CellFlags::BOLD | CellFlags::DIM); }
                23 => { self.current_flags.remove(CellFlags::ITALIC); }
                24 => { self.current_flags.remove(CellFlags::UNDERLINE); }
                25 => { self.current_flags.remove(CellFlags::BLINK); }
                27 => { self.current_flags.remove(CellFlags::REVERSE); }
                28 => { self.current_flags.remove(CellFlags::HIDDEN); }
                29 => { self.current_flags.remove(CellFlags::STRIKE); }
                30..=37 => { self.current_fg = TermColor::Named(p as u8 - 30); }
                38 => { self.current_fg = self.parse_color_ext(&mut iter); }
                39 => { self.current_fg = TermColor::Default; }
                40..=47 => { self.current_bg = TermColor::Named(p as u8 - 40); }
                48 => { self.current_bg = self.parse_color_ext(&mut iter); }
                49 => { self.current_bg = TermColor::Default; }
                90..=97  => { self.current_fg = TermColor::Named(p as u8 - 90 + 8); }
                100..=107 => { self.current_bg = TermColor::Named(p as u8 - 100 + 8); }
                _ => {}
            }
        }
    }

    fn parse_color_ext<'a>(
        &self,
        iter: &mut vte::ParamsIter<'a>,
    ) -> TermColor {
        match iter.next().and_then(|s| s.first().copied()) {
            Some(2) => {
                let r = iter.next().and_then(|s| s.first().copied()).unwrap_or(0);
                let g = iter.next().and_then(|s| s.first().copied()).unwrap_or(0);
                let b = iter.next().and_then(|s| s.first().copied()).unwrap_or(0);
                TermColor::Rgb(r as u8, g as u8, b as u8)
            }
            Some(5) => {
                let idx = iter.next().and_then(|s| s.first().copied()).unwrap_or(0);
                TermColor::Indexed(idx as u8)
            }
            _ => TermColor::Default,
        }
    }
}

// ── VTE performer ────────────────────────────────────────────────────────────

/// Wraps `&mut TerminalState` and implements `vte::Perform`.
pub struct Performer<'a>(pub &'a mut TerminalState);

impl<'a> Perform for Performer<'a> {
    fn print(&mut self, c: char) {
        self.0.put_char(c);
    }

    fn execute(&mut self, byte: u8) {
        match byte {
            b'\n' | b'\x0C' | b'\x0B' => {
                self.0.linefeed();
            }
            b'\r' => {
                self.0.cursor.col = 0;
            }
            b'\x08' => {
                // Backspace
                if self.0.cursor.col > 0 {
                    self.0.cursor.col -= 1;
                }
            }
            b'\x07' => { /* Bell – ignored */ }
            b'\t' => {
                // Tab stop every 8 columns
                self.0.cursor.col = ((self.0.cursor.col / 8) + 1) * 8;
                self.0.cursor.col = self.0.cursor.col.min(self.0.cols - 1);
            }
            _ => {}
        }
    }

    fn csi_dispatch(&mut self, params: &Params, intermediates: &[u8], _ignore: bool, action: char) {
        let mut p_iter = params.iter();
        let p1 = p_iter.next().and_then(|v| v.first().copied()).unwrap_or(0) as usize;
        let p2 = p_iter.next().and_then(|v| v.first().copied()).unwrap_or(0) as usize;

        // DEC private-mode sequences: ESC [ ? Pm h/l
        if intermediates.contains(&b'?') {
            let enable = action == 'h';
            self.0.handle_dec_mode(p1, enable);
            return;
        }

        match action {
            // Cursor Up / Down / Forward / Back
            'A' => { self.0.cursor.row = self.0.cursor.row.saturating_sub(p1.max(1)); }
            'B' => { self.0.cursor.row = (self.0.cursor.row + p1.max(1)).min(self.0.rows - 1); }
            'C' => { self.0.cursor.col = (self.0.cursor.col + p1.max(1)).min(self.0.cols - 1); }
            'D' => { self.0.cursor.col = self.0.cursor.col.saturating_sub(p1.max(1)); }
            'E' => {
                self.0.cursor.row = (self.0.cursor.row + p1.max(1)).min(self.0.rows - 1);
                self.0.cursor.col = 0;
            }
            'F' => {
                self.0.cursor.row = self.0.cursor.row.saturating_sub(p1.max(1));
                self.0.cursor.col = 0;
            }
            'G' => { self.0.cursor.col = p1.saturating_sub(1).min(self.0.cols - 1); }
            'H' | 'f' => {
                // CUP – Cursor Position (1-based; 0 treated as 1)
                let row = if p1 == 0 { 0 } else { (p1 - 1).min(self.0.rows - 1) };
                let col = if p2 == 0 { 0 } else { (p2 - 1).min(self.0.cols - 1) };
                self.0.cursor.row = row;
                self.0.cursor.col = col;
            }
            'J' => { self.0.erase_display(p1); }
            'K' => { self.0.erase_line(p1); }
            'L' => {
                // IL – Insert Line: insert n blank lines at the cursor row,
                // shifting existing lines down within the scroll region.
                let cur_row = self.0.cursor.row;
                let top     = self.0.scroll_top;
                let bot     = self.0.scroll_bot;
                if cur_row >= top && cur_row <= bot {
                    let n    = p1.max(1).min(bot - cur_row + 1);
                    let cols = self.0.cols;
                    for r in (cur_row..=bot).rev() {
                        for c in 0..cols {
                            self.0.grid[r * cols + c] = if r >= cur_row + n {
                                self.0.grid[(r - n) * cols + c].clone()
                            } else {
                                Cell::default()
                            };
                        }
                    }
                }
            }
            'M' => {
                // DL – Delete Line: delete n lines at the cursor row,
                // shifting existing lines up within the scroll region.
                let cur_row = self.0.cursor.row;
                let top     = self.0.scroll_top;
                let bot     = self.0.scroll_bot;
                if cur_row >= top && cur_row <= bot {
                    let n    = p1.max(1).min(bot - cur_row + 1);
                    let cols = self.0.cols;
                    for r in cur_row..=bot {
                        for c in 0..cols {
                            self.0.grid[r * cols + c] = if r + n <= bot {
                                self.0.grid[(r + n) * cols + c].clone()
                            } else {
                                Cell::default()
                            };
                        }
                    }
                }
            }
            'P' => {
                // DCH – Delete Character
                let n   = p1.max(1);
                let row = self.0.cursor.row;
                let col = self.0.cursor.col;
                let cols = self.0.cols;
                for c in col..cols {
                    self.0.grid[row * cols + c] = if c + n < cols {
                        self.0.grid[row * cols + c + n].clone()
                    } else {
                        Cell::default()
                    };
                }
            }
            '@' => {
                // ICH – Insert Character
                let n    = p1.max(1);
                let row  = self.0.cursor.row;
                let col  = self.0.cursor.col;
                let cols = self.0.cols;
                for c in (col..cols).rev() {
                    self.0.grid[row * cols + c] = if c >= col + n {
                        self.0.grid[row * cols + c - n].clone()
                    } else {
                        Cell::default()
                    };
                }
            }
            'S' => { self.0.scroll_up(p1.max(1)); }
            'T' => { self.0.scroll_down(p1.max(1)); }
            'd' => { self.0.cursor.row = p1.saturating_sub(1).min(self.0.rows - 1); }
            'h' | 'l' => { /* SM / RM public modes – ignored */ }
            'm' => { self.0.apply_sgr(params); }
            'r' => {
                // DECSTBM – Set Scroll Region
                let top = p1.saturating_sub(1).min(self.0.rows - 1);
                let bot = if p2 == 0 { self.0.rows - 1 } else { (p2 - 1).min(self.0.rows - 1) };
                if top < bot {
                    self.0.scroll_top = top;
                    self.0.scroll_bot = bot;
                }
                self.0.cursor.col = 0;
                self.0.cursor.row = 0;
            }
            's' => {
                self.0.cursor.saved_col = self.0.cursor.col;
                self.0.cursor.saved_row = self.0.cursor.row;
            }
            'u' => {
                self.0.cursor.col = self.0.cursor.saved_col;
                self.0.cursor.row = self.0.cursor.saved_row;
            }
            _ => {}
        }
    }

    fn esc_dispatch(&mut self, intermediates: &[u8], _ignore: bool, byte: u8) {
        match (intermediates.first(), byte) {
            (None, b'7') => {
                self.0.cursor.saved_col = self.0.cursor.col;
                self.0.cursor.saved_row = self.0.cursor.row;
            }
            (None, b'8') => {
                self.0.cursor.col = self.0.cursor.saved_col;
                self.0.cursor.row = self.0.cursor.saved_row;
            }
            (None, b'D') => { self.0.linefeed(); }
            (None, b'E') => {
                self.0.cursor.col = 0;
                self.0.linefeed();
            }
            (None, b'M') => {
                // Reverse index
                if self.0.cursor.row == self.0.scroll_top {
                    self.0.scroll_down(1);
                } else if self.0.cursor.row > 0 {
                    self.0.cursor.row -= 1;
                }
            }
            (None, b'c') => {
                // RIS – full reset
                let cols            = self.0.cols;
                let rows            = self.0.rows;
                let scrollback_limit = self.0.scrollback_limit;
                *self.0 = TerminalState::new(cols, rows, scrollback_limit);
            }
            _ => {}
        }
    }

    fn osc_dispatch(&mut self, params: &[&[u8]], _bell_terminated: bool) {
        // OSC 0 or OSC 2 – set window title
        if params.len() >= 2 && (params[0] == b"0" || params[0] == b"2") {
            if let Ok(title) = std::str::from_utf8(params[1]) {
                self.0.title = Some(title.to_owned());
            }
        }
    }

    fn hook(&mut self, _params: &Params, _intermediates: &[u8], _ignore: bool, _action: char) {}
    fn put(&mut self, _byte: u8) {}
    fn unhook(&mut self) {}
}

/// Convenience: process a chunk of PTY bytes against the terminal state.
pub fn process_bytes(state: &mut TerminalState, parser: &mut vte::Parser, data: &[u8]) {
    let mut perf = Performer(state);
    for &b in data {
        parser.advance(&mut perf, b);
    }
}

