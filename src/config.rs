//! Configuration – loaded from `~/.config/weighterm/config.toml`.
//! Falls back gracefully to built-in defaults when the file is absent.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// ── default helpers ─────────────────────────────────────────────────────────

fn d_width() -> u32 { 1024 }
fn d_height() -> u32 { 768 }
fn d_title() -> String { "weighterm".into() }
fn d_opacity() -> f32 { 1.0 }

fn d_font_family() -> String { "monospace".into() }
fn d_font_size() -> f32 { 14.0 }
fn d_ligatures() -> bool { true }

fn d_theme_name() -> String { "night-owl".into() }
fn d_bg() -> String { "#011627".into() }
fn d_fg() -> String { "#d6deeb".into() }
fn d_cursor() -> String { "#80a4c2".into() }
fn d_black() -> String { "#011627".into() }
fn d_red() -> String { "#ef5350".into() }
fn d_green() -> String { "#22da6e".into() }
fn d_yellow() -> String { "#addb67".into() }
fn d_blue() -> String { "#82aaff".into() }
fn d_magenta() -> String { "#c792ea".into() }
fn d_cyan() -> String { "#21c7a8".into() }
fn d_white() -> String { "#d6deeb".into() }
fn d_bright_black() -> String { "#575656".into() }
fn d_bright_red() -> String { "#ef5350".into() }
fn d_bright_green() -> String { "#22da6e".into() }
fn d_bright_yellow() -> String { "#ffeb95".into() }
fn d_bright_blue() -> String { "#82aaff".into() }
fn d_bright_magenta() -> String { "#c792ea".into() }
fn d_bright_cyan() -> String { "#7fdbca".into() }
fn d_bright_white() -> String { "#ffffff".into() }

fn d_scrollback() -> usize { 10_000 }
fn d_scroll_mult() -> f32 { 3.0 }

// ── sub-structs ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WindowConfig {
    #[serde(default = "d_width")]
    pub width: u32,
    #[serde(default = "d_height")]
    pub height: u32,
    #[serde(default = "d_title")]
    pub title: String,
    #[serde(default = "d_opacity")]
    pub opacity: f32,
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            width: d_width(),
            height: d_height(),
            title: d_title(),
            opacity: d_opacity(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FontConfig {
    #[serde(default = "d_font_family")]
    pub family: String,
    #[serde(default = "d_font_size")]
    pub size: f32,
    #[serde(default = "d_ligatures")]
    pub ligatures: bool,
}

impl Default for FontConfig {
    fn default() -> Self {
        Self {
            family: d_font_family(),
            size: d_font_size(),
            ligatures: d_ligatures(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ThemeConfig {
    #[serde(default = "d_theme_name")]
    pub name: String,

    #[serde(default = "d_bg")]       pub background:    String,
    #[serde(default = "d_fg")]       pub foreground:    String,
    #[serde(default = "d_cursor")]   pub cursor:        String,

    #[serde(default = "d_black")]    pub black:         String,
    #[serde(default = "d_red")]      pub red:           String,
    #[serde(default = "d_green")]    pub green:         String,
    #[serde(default = "d_yellow")]   pub yellow:        String,
    #[serde(default = "d_blue")]     pub blue:          String,
    #[serde(default = "d_magenta")]  pub magenta:       String,
    #[serde(default = "d_cyan")]     pub cyan:          String,
    #[serde(default = "d_white")]    pub white:         String,

    #[serde(default = "d_bright_black")]   pub bright_black:   String,
    #[serde(default = "d_bright_red")]     pub bright_red:     String,
    #[serde(default = "d_bright_green")]   pub bright_green:   String,
    #[serde(default = "d_bright_yellow")]  pub bright_yellow:  String,
    #[serde(default = "d_bright_blue")]    pub bright_blue:    String,
    #[serde(default = "d_bright_magenta")] pub bright_magenta: String,
    #[serde(default = "d_bright_cyan")]    pub bright_cyan:    String,
    #[serde(default = "d_bright_white")]   pub bright_white:   String,
}

impl Default for ThemeConfig {
    fn default() -> Self {
        Self {
            name:           d_theme_name(),
            background:     d_bg(),
            foreground:     d_fg(),
            cursor:         d_cursor(),
            black:          d_black(),
            red:            d_red(),
            green:          d_green(),
            yellow:         d_yellow(),
            blue:           d_blue(),
            magenta:        d_magenta(),
            cyan:           d_cyan(),
            white:          d_white(),
            bright_black:   d_bright_black(),
            bright_red:     d_bright_red(),
            bright_green:   d_bright_green(),
            bright_yellow:  d_bright_yellow(),
            bright_blue:    d_bright_blue(),
            bright_magenta: d_bright_magenta(),
            bright_cyan:    d_bright_cyan(),
            bright_white:   d_bright_white(),
        }
    }
}

impl ThemeConfig {
    /// Return the ANSI colour hex string for palette index 0-15.
    pub fn ansi_color(&self, idx: u8) -> &str {
        match idx {
            0  => &self.black,
            1  => &self.red,
            2  => &self.green,
            3  => &self.yellow,
            4  => &self.blue,
            5  => &self.magenta,
            6  => &self.cyan,
            7  => &self.white,
            8  => &self.bright_black,
            9  => &self.bright_red,
            10 => &self.bright_green,
            11 => &self.bright_yellow,
            12 => &self.bright_blue,
            13 => &self.bright_magenta,
            14 => &self.bright_cyan,
            15 => &self.bright_white,
            _  => &self.foreground,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PerformanceConfig {
    #[serde(default = "d_scrollback")]
    pub scrollback_lines: usize,
    #[serde(default = "d_scroll_mult")]
    pub scroll_multiplier: f32,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            scrollback_lines: d_scrollback(),
            scroll_multiplier: d_scroll_mult(),
        }
    }
}

// ── top-level Config ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Config {
    #[serde(default)]
    pub window: WindowConfig,
    #[serde(default)]
    pub font: FontConfig,
    #[serde(default)]
    pub theme: ThemeConfig,
    #[serde(default)]
    pub performance: PerformanceConfig,
}

impl Config {
    /// Load configuration.  Looks in (highest priority first):
    ///   1. `$XDG_CONFIG_HOME/weighterm/config.toml`
    ///   2. `~/.config/weighterm/config.toml`
    ///   3. built-in defaults
    pub fn load() -> Self {
        if let Some(path) = Self::config_path() {
            match std::fs::read_to_string(&path) {
                Ok(text) => match toml::from_str::<Config>(&text) {
                    Ok(cfg) => {
                        log::info!("Loaded config from {}", path.display());
                        return cfg;
                    }
                    Err(e) => {
                        log::warn!("Config parse error in {}: {e}", path.display());
                    }
                },
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => log::warn!("Cannot read {}: {e}", path.display()),
            }
        }
        log::info!("Using built-in default configuration");
        Self::default()
    }

    fn config_path() -> Option<PathBuf> {
        let base = std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| dirs::config_dir())?;
        Some(base.join("weighterm").join("config.toml"))
    }

    /// Parse a `#rrggbb` hex colour string into linear [r, g, b, a] floats.
    pub fn parse_hex_color(hex: &str) -> [f32; 4] {
        let h = hex.trim_start_matches('#');
        if h.len() == 6 {
            let r = u8::from_str_radix(&h[0..2], 16).unwrap_or(0) as f32 / 255.0;
            let g = u8::from_str_radix(&h[2..4], 16).unwrap_or(0) as f32 / 255.0;
            let b = u8::from_str_radix(&h[4..6], 16).unwrap_or(0) as f32 / 255.0;
            [r, g, b, 1.0]
        } else {
            [1.0, 1.0, 1.0, 1.0]
        }
    }
}
