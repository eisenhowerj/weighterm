//! weighterm – entry point and winit event-loop application.
//!
//! Window lifecycle
//! ────────────────
//!   Event::Resumed          – create the OS window + wgpu renderer + PTY
//!   Event::WindowEvent      – handle resize / keyboard / close
//!   Event::RedrawRequested  – GPU render frame
//!   Event::UserEvent        – receive PTY output bytes on the main thread
//!
//! Window controls
//! ───────────────
//!   • Ctrl+Shift+Q / clicking the title-bar ×  → close
//!   • Super+Up / F11                            → maximise toggle
//!   • Super+Down                                → minimise
//!   • Alt+F4                                    → close

use std::sync::{Arc, Mutex};
use winit::event::{ElementState, Event, KeyEvent, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoopBuilder};
use winit::keyboard::{Key, ModifiersState, NamedKey};
use winit::window::{Window, WindowBuilder, WindowButtons};

mod config;
mod pty;
mod renderer;
mod terminal;

use config::Config;
use renderer::Renderer;
use terminal::{process_bytes, TerminalState};

// ── Custom event type ────────────────────────────────────────────────────────

#[derive(Debug)]
enum AppEvent {
    /// A chunk of bytes arrived from the PTY.
    PtyData(Vec<u8>),
    /// The PTY child process has exited.
    PtyExit,
}

// ── main ──────────────────────────────────────────────────────────────────────

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("warn,weighterm=info"),
    )
    .init();

    let cfg = Arc::new(Config::load());

    let event_loop = EventLoopBuilder::<AppEvent>::with_user_event()
        .build()
        .expect("Failed to create event loop");

    let proxy = event_loop.create_proxy();

    // Mutable application state (all owned by the event-loop closure)
    let mut window: Option<Arc<Window>> = None;
    let mut renderer: Option<Renderer> = None;
    let terminal = Arc::new(Mutex::new(TerminalState::new(
        80,
        24,
        cfg.performance.scrollback_lines,
    )));
    let mut vte_parser = vte::Parser::new();
    let mut pty_handle: Option<pty::PtyHandle> = None;
    let mut modifiers = ModifiersState::empty();
    let mut maximised = false;

    let cfg_clone = Arc::clone(&cfg);

    // winit 0.29: closure takes (event, &EventLoopWindowTarget), no ControlFlow param.
    // Exit via elwt.exit(), flow control via elwt.set_control_flow().
    let _ = event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Wait);

        match event {
            // ── Window creation ───────────────────────────────────────────────
            Event::Resumed => {
                if window.is_some() {
                    return;
                }

                let win = WindowBuilder::new()
                    .with_title(&cfg_clone.window.title)
                    .with_inner_size(winit::dpi::PhysicalSize::new(
                        cfg_clone.window.width,
                        cfg_clone.window.height,
                    ))
                    .with_enabled_buttons(
                        WindowButtons::CLOSE | WindowButtons::MINIMIZE | WindowButtons::MAXIMIZE,
                    )
                    .build(elwt)
                    .expect("Failed to create window");

                let win = Arc::new(win);

                match pollster::block_on(Renderer::new(Arc::clone(&win), &cfg_clone)) {
                    Ok(r) => {
                        renderer = Some(r);
                    }
                    Err(e) => {
                        log::error!("Renderer init failed: {e}");
                        elwt.exit();
                        return;
                    }
                }

                // Resize terminal to match the renderer's grid size
                if let Some(ref r) = renderer {
                    let (cols, rows) = r.grid_size();
                    if let Ok(mut term) = terminal.lock() {
                        term.resize(cols, rows);
                        // The PTY is about to be spawned with these exact dimensions,
                        // so clear the pending_resize flag to avoid a redundant
                        // TIOCSWINSZ call on the first PTY data event.
                        term.pending_resize = None;
                    }
                    // Spawn PTY with the correct dimensions
                    let proxy2 = proxy.clone();
                    match pty::spawn(cols as u16, rows as u16) {
                        Ok(handle) => {
                            let rx = handle.output_rx.clone();
                            std::thread::Builder::new()
                                .name("pty-forwarder".into())
                                .spawn(move || {
                                    for data in rx {
                                        if proxy2.send_event(AppEvent::PtyData(data)).is_err() {
                                            break;
                                        }
                                    }
                                    let _ = proxy2.send_event(AppEvent::PtyExit);
                                })
                                .unwrap();
                            pty_handle = Some(handle);
                        }
                        Err(e) => log::error!("PTY spawn failed: {e}"),
                    }
                }

                window = Some(win);
                if let Some(ref w) = window {
                    w.request_redraw();
                }
            }

            // ── Window events ─────────────────────────────────────────────────
            Event::WindowEvent { event, .. } => {
                match event {
                    WindowEvent::CloseRequested => {
                        log::info!("Window close requested");
                        elwt.exit();
                    }

                    WindowEvent::Resized(size) => {
                        if let Some(ref mut r) = renderer {
                            r.resize(size.width, size.height);
                            let (cols, rows) = r.grid_size();
                            if let Ok(mut term) = terminal.lock() {
                                term.resize(cols, rows);
                            }
                            if let Some(ref h) = pty_handle {
                                h.resize(cols as u16, rows as u16);
                            }
                        }
                        if let Some(ref w) = window {
                            w.request_redraw();
                        }
                    }

                    WindowEvent::ModifiersChanged(mods) => {
                        modifiers = mods.state();
                    }

                    WindowEvent::KeyboardInput { event, .. } => {
                        handle_key(
                            &event,
                            modifiers,
                            &window,
                            &pty_handle,
                            &mut maximised,
                            elwt,
                        );
                    }

                    WindowEvent::MouseWheel { delta, .. } => {
                        let lines = match delta {
                            MouseScrollDelta::LineDelta(_, y) => y,
                            MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 20.0,
                        };
                        let n = (lines.abs() * cfg_clone.performance.scroll_multiplier).max(1.0)
                            as usize;
                        // TODO: implement scrollback viewport – update terminal
                        // scroll offset and re-render the correct slice.
                        if lines > 0.0 {
                            log::debug!("Scroll up {n} (scrollback not yet implemented)");
                        } else {
                            log::debug!("Scroll down {n} (scrollback not yet implemented)");
                        }
                        if let Some(ref w) = window {
                            w.request_redraw();
                        }
                    }

                    // In winit 0.29, RedrawRequested is a WindowEvent
                    WindowEvent::RedrawRequested => {
                        if let (Some(ref mut r), Ok(term)) = (&mut renderer, terminal.lock()) {
                            match r.render(&term, &cfg_clone) {
                                Ok(()) => {}
                                Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                                    r.resize(r.width, r.height);
                                }
                                Err(wgpu::SurfaceError::OutOfMemory) => {
                                    log::error!("wgpu out of memory");
                                    elwt.exit();
                                }
                                Err(e) => log::warn!("Render error: {e:?}"),
                            }
                        }
                    }

                    _ => {}
                }
            }

            // ── PTY bytes ─────────────────────────────────────────────────────
            Event::UserEvent(AppEvent::PtyData(data)) => {
                if let Ok(mut term) = terminal.lock() {
                    process_bytes(&mut term, &mut vte_parser, &data);
                    if let (Some(ref h), Some((cols, rows))) =
                        (&pty_handle, term.pending_resize.take())
                    {
                        h.resize(cols, rows);
                    }
                    if let (Some(ref w), Some(title)) = (&window, term.title.take()) {
                        w.set_title(&title);
                    }
                }
                if let Some(ref w) = window {
                    w.request_redraw();
                }
            }

            Event::UserEvent(AppEvent::PtyExit) => {
                log::info!("Shell exited – closing");
                elwt.exit();
            }

            _ => {}
        }
    });
}

// ── Keyboard → PTY bytes ──────────────────────────────────────────────────────

fn handle_key(
    event: &KeyEvent,
    modifiers: ModifiersState,
    window: &Option<Arc<Window>>,
    pty_handle: &Option<pty::PtyHandle>,
    maximised: &mut bool,
    elwt: &winit::event_loop::EventLoopWindowTarget<AppEvent>,
) {
    if event.state != ElementState::Pressed {
        return;
    }

    let ctrl = modifiers.control_key();
    let shift = modifiers.shift_key();
    let alt = modifiers.alt_key();
    let logo = modifiers.super_key();

    // ── Window-control shortcuts ──────────────────────────────────────────────
    match &event.logical_key {
        Key::Named(NamedKey::F11) => {
            if let Some(w) = window {
                *maximised = !*maximised;
                w.set_maximized(*maximised);
            }
            return;
        }
        Key::Named(NamedKey::F4) if alt => {
            elwt.exit();
            return;
        }
        Key::Named(NamedKey::ArrowUp) if logo => {
            if let Some(w) = window {
                *maximised = !*maximised;
                w.set_maximized(*maximised);
            }
            return;
        }
        Key::Named(NamedKey::ArrowDown) if logo => {
            if let Some(w) = window {
                w.set_minimized(true);
            }
            return;
        }
        Key::Character(s) if ctrl && shift => {
            match s.as_str() {
                "q" | "Q" => {
                    elwt.exit();
                    return;
                }
                "v" | "V" => {
                    // Paste – TODO: integrate clipboard crate
                    return;
                }
                _ => {}
            }
        }
        _ => {}
    }

    // ── Translate to PTY bytes ────────────────────────────────────────────────
    let bytes: Vec<u8> = match &event.logical_key {
        Key::Character(s) => {
            if ctrl {
                if let Some(c) = s.chars().next() {
                    let upper = c.to_ascii_uppercase();
                    if upper.is_ascii_alphabetic() {
                        vec![upper as u8 - b'@']
                    } else {
                        match upper {
                            '[' | '3' => vec![0x1b],
                            '\\' | '4' => vec![0x1c],
                            ']' | '5' => vec![0x1d],
                            '^' | '6' => vec![0x1e],
                            '_' | '7' | '/' => vec![0x1f],
                            ' ' | '2' => vec![0x00],
                            _ => s.as_bytes().to_vec(),
                        }
                    }
                } else {
                    vec![]
                }
            } else if alt {
                let mut v = vec![0x1b];
                v.extend_from_slice(s.as_bytes());
                v
            } else {
                s.as_bytes().to_vec()
            }
        }
        Key::Named(named) => named_key_to_bytes(named, ctrl, alt),
        _ => vec![],
    };

    if !bytes.is_empty() {
        if let Some(h) = pty_handle {
            h.write(bytes);
        }
    }
}

// ── Named-key → escape-sequence lookup ───────────────────────────────────────

fn named_key_to_bytes(key: &NamedKey, ctrl: bool, alt: bool) -> Vec<u8> {
    let esc = |s: &str| -> Vec<u8> {
        let mut v = vec![0x1b];
        v.extend_from_slice(s.as_bytes());
        v
    };
    let maybe_alt = |bytes: Vec<u8>| -> Vec<u8> {
        if alt {
            let mut v = vec![0x1b];
            v.extend_from_slice(&bytes);
            v
        } else {
            bytes
        }
    };

    match key {
        NamedKey::Enter => maybe_alt(vec![b'\r']),
        NamedKey::Backspace => {
            if ctrl {
                vec![0x08]
            } else {
                maybe_alt(vec![0x7f])
            }
        }
        NamedKey::Escape => vec![0x1b],
        NamedKey::Tab => {
            if ctrl {
                esc("[Z")
            } else {
                maybe_alt(vec![b'\t'])
            }
        }
        NamedKey::ArrowUp => esc("[A"),
        NamedKey::ArrowDown => esc("[B"),
        NamedKey::ArrowRight => esc("[C"),
        NamedKey::ArrowLeft => esc("[D"),
        NamedKey::Home => esc("[H"),
        NamedKey::End => esc("[F"),
        NamedKey::Insert => esc("[2~"),
        NamedKey::Delete => esc("[3~"),
        NamedKey::PageUp => esc("[5~"),
        NamedKey::PageDown => esc("[6~"),
        NamedKey::F1 => esc("OP"),
        NamedKey::F2 => esc("OQ"),
        NamedKey::F3 => esc("OR"),
        NamedKey::F4 => esc("OS"),
        NamedKey::F5 => esc("[15~"),
        NamedKey::F6 => esc("[17~"),
        NamedKey::F7 => esc("[18~"),
        NamedKey::F8 => esc("[19~"),
        NamedKey::F9 => esc("[20~"),
        NamedKey::F10 => esc("[21~"),
        NamedKey::F11 => esc("[23~"),
        NamedKey::F12 => esc("[24~"),
        _ => vec![],
    }
}
