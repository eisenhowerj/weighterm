//! Pseudo-terminal management.
//!
//! Opens a PTY pair, spawns the user's shell inside the slave end, and
//! ferries bytes between the master FD and the rest of the application via
//! a pair of crossbeam channels:
//!
//!  - **output channel**: PTY → application (bytes to feed the VTE parser)
//!  - **input channel**:  application → PTY (keyboard / paste bytes)

use std::io::{Read, Write};
use std::sync::Arc;
use crossbeam_channel::{Receiver, Sender};
use portable_pty::{native_pty_system, CommandBuilder, PtySize};

/// Handle through which the main thread communicates with the PTY.
pub struct PtyHandle {
    /// Bytes read from the shell arrive here.
    pub output_rx: Receiver<Vec<u8>>,
    /// Bytes to write to the shell go here.
    pub input_tx: Sender<Vec<u8>>,
    /// PTY master – kept alive for resize operations.
    master: Box<dyn portable_pty::MasterPty + Send>,
}

impl PtyHandle {
    /// Resize the PTY (called when the window / font size changes).
    pub fn resize(&self, cols: u16, rows: u16) {
        let _ = self.master.resize(PtySize {
            rows,
            cols,
            pixel_width:  0,
            pixel_height: 0,
        });
    }

    /// Write bytes to the PTY (key input, paste, etc.).
    pub fn write(&self, data: Vec<u8>) {
        let _ = self.input_tx.send(data);
    }
}

/// Spawn a PTY with the user's shell running inside it.
///
/// Returns a `PtyHandle` for the caller and a `ChildExitHandle` that resolves
/// when the child exits.
pub fn spawn(cols: u16, rows: u16) -> Result<(PtyHandle, Arc<std::sync::Mutex<Option<u32>>>), Box<dyn std::error::Error>> {
    let pty_system = native_pty_system();

    let pair = pty_system.openpty(PtySize {
        rows,
        cols,
        pixel_width:  0,
        pixel_height: 0,
    })?;

    // Determine shell from $SHELL env var
    let shell = std::env::var("SHELL").unwrap_or_else(|_| "/bin/sh".into());
    let mut cmd = CommandBuilder::new(&shell);
    cmd.env("TERM",          "xterm-256color");
    cmd.env("TERM_PROGRAM",  "weighterm");
    cmd.env("COLORTERM",     "truecolor");

    let _child = pair.slave.spawn_command(cmd)?;

    // ── Output: master → application ────────────────────────────────────────
    let mut reader   = pair.master.try_clone_reader()?;
    let (out_tx, out_rx) = crossbeam_channel::unbounded::<Vec<u8>>();

    std::thread::Builder::new()
        .name("pty-reader".into())
        .spawn(move || {
            let mut buf = [0u8; 4096];
            loop {
                match reader.read(&mut buf) {
                    Ok(0) | Err(_) => break,
                    Ok(n)          => {
                        if out_tx.send(buf[..n].to_vec()).is_err() {
                            break;
                        }
                    }
                }
            }
            log::info!("PTY reader thread exiting");
        })?;

    // ── Input: application → master ─────────────────────────────────────────
    let mut writer = pair.master.take_writer()?;
    let (in_tx, in_rx) = crossbeam_channel::unbounded::<Vec<u8>>();

    std::thread::Builder::new()
        .name("pty-writer".into())
        .spawn(move || {
            for data in in_rx {
                if writer.write_all(&data).is_err() {
                    break;
                }
            }
            log::info!("PTY writer thread exiting");
        })?;

    let exit_status = Arc::new(std::sync::Mutex::new(None::<u32>));

    Ok((
        PtyHandle {
            output_rx: out_rx,
            input_tx:  in_tx,
            master:    pair.master,
        },
        exit_status,
    ))
}
