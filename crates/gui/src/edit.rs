//! Background per-block edit for the inspector.
//!
//! One tensor is re-encoded to the requested type while every other tensor is
//! copied byte-for-byte by [`rustblock_core::retarget_tensors`]. The work runs
//! on its own thread so a multi-gigabyte copy never blocks the window.

use std::path::PathBuf;
use std::sync::mpsc::Sender;
use std::thread::{self, JoinHandle};

use rustblock_core::{retarget_tensors, Phase, Progress, Report, Target};

#[derive(Debug)]
pub enum EditEvent {
    Progress {
        done: u64,
        total: u64,
        label: String,
    },
    Done {
        path: PathBuf,
        report: Report,
    },
    Failed(String),
}

pub fn spawn(
    input: PathBuf,
    output: PathBuf,
    tensor: String,
    target: Target,
    tx: Sender<EditEvent>,
) -> JoinHandle<()> {
    thread::spawn(move || {
        let changes = vec![(tensor, target)];
        let mut progress = |p: &Progress| {
            let phase = match p.phase {
                Phase::Converting => "Converting",
                Phase::Verifying => "Verifying",
            };
            let label = if p.tensor_name.is_empty() {
                phase.to_string()
            } else {
                format!("{phase} · {}", p.tensor_name)
            };
            let _ = tx.send(EditEvent::Progress {
                done: p.bytes_done,
                total: p.bytes_total,
                label,
            });
        };
        match retarget_tensors(&input, &output, &changes, Some(&mut progress)) {
            Ok(report) => {
                let _ = tx.send(EditEvent::Done {
                    path: output,
                    report,
                });
            }
            Err(error) => {
                let _ = tx.send(EditEvent::Failed(error.to_string()));
            }
        }
    })
}
