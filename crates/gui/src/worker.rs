//! Background conversion/verification worker.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use rustblock_core::{
    convert, transcode, ConvertOptions, Gguf, Phase, Progress, Report, Target, VerifyMode,
};

use crate::model::{InstallResult, InstallTarget, VerifyKind, VerifyOutcome};
use crate::targets;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkKind {
    Convert,
    Verify,
}

#[derive(Debug, Clone)]
pub struct WorkItem {
    pub id: u64,
    pub kind: WorkKind,
    pub input: PathBuf,
    pub output: PathBuf,
    pub target: Target,
    pub verify: VerifyMode,
    pub force: bool,
    pub install: InstallTarget,
}

#[derive(Debug)]
pub enum Event {
    Started {
        id: u64,
    },
    Progress {
        id: u64,
        phase: Phase,
        done: u64,
        total: u64,
        tensor_index: usize,
        tensor_name: String,
        tensors_total: usize,
    },
    Done {
        id: u64,
        report: Report,
        target: Target,
        install: Option<InstallResult>,
    },
    Verified {
        id: u64,
        outcome: VerifyOutcome,
        elapsed: Duration,
    },
    Failed {
        id: u64,
        error: String,
    },
    BatchDone {
        converted: usize,
        verified: usize,
        failed: usize,
        elapsed: Duration,
    },
}

pub fn spawn(items: Vec<WorkItem>, cancel: Arc<AtomicBool>) -> (Receiver<Event>, JoinHandle<()>) {
    let (tx, rx) = mpsc::channel();
    let handle = thread::spawn(move || run_batch(items, tx, cancel));
    (rx, handle)
}

fn run_batch(items: Vec<WorkItem>, tx: Sender<Event>, cancel: Arc<AtomicBool>) {
    let started = Instant::now();
    let mut converted = 0usize;
    let mut verified = 0usize;
    let mut failed = 0usize;

    for item in items {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        if tx.send(Event::Started { id: item.id }).is_err() {
            return;
        }
        let item_started = Instant::now();

        match item.kind {
            WorkKind::Convert => {
                let mut progress = |p: &Progress| send_progress(&tx, item.id, p);
                match transcode(
                    &item.input,
                    &item.output,
                    item.target,
                    ConvertOptions {
                        force: item.force,
                        dry_run: false,
                        verify: item.verify,
                    },
                    Some(&mut progress),
                ) {
                    Ok(report) => {
                        let install = if matches!(item.install, InstallTarget::None) {
                            None
                        } else {
                            Some(
                                match targets::install_model(
                                    &item.install,
                                    &item.output,
                                    &item
                                        .output
                                        .file_name()
                                        .map(|n| n.to_string_lossy().into_owned())
                                        .unwrap_or_else(|| "model.gguf".to_string()),
                                ) {
                                    Ok(message) => InstallResult::Installed(message),
                                    Err(error) => InstallResult::Failed(error),
                                },
                            )
                        };
                        converted += 1;
                        if tx
                            .send(Event::Done {
                                id: item.id,
                                report,
                                target: item.target,
                                install,
                            })
                            .is_err()
                        {
                            return;
                        }
                    }
                    Err(error) => {
                        failed += 1;
                        if tx
                            .send(Event::Failed {
                                id: item.id,
                                error: error.to_string(),
                            })
                            .is_err()
                        {
                            return;
                        }
                    }
                }
            }
            WorkKind::Verify => match verify_item(&item, &tx) {
                Ok(outcome) => {
                    verified += 1;
                    if tx
                        .send(Event::Verified {
                            id: item.id,
                            outcome,
                            elapsed: item_started.elapsed(),
                        })
                        .is_err()
                    {
                        return;
                    }
                }
                Err(error) => {
                    failed += 1;
                    if tx.send(Event::Failed { id: item.id, error }).is_err() {
                        return;
                    }
                }
            },
        }
    }

    let _ = tx.send(Event::BatchDone {
        converted,
        verified,
        failed,
        elapsed: started.elapsed(),
    });
}

fn verify_item(item: &WorkItem, tx: &Sender<Event>) -> Result<VerifyOutcome, String> {
    let g = Gguf::open(&item.input).map_err(|e| e.to_string())?;

    if g.has_ptq1() {
        return Ok(VerifyOutcome {
            tensors: g.tensors.len(),
            kind: VerifyKind::Structure,
            label: "PTQ1_0 is structurally readable but cannot be converted losslessly".to_string(),
        });
    }

    if g.pq2_count() > 0 {
        let dummy = item.input.with_extension("verify.gguf");
        let mut progress = |p: &Progress| send_progress(tx, item.id, p);
        let report = convert(
            &item.input,
            &dummy,
            ConvertOptions {
                dry_run: true,
                verify: VerifyMode::Full,
                ..Default::default()
            },
            Some(&mut progress),
        )
        .map_err(|e| e.to_string())?;
        return Ok(VerifyOutcome {
            tensors: report.tensors_pq2,
            kind: VerifyKind::Exact,
            label: format!(
                "Lossless reblock verified for {} PQ2_0 tensors",
                report.tensors_pq2
            ),
        });
    }

    Ok(VerifyOutcome {
        tensors: g.tensors.len(),
        kind: VerifyKind::Structure,
        label: format!(
            "Header, tensor table, offsets and alignment are valid ({} tensors)",
            g.tensors.len()
        ),
    })
}

fn send_progress(tx: &Sender<Event>, id: u64, p: &Progress) {
    let _ = tx.send(Event::Progress {
        id,
        phase: p.phase,
        done: p.bytes_done,
        total: p.bytes_total,
        tensor_index: p.tensor_index,
        tensor_name: p.tensor_name.clone(),
        tensors_total: p.tensors_total,
    });
}
