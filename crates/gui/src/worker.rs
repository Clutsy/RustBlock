//! Background worker. One thread processes the selected jobs sequentially and
//! reports progress over a channel, so the UI never blocks on disk I/O.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use ternary_reblock_core::{
    convert, ConvertOptions, Gguf, Phase, Progress, Report, VerifyMode, TY_Q2_0,
};

use crate::model::VerifyOutcome;

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
    pub verify: VerifyMode,
    pub force: bool,
}

#[derive(Debug)]
pub enum Event {
    Started {
        id: u64,
        kind: WorkKind,
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
        if tx
            .send(Event::Started {
                id: item.id,
                kind: item.kind,
            })
            .is_err()
        {
            return;
        }
        let item_started = Instant::now();

        let result = match item.kind {
            WorkKind::Convert => {
                let mut progress = |p: &Progress| send_progress(&tx, item.id, p);
                convert(
                    &item.input,
                    &item.output,
                    ConvertOptions {
                        force: item.force,
                        dry_run: false,
                        verify: item.verify,
                    },
                    Some(&mut progress),
                )
                .map(Outcome::Converted)
                .map_err(|e| e.to_string())
            }
            WorkKind::Verify => verify_item(&item, &tx).map(Outcome::Verified),
        };

        match result {
            Ok(Outcome::Converted(report)) => {
                converted += 1;
                if tx
                    .send(Event::Done {
                        id: item.id,
                        report,
                    })
                    .is_err()
                {
                    return;
                }
            }
            Ok(Outcome::Verified(outcome)) => {
                verified += 1;
                let elapsed = item_started.elapsed();
                if tx
                    .send(Event::Verified {
                        id: item.id,
                        outcome,
                        elapsed,
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
        }
    }

    let _ = tx.send(Event::BatchDone {
        converted,
        verified,
        failed,
        elapsed: started.elapsed(),
    });
}

enum Outcome {
    Converted(Report),
    Verified(VerifyOutcome),
}

fn verify_item(item: &WorkItem, tx: &Sender<Event>) -> Result<VerifyOutcome, String> {
    let g = Gguf::open(&item.input).map_err(|e| e.to_string())?;

    if g.has_ptq1() {
        return Ok(VerifyOutcome::Ptq1 {
            tensors: g.tensors.len(),
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
        return Ok(VerifyOutcome::Pq2 {
            tensors: report.tensors_pq2,
        });
    }

    let q2 = g.tensors.iter().filter(|t| t.ty == TY_Q2_0).count();
    if q2 > 0 || g.file_type == Some(41) {
        Ok(VerifyOutcome::Native {
            tensors: g.tensors.len(),
        })
    } else {
        Ok(VerifyOutcome::Other {
            tensors: g.tensors.len(),
        })
    }
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
