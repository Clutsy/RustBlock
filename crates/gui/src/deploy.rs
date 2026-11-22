//! Copy a finished GGUF into the model folder of a local runtime.
//!
//! Ollama is imported with its own Modelfile command; LM Studio, Jan and
//! Prism-style folders are plain copies. Everything runs on a worker thread so
//! a multi-gigabyte copy never blocks the window.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use rustblock_core::{install_ollama, InstallError};

use crate::model::{config_dir, Settings};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Runtime {
    Ollama,
    LmStudio,
    Jan,
    Prism,
}

impl Runtime {
    pub const ALL: [Runtime; 4] = [
        Runtime::Ollama,
        Runtime::LmStudio,
        Runtime::Jan,
        Runtime::Prism,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Ollama => "Ollama",
            Self::LmStudio => "LM Studio",
            Self::Jan => "Jan",
            Self::Prism => "Prism / other",
        }
    }

    pub fn hint(self) -> &'static str {
        match self {
            Self::Ollama => "Write a Modelfile and run ollama create. The model is registered, not copied by hand.",
            Self::LmStudio => "Copy into ~/.lmstudio/models/rustblock/<model>/ so My Models lists it.",
            Self::Jan => "Copy into the Jan data folder and write model.yml for the llama.cpp engine.",
            Self::Prism => "Copy into the configured folder, ready for llama.cpp or Prism.",
        }
    }
}

#[derive(Debug)]
pub enum DeployEvent {
    Progress { done: u64, total: u64 },
    Done { folder: PathBuf, message: String },
    Failed(String),
}

#[derive(Debug)]
pub struct DeployUi {
    pub model: PathBuf,
    pub runtime: Option<Runtime>,
    pub busy: bool,
    pub done: u64,
    pub total: u64,
    pub message: Option<String>,
    pub folder: Option<PathBuf>,
    pub rx: Option<Receiver<DeployEvent>>,
    pub worker: Option<JoinHandle<()>>,
    pub cancel: Arc<AtomicBool>,
}

impl DeployUi {
    pub fn new(model: PathBuf) -> Self {
        Self {
            model,
            runtime: None,
            busy: false,
            done: 0,
            total: 0,
            message: None,
            folder: None,
            rx: None,
            worker: None,
            cancel: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn fraction(&self) -> f32 {
        if self.total > 0 {
            (self.done as f64 / self.total as f64).clamp(0.0, 1.0) as f32
        } else {
            0.0
        }
    }
}

pub fn spawn(
    runtime: Runtime,
    model: PathBuf,
    settings: Settings,
    tx: Sender<DeployEvent>,
    cancel: Arc<AtomicBool>,
) -> JoinHandle<()> {
    thread::spawn(move || run(runtime, &model, &settings, &tx, &cancel))
}

fn run(
    runtime: Runtime,
    model: &Path,
    settings: &Settings,
    tx: &Sender<DeployEvent>,
    cancel: &AtomicBool,
) {
    let result = match runtime {
        Runtime::Ollama => install_to_ollama(model, tx),
        other => copy_to_runtime(other, model, settings, tx, cancel),
    };
    if let Err(error) = result {
        let _ = tx.send(DeployEvent::Failed(error));
    }
}

fn install_to_ollama(model: &Path, tx: &Sender<DeployEvent>) -> Result<(), String> {
    let name = safe_stem(model);
    let directory = config_dir().join("ollama").join(&name);
    match install_ollama(model, &name, &directory, true) {
        Ok(_) => {
            let _ = tx.send(DeployEvent::Done {
                folder: directory,
                message: format!("Ollama model '{name}' created. Test it with: ollama run {name}"),
            });
        }
        Err(InstallError::MissingOllama) => {
            let _ = tx.send(DeployEvent::Done {
                folder: directory.clone(),
                message: format!(
                    "Modelfile written to {}. Install Ollama, then run: ollama create {name} -f {}/Modelfile",
                    directory.display(),
                    directory.display()
                ),
            });
        }
        Err(error) => return Err(format!("Ollama import failed: {error}")),
    }
    Ok(())
}

fn copy_to_runtime(
    runtime: Runtime,
    model: &Path,
    settings: &Settings,
    tx: &Sender<DeployEvent>,
    cancel: &AtomicBool,
) -> Result<(), String> {
    let stem = safe_stem(model);
    let folder = match runtime {
        Runtime::LmStudio => settings.lmstudio_dir.join("rustblock").join(&stem),
        Runtime::Jan => settings
            .jan_dir
            .join("llamacpp")
            .join("models")
            .join("rustblock")
            .join(&stem),
        Runtime::Prism => settings.prism_dir.join("rustblock").join(&stem),
        Runtime::Ollama => unreachable!("Ollama is imported, not copied"),
    };
    fs::create_dir_all(&folder)
        .map_err(|error| format!("cannot create {}: {error}", folder.display()))?;
    let name = model
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| format!("{stem}.gguf"));
    let destination = folder.join(&name);
    let total = fs::metadata(model).map(|meta| meta.len()).unwrap_or(0);
    let _ = tx.send(DeployEvent::Progress { done: 0, total });
    copy_with_progress(model, &destination, total, tx, cancel)?;
    if runtime == Runtime::Jan {
        let relative = format!("llamacpp/models/rustblock/{stem}/{name}");
        let yml = format!(
            "model_path: {relative}\nname: {stem}\nsize_bytes: {total}\nembedding: false\n"
        );
        fs::write(folder.join("model.yml"), yml)
            .map_err(|error| format!("cannot write model.yml: {error}"))?;
    }
    let _ = tx.send(DeployEvent::Done {
        folder: folder.clone(),
        message: format!(
            "Copied to {}. Restart {} to see the model.",
            folder.display(),
            runtime.label()
        ),
    });
    Ok(())
}

fn copy_with_progress(
    source: &Path,
    destination: &Path,
    total: u64,
    tx: &Sender<DeployEvent>,
    cancel: &AtomicBool,
) -> Result<(), String> {
    let mut input = fs::File::open(source)
        .map_err(|error| format!("cannot open {}: {error}", source.display()))?;
    let mut output = fs::File::create(destination)
        .map_err(|error| format!("cannot create {}: {error}", destination.display()))?;
    let mut buffer = vec![0u8; 1024 * 1024];
    let mut done = 0u64;
    let mut last = Instant::now();
    loop {
        if cancel.load(Ordering::Relaxed) {
            drop(output);
            let _ = fs::remove_file(destination);
            return Err("Copy cancelled.".to_string());
        }
        let read = input
            .read(&mut buffer)
            .map_err(|error| format!("read failed: {error}"))?;
        if read == 0 {
            break;
        }
        output
            .write_all(&buffer[..read])
            .map_err(|error| format!("write failed: {error}"))?;
        done += read as u64;
        if last.elapsed() >= Duration::from_millis(120) || done == total {
            let _ = tx.send(DeployEvent::Progress { done, total });
            last = Instant::now();
        }
    }
    output
        .flush()
        .map_err(|error| format!("flush failed: {error}"))?;
    let _ = tx.send(DeployEvent::Progress { done, total });
    Ok(())
}

fn safe_stem(model: &Path) -> String {
    let raw = model
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model".to_string());
    let cleaned: String = raw
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
                ch
            } else {
                '-'
            }
        })
        .collect();
    if cleaned.trim_matches('-').is_empty() {
        "model".to_string()
    } else {
        cleaned
    }
}
