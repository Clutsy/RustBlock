//! Detection and installation into local model managers.
//!
//! LM Studio reads plain GGUF files from its models tree, so a copy is enough.
//! Ollama uses a content-addressed blob store and manifests; copying a loose
//! GGUF into `blobs/` does not work. We therefore call the official
//! `ollama create` command with a generated Modelfile.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::model::{home_dir, shorten, InstallTarget};

#[derive(Debug, Clone)]
pub struct Detected {
    pub label: String,
    pub detail: String,
    pub target: InstallTarget,
}

pub fn detect() -> Vec<Detected> {
    let mut out = vec![Detected {
        label: "Do not install".to_string(),
        detail: "Keep the converted file where it is.".to_string(),
        target: InstallTarget::None,
    }];
    let mut seen = HashSet::new();

    let mut add = |label: &str, detail: String, target: InstallTarget| {
        let key = target.label();
        if seen.insert(key) {
            out.push(Detected {
                label: label.to_string(),
                detail,
                target,
            });
        }
    };

    for path in lmstudio_candidates() {
        if path.is_dir() {
            add(
                "LM Studio",
                format!(
                    "Copy into {} (RustBlock/<model>)",
                    shorten(&path.display().to_string(), 62)
                ),
                InstallTarget::LmStudio(path),
            );
        }
    }

    if let Some(exe) = find_in_path(if cfg!(windows) {
        "ollama.exe"
    } else {
        "ollama"
    }) {
        let models = ollama_candidates()
            .into_iter()
            .find(|p| p.exists())
            .or_else(|| home_dir().map(|h| h.join(".ollama").join("models")))
            .unwrap_or_else(|| PathBuf::from(".ollama/models"));
        add(
            "Ollama",
            format!(
                "Run `ollama create` using {}",
                shorten(&exe.display().to_string(), 58)
            ),
            InstallTarget::Ollama { exe, models },
        );
    }

    out
}

fn lmstudio_candidates() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Some(p) = std::env::var_os("LMSTUDIO_MODELS") {
        paths.push(PathBuf::from(p));
    }
    if let Some(home) = home_dir() {
        paths.push(home.join(".lmstudio").join("models"));
        paths.push(home.join(".cache").join("lm-studio").join("models"));
    }
    paths
}

fn ollama_candidates() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Some(p) = std::env::var_os("OLLAMA_MODELS") {
        paths.push(PathBuf::from(p));
    }
    if let Some(home) = home_dir() {
        paths.push(home.join(".ollama").join("models"));
    }
    paths.push(PathBuf::from("/usr/share/ollama/.ollama/models"));
    paths
}

fn find_in_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
        if cfg!(windows) && !name.to_ascii_lowercase().ends_with(".exe") {
            let exe = dir.join(format!("{name}.exe"));
            if exe.is_file() {
                return Some(exe);
            }
        }
    }
    None
}

/// Performs the install step after a successful conversion. The converted file
/// is never deleted; LM Studio gets a copy, Ollama imports from the original.
pub fn install_model(
    target: &InstallTarget,
    model_path: &Path,
    display_name: &str,
) -> Result<String, String> {
    match target {
        InstallTarget::None => Ok(String::new()),
        InstallTarget::LmStudio(root) => {
            let stem = safe_stem(display_name);
            let dest_dir = root.join("RustBlock").join(&stem);
            fs::create_dir_all(&dest_dir).map_err(|e| e.to_string())?;
            let file_name = model_path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| "model.gguf".to_string());
            let dest = dest_dir.join(file_name);
            if same_path(model_path, &dest) {
                return Ok(dest.display().to_string());
            }
            fs::copy(model_path, &dest).map_err(|e| e.to_string())?;
            Ok(dest.display().to_string())
        }
        InstallTarget::Folder(root) => {
            fs::create_dir_all(root).map_err(|e| e.to_string())?;
            let file_name = model_path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| "model.gguf".to_string());
            let dest = root.join(file_name);
            if same_path(model_path, &dest) {
                return Ok(dest.display().to_string());
            }
            fs::copy(model_path, &dest).map_err(|e| e.to_string())?;
            Ok(dest.display().to_string())
        }
        InstallTarget::Ollama { exe, models } => {
            let _ = models;
            let abs = fs::canonicalize(model_path).unwrap_or_else(|_| model_path.to_path_buf());
            let name = format!("rustblock-{}", safe_stem(display_name));
            let modelfile = std::env::temp_dir().join(format!(
                "rustblock-modelfile-{}-{}.txt",
                std::process::id(),
                name
            ));
            fs::write(
                &modelfile,
                format!(
                    "FROM \"{}\"\n",
                    abs.display()
                        .to_string()
                        .replace(std::path::MAIN_SEPARATOR, "/")
                ),
            )
            .map_err(|e| e.to_string())?;
            let output = Command::new(exe)
                .arg("create")
                .arg(&name)
                .arg("-f")
                .arg(&modelfile)
                .output();
            let _ = fs::remove_file(&modelfile);
            match output {
                Ok(out) if out.status.success() => Ok(format!("ollama create {name}")),
                Ok(out) => {
                    let err = String::from_utf8_lossy(&out.stderr);
                    Err(format!(
                        "`ollama create {name}` failed: {}",
                        shorten(err.trim(), 180)
                    ))
                }
                Err(e) => Err(format!("could not run ollama: {e}")),
            }
        }
    }
}

fn safe_stem(name: &str) -> String {
    let stem = Path::new(name)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model".to_string());
    let mut out = String::new();
    for c in stem.chars() {
        if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
            out.push(c);
        } else if c.is_whitespace() {
            out.push('-');
        }
    }
    let out = out.trim_matches('-').to_string();
    if out.is_empty() {
        "model".to_string()
    } else {
        out
    }
}

fn same_path(a: &Path, b: &Path) -> bool {
    match (fs::canonicalize(a), fs::canonicalize(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => a == b,
    }
}
