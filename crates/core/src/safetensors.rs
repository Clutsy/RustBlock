//! Optional bridge to Hugging Face's `convert_hf_to_gguf.py`.
//!
//! RustBlock does not bundle Python, transformers or the conversion script.
//! The bridge exists for users who already have a working Hugging Face
//! conversion environment; it never downloads or executes an arbitrary script
//! by itself.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use crate::imatrix::find_tool;

#[derive(Debug, thiserror::Error)]
pub enum SafetensorsError {
    #[error("convert_hf_to_gguf.py was not found; set RUSTBLOCK_HF_CONVERT or pass --script")]
    MissingScript,
    #[error("Python was not found; install Python 3 or set RUSTBLOCK_PYTHON")]
    MissingPython,
    #[error("conversion failed: {0}")]
    Failed(String),
    #[error("conversion command could not be started: {0}")]
    Spawn(String),
}

/// Finds the official conversion script only in an explicitly configured path
/// or on `PATH`. It does not search the network or clone a repository.
pub fn find_script() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("RUSTBLOCK_HF_CONVERT") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Some(path);
        }
    }
    find_tool("convert_hf_to_gguf.py")
}

pub fn find_python() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("RUSTBLOCK_PYTHON") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Some(path);
        }
    }
    for name in ["python3", "python"] {
        if let Some(path) = find_tool(name) {
            return Some(path);
        }
    }
    None
}

/// Run the Hugging Face converter. `outtype` is passed through and accepts
/// values supported by the script (`f16`, `bf16`, `f32`, `q8_0`, ...).
pub fn convert_with_hf_script(
    input_dir: &Path,
    output: &Path,
    script: Option<&Path>,
    outtype: &str,
) -> Result<Output, SafetensorsError> {
    let script = script
        .map(Path::to_path_buf)
        .or_else(find_script)
        .ok_or(SafetensorsError::MissingScript)?;
    let python = find_python().ok_or(SafetensorsError::MissingPython)?;
    let output = output.to_path_buf();
    let output_arg = output.to_string_lossy().into_owned();
    let input_arg = input_dir.to_string_lossy().into_owned();
    let status = Command::new(&python)
        .arg(&script)
        .arg(&input_arg)
        .arg("--outfile")
        .arg(&output_arg)
        .arg("--outtype")
        .arg(outtype)
        .output()
        .map_err(|error| SafetensorsError::Spawn(error.to_string()))?;
    if !status.status.success() {
        return Err(SafetensorsError::Failed(
            String::from_utf8_lossy(&status.stderr)
                .lines()
                .rev()
                .take(8)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect::<Vec<_>>()
                .join("\n"),
        ));
    }
    if !output.is_file() {
        return Err(SafetensorsError::Failed(format!(
            "{} did not create an output GGUF",
            script.display()
        )));
    }
    Ok(status)
}
