//! Local runtime helpers: Ollama/LM Studio hand-off and a smoke test.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::imatrix::find_tool;

#[derive(Debug, thiserror::Error)]
pub enum InstallError {
    #[error("I/O error: {0}")]
    Io(String),
    #[error("ollama is not installed")]
    MissingOllama,
    #[error("llama-cli is not installed; set LLAMA_CLI or install llama.cpp")]
    MissingLlamaCli,
    #[error("runtime command failed: {0}")]
    Runtime(String),
}

#[derive(Debug, Clone)]
pub struct InstallReport {
    pub runtime: String,
    pub model: Option<PathBuf>,
    pub instructions: PathBuf,
    pub command: Option<String>,
}

#[derive(Debug, Clone)]
pub struct QuickTestReport {
    pub output: String,
    pub command: String,
}

/// Write a Modelfile that imports an existing GGUF and, when Ollama is
/// available, create the named local model. The model file is never modified.
pub fn install_ollama(
    model: &Path,
    name: &str,
    directory: &Path,
    run: bool,
) -> Result<InstallReport, InstallError> {
    fs::create_dir_all(directory).map_err(|error| InstallError::Io(error.to_string()))?;
    let modelfile = directory.join("Modelfile");
    fs::write(&modelfile, format!("FROM {}\n", model.display()))
        .map_err(|error| InstallError::Io(error.to_string()))?;
    let mut report = InstallReport {
        runtime: "Ollama".to_string(),
        model: None,
        instructions: modelfile.clone(),
        command: Some(format!(
            "ollama create {} -f {}",
            shell_word(name),
            modelfile.display()
        )),
    };
    if run {
        let Some(ollama) = find_tool("ollama") else {
            return Err(InstallError::MissingOllama);
        };
        let output = Command::new(ollama)
            .args(["create", name, "-f"])
            .arg(&modelfile)
            .output()
            .map_err(|error| InstallError::Runtime(error.to_string()))?;
        if !output.status.success() {
            return Err(InstallError::Runtime(
                String::from_utf8_lossy(&output.stderr).trim().to_string(),
            ));
        }
        report.model = Some(PathBuf::from(name));
    }
    Ok(report)
}

/// LM Studio imports GGUF files through its UI. Generate a small, explicit
/// hand-off file instead of writing into the user's application data.
pub fn write_lmstudio_instructions(
    model: &Path,
    directory: &Path,
) -> Result<InstallReport, InstallError> {
    fs::create_dir_all(directory).map_err(|error| InstallError::Io(error.to_string()))?;
    let path = directory.join("LM-Studio-import.txt");
    fs::write(
        &path,
        format!(
            "RustBlock -> LM Studio\n\n1. Open LM Studio.\n2. My Models -> Import model.\n3. Choose: {}\n\nThe GGUF file is used in place; RustBlock does not copy or modify it.\n",
            model.display()
        ),
    )
    .map_err(|error| InstallError::Io(error.to_string()))?;
    Ok(InstallReport {
        runtime: "LM Studio".to_string(),
        model: Some(model.to_path_buf()),
        instructions: path,
        command: None,
    })
}

/// Run a short `llama-cli` generation to prove the converted file loads.
pub fn quick_test(
    model: &Path,
    prompt: &str,
    tokens: u32,
) -> Result<QuickTestReport, InstallError> {
    let tool = find_tool("llama-cli").ok_or(InstallError::MissingLlamaCli)?;
    let output = Command::new(&tool)
        .arg("-m")
        .arg(model)
        .arg("-p")
        .arg(prompt)
        .arg("-n")
        .arg(tokens.max(1).to_string())
        .output()
        .map_err(|error| InstallError::Runtime(error.to_string()))?;
    if !output.status.success() {
        return Err(InstallError::Runtime(
            String::from_utf8_lossy(&output.stderr).trim().to_string(),
        ));
    }
    Ok(QuickTestReport {
        output: String::from_utf8_lossy(&output.stdout).into_owned(),
        command: format!(
            "{} -m {} -p {prompt:?} -n {}",
            tool.display(),
            model.display(),
            tokens.max(1)
        ),
    })
}

fn shell_word(value: &str) -> String {
    if value
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || "._-".contains(c))
    {
        value.to_string()
    } else {
        format!("{value:?}")
    }
}
