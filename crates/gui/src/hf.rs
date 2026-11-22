//! Hugging Face Hub client for GGUF discovery and downloads.
//!
//! Search results are enriched from the public model API (author, license,
//! base model, tags, file count). Selecting a repository fetches its model
//! card, README summary, architecture facts and the real GGUF file list.
//! Setting `HF_TOKEN` enables gated/private repositories.
//! Downloads are resumable through `.part` files and byte ranges.

use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Mutex, OnceLock, RwLock};
use std::time::Instant;

use crate::model::Source;

const DEFAULT_BASE: &str = "https://huggingface.co";
const MODELSCOPE: &str = "https://modelscope.cn";
const UA: &str = "RustBlock/0.1 (GGUF conversion desktop app)";

static ENDPOINT: OnceLock<RwLock<String>> = OnceLock::new();

/// Point the Hub client at a mirror such as `https://hf-mirror.com`.
pub fn set_endpoint(value: &str) {
    let value = value.trim().trim_end_matches('/');
    let endpoint = if value.starts_with("http") {
        value.to_string()
    } else {
        DEFAULT_BASE.to_string()
    };
    let lock = ENDPOINT.get_or_init(|| RwLock::new(DEFAULT_BASE.to_string()));
    if let Ok(mut guard) = lock.write() {
        *guard = endpoint;
    }
}

fn base() -> String {
    ENDPOINT
        .get_or_init(|| RwLock::new(DEFAULT_BASE.to_string()))
        .read()
        .map(|guard| guard.clone())
        .unwrap_or_else(|_| DEFAULT_BASE.to_string())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HfSort {
    Downloads,
    Likes,
    Recent,
}

impl HfSort {
    pub fn api(self) -> &'static str {
        match self {
            Self::Downloads => "downloads",
            Self::Likes => "likes",
            Self::Recent => "lastModified",
        }
    }
}

/// A model reference extracted from a Hugging Face URL or a bare `owner/model` id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfLink {
    pub repo: String,
    /// A `.gguf` file from a `/blob/` or `/resolve/` link, when present.
    pub file: Option<String>,
}

/// Accept a full `huggingface.co` link, an `hf.co` link or a bare repo id.
///
/// Dataset and Space links are rejected: they are not model repositories and
/// cannot provide GGUF files to the converter.
pub fn parse_link(input: &str) -> Option<HfLink> {
    let raw = input.trim();
    if raw.is_empty() || raw.chars().any(char::is_whitespace) {
        return None;
    }
    let no_fragment = raw.split(['?', '#']).next().unwrap_or(raw);
    let no_scheme = no_fragment
        .strip_prefix("https://")
        .or_else(|| no_fragment.strip_prefix("http://"))
        .unwrap_or(no_fragment);
    let no_scheme = no_scheme.trim_end_matches('/');
    let (host, path) = no_scheme.split_once('/')?;
    let host = host.to_ascii_lowercase();
    let is_hf_host = matches!(
        host.as_str(),
        "huggingface.co" | "www.huggingface.co" | "hf.co" | "www.hf.co"
    );
    let is_ms_host = matches!(
        host.as_str(),
        "modelscope.cn" | "www.modelscope.cn" | "modelscope.ai"
    );

    let (segments, from_hf): (Vec<&str>, bool) = if is_hf_host || is_ms_host {
        let mut parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
        if is_ms_host && parts.first() == Some(&"models") {
            parts.remove(0);
        }
        (parts, true)
    } else if !no_fragment.contains("://") && is_repo_pair(&host, path) {
        // A bare `owner/model` reference.
        (vec![host.as_str(), path], false)
    } else {
        return None;
    };

    let mut rest = segments.as_slice();
    if from_hf {
        if rest.first() == Some(&"datasets") || rest.first() == Some(&"spaces") {
            return None;
        }
        if rest.starts_with(&["api", "models"]) {
            rest = &rest[2..];
        } else if rest.first() == Some(&"models") {
            rest = &rest[1..];
        }
    }
    let owner = *rest.first()?;
    let name = *rest.get(1)?;
    if owner.is_empty() || name.is_empty() || owner == "api" || name == "api" {
        return None;
    }
    let repo = format!("{owner}/{name}");

    let mut file = None;
    if from_hf {
        if let Some(pos) = rest.iter().position(|s| *s == "blob" || *s == "resolve") {
            // /blob/<revision>/<path/to/file.gguf>
            let file_parts = rest.get(pos + 2..).unwrap_or_default();
            if !file_parts.is_empty() {
                let joined = file_parts.join("/");
                if joined.to_ascii_lowercase().ends_with(".gguf") {
                    file = Some(
                        urlencoding::decode(&joined)
                            .map(|s| s.into_owned())
                            .unwrap_or(joined),
                    );
                }
            }
        }
    }
    Some(HfLink { repo, file })
}

fn is_repo_pair(host: &str, path: &str) -> bool {
    !host.is_empty()
        && !path.is_empty()
        && !path.contains('/')
        && !host.contains('.')
        && host != "models"
        && !matches!(
            host,
            "datasets" | "spaces" | "docs" | "blog" | "papers" | "collections"
        )
}

/// Fetch one model by exact repo id. Used by the paste-a-link flow.
pub fn fetch_model(repo: &str) -> Result<HfModel, String> {
    let url = format!("{}/api/models/{}", base(), encode_path(repo));
    let value = get_json(&url)?;
    parse_model(&value).ok_or_else(|| format!("Hugging Face did not return a model for {repo}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_hugging_face_model_links() {
        assert_eq!(
            parse_link("https://huggingface.co/Qwen/Qwen3-8B-GGUF").unwrap(),
            HfLink {
                repo: "Qwen/Qwen3-8B-GGUF".into(),
                file: None
            }
        );
        assert_eq!(
            parse_link("huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/main").unwrap(),
            HfLink {
                repo: "unsloth/Qwen3.8-27B-GGUF".into(),
                file: None
            }
        );
        assert_eq!(
            parse_link("https://hf.co/owner/model/blob/main/sub/model-Q4_K_M.gguf").unwrap(),
            HfLink {
                repo: "owner/model".into(),
                file: Some("sub/model-Q4_K_M.gguf".into())
            }
        );
        assert_eq!(
            parse_link("owner/model").unwrap(),
            HfLink {
                repo: "owner/model".into(),
                file: None
            }
        );
    }

    #[test]
    fn rejects_non_model_links() {
        assert!(parse_link("https://huggingface.co/datasets/owner/data").is_none());
        assert!(parse_link("https://example.com/owner/model").is_none());
        assert!(parse_link("not a link").is_none());
    }

    #[test]
    fn accepts_only_supported_gguf_models() {
        assert!(file_support("Qwen/Qwen3-8B-Q4_K_M.gguf").is_ok());
        assert!(file_support("nested/model-BF16.gguf").is_ok());
        assert!(file_support("model.safetensors").is_err());
        assert!(file_support("imatrix_unsloth.gguf").is_err());
        assert!(file_support("mmproj-model-f16.gguf").is_err());
        assert!(is_projector("mmproj-model-f16.gguf"));
        assert!(is_projector("nested/projector-vision.gguf"));
        assert!(!is_projector("model-Q4_K_M.gguf"));
        assert!(file_support("model-lora.gguf").is_err());
        assert!(file_support("model-PTQ1_0.gguf").is_err());
        assert!(file_support("model-00001-of-00003.gguf").is_err());
        assert!(file_support("proof-of-concept.gguf").is_ok());
    }

    #[test]
    fn download_rejects_unsupported_files_before_touching_network() {
        let (tx, _rx) = std::sync::mpsc::channel();
        let cancel = std::sync::atomic::AtomicBool::new(false);
        let error = download_parallel(
            "owner/repo",
            "main",
            "imatrix_unsloth.gguf",
            Path::new("target"),
            None,
            &cancel,
            &tx,
        )
        .unwrap_err();
        assert!(error.contains("importance matrix"));
        assert!(!Path::new("target/imatrix_unsloth.gguf").exists());
    }

    #[test]
    fn projector_download_rejects_a_model_before_touching_network() {
        let (tx, _rx) = std::sync::mpsc::channel();
        let cancel = std::sync::atomic::AtomicBool::new(false);
        let error = download_projector(
            "owner/repo",
            "main",
            "model-Q4_K_M.gguf",
            Path::new("target"),
            None,
            &cancel,
            &tx,
        )
        .unwrap_err();
        assert!(error.contains("not a vision projector"));
        assert!(!Path::new("target/model-Q4_K_M.gguf").exists());
    }

    #[test]
    fn validation_rejects_a_non_gguf_download() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("fake.gguf");
        std::fs::write(&file, b"not a gguf").unwrap();
        assert!(validate_download(&file).is_err());
    }
}

#[derive(Debug, Clone)]
pub struct HfModel {
    pub id: String,
    pub author: String,
    pub name: String,
    pub downloads: u64,
    pub last_modified: String,
    pub gated: bool,
    pub sha: String,
    pub gguf_files: usize,
    /// License declared on the Hub, when the API exposes one.
    pub license: Option<String>,
    /// Public page for this repository on its own source.
    pub web: String,
}

impl HfModel {
    pub fn web_url(&self) -> String {
        self.web.clone()
    }
}

#[derive(Debug, Clone)]
pub struct HfFile {
    pub path: String,
    pub size: u64,
    /// SHA-256 published by the Hub for LFS files. Used as the expected
    /// digest when the download finishes.
    pub sha256: Option<String>,
    /// False for GGUF containers that are not standalone models (importance
    /// matrices, projectors, adapters, split shards, rejected Prism builds).
    pub supported: bool,
    /// A vision projector can be downloaded explicitly next to its model, but
    /// it must never enter the conversion queue as a standalone model.
    pub projector: bool,
}

pub fn is_projector(path: &str) -> bool {
    let name = path.to_ascii_lowercase();
    name.contains("mmproj") || name.contains("projector")
}

pub fn file_name(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

/// Reject files that have a .gguf suffix but are not a supported standalone
/// model. Downloads are gated by this check as well as the file list UI.
pub fn file_support(path: &str) -> Result<(), &'static str> {
    let lower = path.to_ascii_lowercase();
    if !lower.ends_with(".gguf") {
        return Err("not a GGUF file");
    }
    let name = lower.rsplit('/').next().unwrap_or(lower.as_str());
    if name.contains("imatrix") || name.contains("importance") {
        return Err("importance matrix, not a model");
    }
    if is_projector(name) {
        return Err("vision projector, not a standalone model");
    }
    if name.contains("lora") || name.contains("adapter") {
        return Err("adapter, not a standalone model");
    }
    if name.contains("ptq1") {
        return Err("PTQ1_0 is not convertible by RustBlock");
    }
    if is_split_shard(name) {
        return Err("split shard, not a standalone model");
    }
    Ok(())
}

fn is_split_shard(name: &str) -> bool {
    let stem = name.strip_suffix(".gguf").unwrap_or(name);
    let Some((left, right)) = stem.rsplit_once("-of-") else {
        return false;
    };
    let right_ok = right.len() == 5 && right.bytes().all(|b| b.is_ascii_digit());
    let left_ok = left
        .rsplit_once('-')
        .is_some_and(|(_, index)| index.len() == 5 && index.bytes().all(|b| b.is_ascii_digit()));
    right_ok && left_ok
}

#[derive(Debug, Clone)]
pub enum HfEvent {
    SearchStarted,
    SearchResults(Vec<HfModel>),
    DetailsStarted(String),
    DetailsLoaded {
        repo: String,
        files: Vec<HfFile>,
    },
    DetailsError {
        repo: String,
        error: String,
    },
    Error(String),
    DownloadStarted {
        total: u64,
    },
    DownloadProgress {
        done: u64,
        total: u64,
        speed: f64,
    },
    Downloaded {
        repo: String,
        file: String,
        path: PathBuf,
        projector: bool,
    },
    DownloadFailed {
        repo: String,
        file: String,
        error: String,
    },
    DownloadCancelled,
}

pub fn search(query: &str, sort: HfSort) -> Result<Vec<HfModel>, String> {
    let search = if query.trim().is_empty() {
        String::new()
    } else {
        format!("&search={}", urlencoding::encode(query.trim()))
    };
    let url = format!(
        "{}/api/models?filter=gguf&sort={}&direction=-1&limit=50&full=true{search}",
        base(),
        sort.api()
    );
    let value = get_json(&url)?;
    let array = value
        .as_array()
        .ok_or_else(|| "Hugging Face returned an unexpected search response".to_string())?;
    Ok(array.iter().filter_map(parse_model).collect())
}

fn parse_model(item: &Value) -> Option<HfModel> {
    let id = item.get("id").and_then(Value::as_str)?.to_string();
    let author = item
        .get("author")
        .and_then(Value::as_str)
        .or_else(|| id.split('/').next())
        .unwrap_or("unknown")
        .to_string();
    let name = id
        .split_once('/')
        .map(|(_, name)| name.to_string())
        .unwrap_or_else(|| id.clone());
    let gguf_files = item
        .get("siblings")
        .and_then(Value::as_array)
        .map(|siblings| {
            siblings
                .iter()
                .filter(|s| {
                    s.get("rfilename")
                        .and_then(Value::as_str)
                        .is_some_and(|name| name.to_ascii_lowercase().ends_with(".gguf"))
                })
                .count()
        })
        .unwrap_or(0);
    let web = format!("{}/{}", base(), id);
    Some(HfModel {
        id,
        author,
        name,
        web,
        downloads: u64_field(item, "downloads").unwrap_or(0),
        last_modified: str_field(item, "lastModified")
            .unwrap_or_default()
            .to_string(),
        gated: gated_field(item),
        sha: str_field(item, "sha")
            .filter(|sha| !sha.is_empty())
            .unwrap_or("main")
            .to_string(),
        gguf_files,
        license: str_field(item, "license").map(str::to_string),
    })
}

pub fn parse_modelscope(input: &str) -> Option<String> {
    let raw = input.trim().trim_end_matches('/');
    if raw.is_empty() || raw.chars().any(char::is_whitespace) {
        return None;
    }
    let no_scheme = raw
        .strip_prefix("https://")
        .or_else(|| raw.strip_prefix("http://"))
        .unwrap_or(raw);
    let (host, path) = no_scheme.split_once('/')?;
    let host = host.to_ascii_lowercase();
    if matches!(
        host.as_str(),
        "modelscope.cn" | "www.modelscope.cn" | "modelscope.ai"
    ) {
        let mut parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
        if parts.first() == Some(&"models") {
            parts.remove(0);
        }
        let owner = *parts.first()?;
        let name = *parts.get(1)?;
        return Some(format!("{owner}/{name}"));
    }
    if !host.contains('.') && !path.contains('/') && !host.is_empty() {
        return Some(format!("{host}/{path}"));
    }
    None
}

pub fn fetch_model_ms(repo: &str) -> HfModel {
    let mut parts = repo.splitn(2, '/');
    let author = parts.next().unwrap_or("unknown").to_string();
    let name = parts.next().unwrap_or(repo).to_string();
    HfModel {
        id: repo.to_string(),
        author,
        name,
        downloads: 0,
        last_modified: String::new(),
        gated: false,
        sha: "master".to_string(),
        gguf_files: 0,
        license: None,
        web: format!("{MODELSCOPE}/models/{repo}"),
    }
}

fn get_json_ms(url: &str) -> Result<Value, String> {
    match ureq::get(url).set("User-Agent", UA).call() {
        Ok(response) => response
            .into_json::<Value>()
            .map_err(|error| format!("invalid ModelScope response: {error}")),
        Err(ureq::Error::Status(code, response)) => {
            let body = response.into_string().unwrap_or_default();
            Err(format!(
                "ModelScope returned HTTP {code}. {}",
                body.chars().take(180).collect::<String>()
            ))
        }
        Err(error) => Err(format!("network error: {error}")),
    }
}

pub fn list_files_ms(repo: &str) -> Result<Vec<HfFile>, String> {
    let url = format!(
        "{MODELSCOPE}/api/v1/models/{}/repo/files?Revision=master&Recursive=true",
        encode_path(repo)
    );
    let value = get_json_ms(&url)?;
    if value.get("Success").and_then(Value::as_bool) == Some(false) {
        let message = value
            .get("Message")
            .and_then(Value::as_str)
            .unwrap_or("ModelScope rejected the request");
        return Err(format!("ModelScope: {message}"));
    }
    let data = value
        .get("Data")
        .or_else(|| value.get("data"))
        .ok_or_else(|| "ModelScope returned an unexpected file list".to_string())?;
    let array = data
        .get("Files")
        .or_else(|| data.get("files"))
        .and_then(Value::as_array)
        .ok_or_else(|| "ModelScope returned an unexpected file list".to_string())?;
    let mut out = Vec::new();
    for item in array {
        let Some(path) = item
            .get("Path")
            .or_else(|| item.get("path"))
            .and_then(Value::as_str)
        else {
            continue;
        };
        if !path.to_ascii_lowercase().ends_with(".gguf") {
            continue;
        }
        let size = item
            .get("Size")
            .or_else(|| item.get("size"))
            .and_then(|value| {
                value
                    .as_u64()
                    .or_else(|| value.as_str().and_then(|text| text.parse().ok()))
            })
            .unwrap_or(0);
        let sha256 = item
            .get("Sha256")
            .or_else(|| item.get("sha256"))
            .and_then(Value::as_str)
            .filter(|oid| oid.len() == 64 && oid.bytes().all(|byte| byte.is_ascii_hexdigit()))
            .map(str::to_string);
        let projector = is_projector(path);
        let support = if projector {
            Err("vision projector")
        } else {
            file_support(path)
        };
        out.push(HfFile {
            path: path.to_string(),
            size,
            sha256,
            supported: support.is_ok(),
            projector,
        });
    }
    if out.is_empty() {
        return Err(format!(
            "ModelScope lists no GGUF files for {repo}. Check the model id."
        ));
    }
    out.sort_by_key(|file| (!file.supported, file.size));
    Ok(out)
}

pub fn resolve_url(source: Source, repo: &str, revision: &str, file: &str) -> String {
    let revision = if revision.is_empty() {
        match source {
            Source::HuggingFace => "main",
            Source::ModelScope => "master",
        }
    } else {
        revision
    };
    match source {
        Source::HuggingFace => format!(
            "{}/{}/resolve/{}/{}",
            base(),
            encode_path(repo),
            encode_path(revision),
            encode_path(file)
        ),
        Source::ModelScope => format!(
            "{MODELSCOPE}/api/v1/models/{}/repo?Revision={}&FilePath={}",
            encode_path(repo),
            urlencoding::encode(revision),
            urlencoding::encode(file)
        ),
    }
}

pub fn list_files(repo: &str, revision: &str) -> Result<Vec<HfFile>, String> {
    let rev = if revision.is_empty() {
        "main"
    } else {
        revision
    };
    let url = format!(
        "{}/api/models/{}/tree/{}?recursive=true",
        base(),
        encode_path(repo),
        encode_path(rev)
    );
    let value = get_json(&url)?;
    let array = value
        .as_array()
        .ok_or_else(|| "Hugging Face returned an unexpected file list".to_string())?;
    let mut out = Vec::new();
    for item in array {
        if item.get("type").and_then(Value::as_str) != Some("file") {
            continue;
        }
        let Some(path) = item.get("path").and_then(Value::as_str) else {
            continue;
        };
        if !path.to_ascii_lowercase().ends_with(".gguf") {
            continue;
        }
        let size = item
            .get("size")
            .and_then(Value::as_u64)
            .or_else(|| {
                item.get("lfs")
                    .and_then(|l| l.get("size"))
                    .and_then(Value::as_u64)
            })
            .unwrap_or(0);
        let sha256 = item
            .get("lfs")
            .and_then(|lfs| lfs.get("oid"))
            .and_then(Value::as_str)
            .filter(|oid| oid.len() == 64 && oid.bytes().all(|byte| byte.is_ascii_hexdigit()))
            .map(str::to_string);
        let projector = is_projector(path);
        let support = if projector {
            Err("vision projector")
        } else {
            file_support(path)
        };
        out.push(HfFile {
            path: path.to_string(),
            size,
            sha256,
            supported: support.is_ok(),
            projector,
        });
    }
    // Keep usable models first; unsupported artifacts remain visible with a
    // warning so the user understands why a file is not downloadable.
    out.sort_by_key(|f| (!f.supported, f.size));
    Ok(out)
}

/// Re-open the completed file before announcing it. A `.gguf` name is not a
/// promise that the container is usable: reject malformed files, PTQ1_0 and
/// tensor types the native engine cannot decode, and remove them from disk.
fn validate_download(path: &Path) -> Result<(), String> {
    let g = rustblock_core::Gguf::open(path)
        .map_err(|e| format!("downloaded file is not a readable GGUF: {e}"))?;
    if g.has_ptq1() {
        return Err("downloaded file contains PTQ1_0, which RustBlock cannot convert".to_string());
    }
    if let Some(tensor) = g
        .tensors
        .iter()
        .find(|tensor| !rustblock_core::source_supported(tensor.ty))
    {
        return Err(format!(
            "downloaded file contains tensor type {} ('{}'), which this build cannot decode",
            tensor.ty, tensor.name
        ));
    }
    Ok(())
}

/// Download through the core's parallel/resumable transfer engine. The GUI
/// used to keep its own single-stream implementation; this wrapper preserves
/// the UI events while using the same 16-range downloader as the CLI.
#[cfg(test)]
pub fn download_parallel(
    repo: &str,
    revision: &str,
    file: &str,
    dest_dir: &Path,
    expected_sha256: Option<&str>,
    cancel: &AtomicBool,
    tx: &Sender<HfEvent>,
) -> Result<PathBuf, String> {
    download_artifact(
        repo,
        revision,
        file,
        dest_dir,
        false,
        expected_sha256,
        cancel,
        tx,
        false,
        Source::HuggingFace,
    )
}

/// Download a vision projector explicitly. Projectors are GGUF containers but
/// are not standalone models, so they are kept out of the model download path.
#[cfg(test)]
pub fn download_projector(
    repo: &str,
    revision: &str,
    file: &str,
    dest_dir: &Path,
    expected_sha256: Option<&str>,
    cancel: &AtomicBool,
    tx: &Sender<HfEvent>,
) -> Result<PathBuf, String> {
    download_artifact(
        repo,
        revision,
        file,
        dest_dir,
        false,
        expected_sha256,
        cancel,
        tx,
        true,
        Source::HuggingFace,
    )
}

/// Download a file from either source. The URL shape and token policy are the
/// only differences; transfers and UI events stay shared with the Hub path.
#[allow(clippy::too_many_arguments)]
pub fn download_from(
    source: Source,
    repo: &str,
    revision: &str,
    file: &str,
    dest_dir: &Path,
    expected_sha256: Option<&str>,
    cancel: &AtomicBool,
    tx: &Sender<HfEvent>,
) -> Result<PathBuf, String> {
    let projector = is_projector(file);
    if source == Source::ModelScope && projector {
        return Err("ModelScope projectors are not supported by this build.".to_string());
    }
    download_artifact(
        repo,
        revision,
        file,
        dest_dir,
        false,
        expected_sha256,
        cancel,
        tx,
        projector,
        source,
    )
}

#[allow(clippy::too_many_arguments)]
fn download_artifact(
    repo: &str,
    revision: &str,
    file: &str,
    dest_dir: &Path,
    overwrite: bool,
    expected_sha256: Option<&str>,
    cancel: &AtomicBool,
    tx: &Sender<HfEvent>,
    projector: bool,
    source: Source,
) -> Result<PathBuf, String> {
    if projector {
        if !is_projector(file) {
            return Err(format!("Cannot download {file}: not a vision projector."));
        }
    } else {
        file_support(file).map_err(|reason| format!("Cannot download {file}: {reason}."))?;
    }
    fs::create_dir_all(dest_dir).map_err(|e| format!("cannot create download folder: {e}"))?;
    let name = file_name(file);
    let name = if name.is_empty() { "model.gguf" } else { name };
    let final_path = dest_dir.join(name);
    let existed = final_path.exists();
    let url = resolve_url(source, repo, revision, file);
    let options = rustblock_core::DownloadOptions {
        connections: if source == Source::ModelScope { 4 } else { 16 },
        force: overwrite,
        expected_sha256: expected_sha256.map(str::to_string),
        token: if source == Source::HuggingFace {
            std::env::var("HF_TOKEN")
                .ok()
                .filter(|value| !value.trim().is_empty())
        } else {
            None
        },
    };
    let started = AtomicBool::new(false);
    let meter = Mutex::new((Instant::now(), 0u64));
    let send_started = |total: u64| {
        let _ = tx.send(HfEvent::DownloadStarted { total });
    };
    let progress_tx = tx.clone();
    let report = rustblock_core::download_url_with_progress(
        &url,
        &final_path,
        &options,
        cancel,
        &move |done, total| {
            if !started.swap(true, Ordering::Relaxed) {
                send_started(total);
            }
            let now = Instant::now();
            let mut previous = meter.lock().expect("download speed lock");
            let elapsed = now.duration_since(previous.0).as_secs_f64();
            let speed = if elapsed > 0.0 && done >= previous.1 {
                (done - previous.1) as f64 / elapsed
            } else {
                0.0
            };
            *previous = (now, done);
            drop(previous);
            let _ = progress_tx.send(HfEvent::DownloadProgress {
                done,
                total: total.max(done),
                speed,
            });
        },
    )
    .map_err(|error| {
        let message = error.to_string();
        if message.contains("cancelled") {
            let _ = tx.send(HfEvent::DownloadCancelled);
            "download cancelled".to_string()
        } else {
            message
        }
    })?;
    if let Err(error) = validate_download(&final_path) {
        if !existed {
            let _ = fs::remove_file(&final_path);
        }
        return Err(error);
    }
    let _ = tx.send(HfEvent::Downloaded {
        repo: repo.to_string(),
        file: file.to_string(),
        path: report.path.clone(),
        projector,
    });
    Ok(report.path)
}

fn gated_field(item: &Value) -> bool {
    match item.get("gated") {
        Some(Value::Bool(v)) => *v,
        Some(Value::String(v)) => !v.is_empty() && v != "false",
        _ => false,
    }
}

fn u64_field(item: &Value, key: &str) -> Option<u64> {
    item.get(key).and_then(|v| {
        v.as_u64()
            .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
    })
}

fn str_field<'a>(item: &'a Value, key: &str) -> Option<&'a str> {
    item.get(key).and_then(Value::as_str)
}

fn get_json(url: &str) -> Result<Value, String> {
    let mut request = ureq::get(url).set("User-Agent", UA);
    if let Ok(token) = std::env::var("HF_TOKEN") {
        if !token.trim().is_empty() {
            request = request.set("Authorization", &format!("Bearer {}", token.trim()));
        }
    }
    match request.call() {
        Ok(response) => response
            .into_json::<Value>()
            .map_err(|e| format!("invalid Hugging Face response: {e}")),
        Err(ureq::Error::Status(code, resp)) => {
            let hint = if code == 429 {
                " Rate limit reached; try again in a few minutes."
            } else if code == 401 || code == 403 {
                " This repository is gated or private: set HF_TOKEN to a token with read access."
            } else {
                ""
            };
            let body = resp.into_string().unwrap_or_default();
            Err(format!(
                "Hugging Face returned HTTP {code}.{hint} {}",
                body.chars().take(180).collect::<String>()
            ))
        }
        Err(e) => Err(format!("network error: {e}")),
    }
}

fn encode_path(path: &str) -> String {
    path.split('/')
        .map(|part| urlencoding::encode(part).into_owned())
        .collect::<Vec<_>>()
        .join("/")
}
