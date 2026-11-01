//! Hugging Face Hub client for GGUF discovery and downloads.
//!
//! Search results are enriched from the public model API (author, license,
//! base model, tags, file count). Selecting a repository fetches its model
//! card, README summary, architecture facts and the real GGUF file list.
//! Setting `HF_TOKEN` enables gated/private repositories.
//! Downloads are resumable through `.part` files and byte ranges.

use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::time::Instant;

use serde_json::Value;

const BASE: &str = "https://huggingface.co";
const UA: &str = "RustBlock/0.1 (GGUF conversion desktop app)";

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

    let (segments, from_hf): (Vec<&str>, bool) = if is_hf_host {
        (path.split('/').filter(|s| !s.is_empty()).collect(), true)
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
    let url = format!("{BASE}/api/models/{}", encode_path(repo));
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
}

#[derive(Debug, Clone)]
pub struct HfModel {
    pub id: String,
    pub author: String,
    pub name: String,
    pub downloads: u64,
    pub likes: u64,
    pub created_at: String,
    pub last_modified: String,
    pub gated: bool,
    pub pipeline: String,
    pub library: String,
    pub tags: Vec<String>,
    pub license: Option<String>,
    pub base_model: Option<String>,
    pub sha: String,
    pub gguf_files: usize,
}

impl HfModel {
    pub fn web_url(&self) -> String {
        format!("{BASE}/{}", self.id)
    }
}

#[derive(Debug, Clone)]
pub struct HfFile {
    pub path: String,
    pub size: u64,
}

#[derive(Debug, Clone, Default)]
pub struct HfDetails {
    pub architecture: Option<String>,
    pub context_length: Option<u64>,
    pub total_gguf: Option<u64>,
    pub used_storage: Option<u64>,
    pub license: Option<String>,
    pub base_model: Option<String>,
    pub languages: Vec<String>,
    pub description: Option<String>,
}

#[derive(Debug, Clone)]
pub enum HfEvent {
    SearchStarted,
    SearchResults(Vec<HfModel>),
    DetailsStarted(String),
    DetailsLoaded {
        repo: String,
        details: HfDetails,
        files: Vec<HfFile>,
    },
    DetailsError {
        repo: String,
        error: String,
    },
    Error(String),
    DownloadStarted {
        repo: String,
        file: String,
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
        "{BASE}/api/models?filter=gguf&sort={}&direction=-1&limit=50&full=true{search}",
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
    let tags = item
        .get("tags")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let license = card_license(item).or_else(|| {
        tags.iter()
            .find_map(|tag| tag.strip_prefix("license:").map(str::to_string))
    });
    let base_model = card_base_model(item).or_else(|| {
        tags.iter()
            .find(|tag| tag.starts_with("base_model:") && !tag.starts_with("base_model:quantized:"))
            .and_then(|tag| tag.strip_prefix("base_model:"))
            .map(str::to_string)
    });
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
    Some(HfModel {
        id,
        author,
        name,
        downloads: u64_field(item, "downloads").unwrap_or(0),
        likes: u64_field(item, "likes").unwrap_or(0),
        created_at: str_field(item, "createdAt").unwrap_or_default().to_string(),
        last_modified: str_field(item, "lastModified")
            .unwrap_or_default()
            .to_string(),
        gated: gated_field(item),
        pipeline: str_field(item, "pipeline_tag")
            .unwrap_or_default()
            .to_string(),
        library: str_field(item, "library_name")
            .unwrap_or_default()
            .to_string(),
        tags,
        license,
        base_model,
        sha: str_field(item, "sha")
            .filter(|sha| !sha.is_empty())
            .unwrap_or("main")
            .to_string(),
        gguf_files,
    })
}

/// Fetch the full model card, README summary and the GGUF file list.
pub fn load_details(repo: &str, revision: &str) -> Result<(HfDetails, Vec<HfFile>), String> {
    let url = format!("{BASE}/api/models/{}", encode_path(repo));
    let value = get_json(&url)?;
    let mut details = HfDetails {
        architecture: value
            .get("gguf")
            .and_then(|g| g.get("architecture"))
            .and_then(Value::as_str)
            .map(str::to_string),
        context_length: value
            .get("gguf")
            .and_then(|g| g.get("context_length"))
            .and_then(Value::as_u64),
        total_gguf: value
            .get("gguf")
            .and_then(|g| g.get("total"))
            .and_then(Value::as_u64),
        used_storage: u64_field(&value, "usedStorage"),
        license: card_license(&value).or_else(|| {
            value
                .get("tags")
                .and_then(Value::as_array)
                .and_then(|tags| {
                    tags.iter().find_map(|tag| {
                        tag.as_str()
                            .and_then(|tag| tag.strip_prefix("license:"))
                            .map(str::to_string)
                    })
                })
        }),
        base_model: card_base_model(&value),
        languages: card_languages(&value),
        description: None,
    };
    if details.base_model.is_none() {
        details.base_model = value
            .get("tags")
            .and_then(Value::as_array)
            .and_then(|tags| {
                tags.iter().find_map(|tag| {
                    tag.as_str()
                        .filter(|tag| {
                            tag.starts_with("base_model:")
                                && !tag.starts_with("base_model:quantized:")
                        })
                        .and_then(|tag| tag.strip_prefix("base_model:"))
                        .map(str::to_string)
                })
            });
    }

    if let Ok(Some(readme)) = fetch_readme(repo, revision) {
        details.description = summarize_readme(&readme);
        if details.license.is_none() {
            details.license = readme_license(&readme);
        }
    }

    let files = list_files(repo, revision)?;
    Ok((details, files))
}

pub fn list_files(repo: &str, revision: &str) -> Result<Vec<HfFile>, String> {
    let rev = if revision.is_empty() {
        "main"
    } else {
        revision
    };
    let url = format!(
        "{BASE}/api/models/{}/tree/{}?recursive=true",
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
        out.push(HfFile {
            path: path.to_string(),
            size,
        });
    }
    out.sort_by_key(|f| f.size);
    Ok(out)
}

/// Downloads one file. Returns the final path.
pub fn download(
    repo: &str,
    revision: &str,
    file: &str,
    dest_dir: &Path,
    overwrite: bool,
    cancel: &AtomicBool,
    tx: &Sender<HfEvent>,
) -> Result<PathBuf, String> {
    fs::create_dir_all(dest_dir).map_err(|e| format!("cannot create download folder: {e}"))?;
    let name = file
        .rsplit('/')
        .next()
        .filter(|n| !n.is_empty())
        .unwrap_or("model.gguf");
    let final_path = dest_dir.join(name);
    if final_path.exists() && !overwrite {
        let _ = tx.send(HfEvent::Downloaded {
            repo: repo.to_string(),
            file: file.to_string(),
            path: final_path.clone(),
        });
        return Ok(final_path);
    }

    let partial = dest_dir.join(format!("{name}.part"));
    let mut start = partial.metadata().map(|m| m.len()).unwrap_or(0);
    if start > 0 && final_path.exists() && !overwrite {
        start = 0;
    }
    let url = format!(
        "{BASE}/{}/resolve/{}/{}\n",
        // Placeholder replaced below; keeping the URL construction in one place
        // makes the encoded path handling explicit.
        encode_path(repo),
        encode_path(if revision.is_empty() {
            "main"
        } else {
            revision
        }),
        encode_path(file)
    );
    let url = url.trim_end().to_string();

    let mut attempt = 0;
    loop {
        if cancel.load(Ordering::Relaxed) {
            let _ = tx.send(HfEvent::DownloadCancelled);
            return Err("download cancelled".to_string());
        }
        let mut request = ureq::get(&url).set("User-Agent", UA);
        if let Ok(token) = std::env::var("HF_TOKEN") {
            if !token.trim().is_empty() {
                request = request.set("Authorization", &format!("Bearer {}", token.trim()));
            }
        }
        if start > 0 {
            request = request.set("Range", &format!("bytes={start}-"));
        }
        let response = match request.call() {
            Ok(r) => r,
            Err(ureq::Error::Status(code, resp)) => {
                if code == 416 && start > 0 {
                    let _ = fs::remove_file(&partial);
                    start = 0;
                    attempt += 1;
                    if attempt < 2 {
                        continue;
                    }
                }
                let body = resp.into_string().unwrap_or_default();
                let hint = if code == 401 || code == 403 {
                    " This repository is gated or private: set HF_TOKEN to a token with read access."
                } else {
                    ""
                };
                return Err(format!(
                    "Hugging Face returned HTTP {code}.{hint} {}",
                    body.chars().take(180).collect::<String>()
                ));
            }
            Err(e) => return Err(format!("network error: {e}")),
        };

        let status = response.status();
        let content_len = response
            .header("Content-Length")
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0);
        if status == 200 && start > 0 {
            // The server ignored Range: restart from zero.
            start = 0;
        }
        let total = if status == 206 {
            start.saturating_add(content_len)
        } else {
            content_len
        };
        let _ = tx.send(HfEvent::DownloadStarted {
            repo: repo.to_string(),
            file: file.to_string(),
            total,
        });

        let mut reader = response.into_reader();
        let mut out_file = OpenOptions::new()
            .create(true)
            .write(true)
            .append(start > 0)
            .truncate(start == 0)
            .open(&partial)
            .map_err(|e| format!("cannot open partial download: {e}"))?;
        let mut done = start;
        let mut last_report = Instant::now();
        let mut last_done = done;
        let mut buf = vec![0u8; 128 * 1024];
        loop {
            if cancel.load(Ordering::Relaxed) {
                let _ = tx.send(HfEvent::DownloadCancelled);
                return Err("download cancelled".to_string());
            }
            match reader.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    out_file
                        .write_all(&buf[..n])
                        .map_err(|e| format!("cannot write download: {e}"))?;
                    done += n as u64;
                    let now = Instant::now();
                    let dt = now.duration_since(last_report).as_secs_f64();
                    if dt >= 0.12 {
                        let speed = (done - last_done) as f64 / dt;
                        let _ = tx.send(HfEvent::DownloadProgress {
                            done,
                            total: total.max(done),
                            speed,
                        });
                        last_report = now;
                        last_done = done;
                    }
                }
                Err(e) => return Err(format!("download interrupted: {e}")),
            }
        }
        out_file.flush().map_err(|e| e.to_string())?;
        drop(out_file);
        if total > 0 && done < total {
            return Err(format!(
                "download ended early ({done} of {total} bytes); click Download to resume"
            ));
        }
        if final_path.exists() {
            fs::remove_file(&final_path).map_err(|e| e.to_string())?;
        }
        fs::rename(&partial, &final_path).map_err(|e| e.to_string())?;
        let _ = tx.send(HfEvent::Downloaded {
            repo: repo.to_string(),
            file: file.to_string(),
            path: final_path.clone(),
        });
        return Ok(final_path);
    }
}

fn fetch_readme(repo: &str, revision: &str) -> Result<Option<String>, String> {
    let rev = if revision.is_empty() {
        "main"
    } else {
        revision
    };
    let url = format!(
        "{BASE}/{}/raw/{}/README.md",
        encode_path(repo),
        encode_path(rev)
    );
    let mut request = ureq::get(&url).set("User-Agent", UA);
    if let Ok(token) = std::env::var("HF_TOKEN") {
        if !token.trim().is_empty() {
            request = request.set("Authorization", &format!("Bearer {}", token.trim()));
        }
    }
    match request.call() {
        Ok(response) => Ok(Some(
            response
                .into_string()
                .map_err(|e| format!("invalid README response: {e}"))?,
        )),
        Err(ureq::Error::Status(404, _)) => Ok(None),
        Err(ureq::Error::Status(401 | 403, _)) => Ok(None),
        Err(ureq::Error::Status(code, _)) => Err(format!("README request returned HTTP {code}")),
        Err(e) => Err(format!("README network error: {e}")),
    }
}

fn summarize_readme(text: &str) -> Option<String> {
    let body = strip_frontmatter(text);
    let mut paragraph = String::new();
    for line in body.lines() {
        let line = line.trim();
        if line.is_empty() {
            if !paragraph.is_empty() {
                break;
            }
            continue;
        }
        if line.starts_with('#')
            || line.starts_with('!')
            || line.starts_with('<')
            || line.starts_with('|')
            || line.starts_with("```")
            || line.starts_with("---")
        {
            continue;
        }
        let clean = clean_markdown_line(line);
        if clean.is_empty() {
            continue;
        }
        if !paragraph.is_empty() {
            paragraph.push(' ');
        }
        paragraph.push_str(&clean);
        if paragraph.len() > 520 {
            break;
        }
    }
    let paragraph = paragraph.trim();
    if paragraph.is_empty() {
        None
    } else {
        Some(paragraph.chars().take(760).collect())
    }
}

fn strip_frontmatter(text: &str) -> &str {
    if let Some(rest) = text.strip_prefix("---") {
        if let Some(end) = rest.find("\n---") {
            return &rest[end + 4..];
        }
    }
    text
}

fn clean_markdown_line(line: &str) -> String {
    let mut out = String::new();
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '[' => {
                let mut label = String::new();
                for c2 in chars.by_ref() {
                    if c2 == ']' {
                        break;
                    }
                    label.push(c2);
                }
                if chars.peek() == Some(&'(') {
                    for c2 in chars.by_ref() {
                        if c2 == ')' {
                            break;
                        }
                    }
                }
                out.push_str(&label);
            }
            '<' => {
                for c2 in chars.by_ref() {
                    if c2 == '>' {
                        break;
                    }
                }
            }
            '*' | '_' | '`' | '\\' => {}
            _ => out.push(c),
        }
    }
    out.trim().to_string()
}

fn readme_license(text: &str) -> Option<String> {
    let front = text.strip_prefix("---")?;
    let end = front.find("\n---")?;
    for line in front[..end].lines() {
        if let Some(value) = line.strip_prefix("license:") {
            let value = value.trim().trim_matches('"').trim_matches('\'');
            if !value.is_empty() {
                return Some(value.to_string());
            }
        }
    }
    None
}

fn card_license(value: &Value) -> Option<String> {
    value
        .get("cardData")
        .and_then(|c| c.get("license"))
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn card_base_model(value: &Value) -> Option<String> {
    let base = value.get("cardData").and_then(|c| c.get("base_model"))?;
    if let Some(s) = base.as_str() {
        return Some(s.to_string());
    }
    base.as_array()
        .and_then(|a| a.iter().filter_map(Value::as_str).next())
        .map(str::to_string)
}

fn card_languages(value: &Value) -> Vec<String> {
    let mut out = Vec::new();
    let Some(lang) = value.get("cardData").and_then(|c| c.get("language")) else {
        return out;
    };
    if let Some(s) = lang.as_str() {
        out.push(s.to_string());
    } else if let Some(a) = lang.as_array() {
        out.extend(a.iter().filter_map(Value::as_str).map(str::to_string));
    }
    out
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
