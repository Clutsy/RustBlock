//! Parallel, resumable HTTP downloads used by the CLI and the desktop app.
//!
//! A single file can use up to 16 range connections; a batch is capped at 8
//! concurrent files. Downloads are written to temporary `.part` files and only
//! renamed after size and optional SHA-256 checks succeed.

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use sha2::{Digest, Sha256};

#[derive(Debug, Clone, thiserror::Error)]
pub enum DownloadError {
    #[error("HTTP request failed: {0}")]
    Http(String),
    #[error("I/O error: {0}")]
    Io(String),
    #[error("download cancelled")]
    Cancelled,
    #[error("server did not report a size; cannot resume or split this download")]
    UnknownSize,
    #[error("download ended at {actual} bytes, expected {expected}")]
    Truncated { actual: u64, expected: u64 },
    #[error("SHA-256 mismatch: expected {expected}, got {actual}")]
    HashMismatch { expected: String, actual: String },
}

#[derive(Debug, Clone)]
pub struct DownloadOptions {
    pub connections: usize,
    pub force: bool,
    pub expected_sha256: Option<String>,
    pub token: Option<String>,
}

impl Default for DownloadOptions {
    fn default() -> Self {
        Self {
            connections: 16,
            force: false,
            expected_sha256: None,
            token: std::env::var("HF_TOKEN")
                .ok()
                .filter(|value| !value.trim().is_empty()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DownloadReport {
    pub path: PathBuf,
    pub bytes: u64,
    pub sha256: String,
    pub resumed: bool,
}

#[derive(Debug, Clone)]
pub enum BatchDownloadEvent {
    Started {
        index: usize,
        url: String,
        destination: PathBuf,
    },
    Progress {
        index: usize,
        done: u64,
        total: u64,
    },
    Finished {
        index: usize,
        result: Result<DownloadReport, DownloadError>,
    },
}

pub fn download_url(
    url: &str,
    destination: &Path,
    options: &DownloadOptions,
    cancel: &AtomicBool,
) -> Result<DownloadReport, DownloadError> {
    download_impl(url, destination, options, cancel, None)
}

/// Same as `download_url`, with a `(bytes_done, bytes_total)` callback used by
/// the desktop app to drive its transfer panel. The callback is shared by all
/// range workers and must be cheap and thread-safe (it usually sends to a
/// channel).
pub fn download_url_with_progress(
    url: &str,
    destination: &Path,
    options: &DownloadOptions,
    cancel: &AtomicBool,
    on_progress: &(dyn Fn(u64, u64) + Sync),
) -> Result<DownloadReport, DownloadError> {
    download_impl(url, destination, options, cancel, Some(on_progress))
}

fn download_impl(
    url: &str,
    destination: &Path,
    options: &DownloadOptions,
    cancel: &AtomicBool,
    on_progress: Option<&(dyn Fn(u64, u64) + Sync)>,
) -> Result<DownloadReport, DownloadError> {
    if destination.exists() && !options.force {
        let bytes = destination.metadata().map(|m| m.len()).unwrap_or(0);
        let sha256 = sha256(destination)?;
        if let Some(expected) = &options.expected_sha256 {
            if !sha256.eq_ignore_ascii_case(expected.trim()) {
                return Err(DownloadError::HashMismatch {
                    expected: expected.clone(),
                    actual: sha256,
                });
            }
        }
        if let Some(callback) = on_progress {
            callback(bytes, bytes);
        }
        return Ok(DownloadReport {
            path: destination.to_path_buf(),
            bytes,
            sha256,
            resumed: true,
        });
    }
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent).map_err(|error| DownloadError::Io(error.to_string()))?;
    }
    let head = request_head(url, options)?;
    let total = head.length.ok_or(DownloadError::UnknownSize)?;
    if cancel.load(Ordering::Relaxed) {
        return Err(DownloadError::Cancelled);
    }
    let connections = if head.ranges {
        options.connections.clamp(1, 16)
    } else {
        1
    };
    let partial = destination.with_extension("part");
    let resumed = connections == 1
        && partial.exists()
        && partial.metadata().map(|m| m.len()).unwrap_or(0) > 0;
    if resumed {
        let already = partial.metadata().map(|m| m.len()).unwrap_or(0);
        download_single(url, &partial, already, total, options, cancel, on_progress)?;
    } else {
        if partial.exists() {
            fs::remove_file(&partial).map_err(|error| DownloadError::Io(error.to_string()))?;
        }
        if connections == 1 {
            download_single(url, &partial, 0, total, options, cancel, on_progress)?;
        } else {
            download_ranges(
                url,
                &partial,
                total,
                connections,
                options,
                cancel,
                on_progress,
            )?;
        }
    }
    let actual = partial.metadata().map(|m| m.len()).unwrap_or(0);
    if total > 0 && actual != total {
        return Err(DownloadError::Truncated {
            actual,
            expected: total,
        });
    }
    let sha256 = sha256(&partial)?;
    if let Some(expected) = &options.expected_sha256 {
        if !sha256.eq_ignore_ascii_case(expected.trim()) {
            let _ = fs::remove_file(&partial);
            return Err(DownloadError::HashMismatch {
                expected: expected.clone(),
                actual: sha256,
            });
        }
    }
    if destination.exists() {
        fs::remove_file(destination).map_err(|error| DownloadError::Io(error.to_string()))?;
    }
    fs::rename(&partial, destination).map_err(|error| DownloadError::Io(error.to_string()))?;
    if let Some(callback) = on_progress {
        callback(total, total);
    }
    Ok(DownloadReport {
        path: destination.to_path_buf(),
        bytes: total,
        sha256,
        resumed,
    })
}

struct HeadInfo {
    length: Option<u64>,
    ranges: bool,
}

fn request_head(url: &str, options: &DownloadOptions) -> Result<HeadInfo, DownloadError> {
    let mut request = ureq::head(url).set("User-Agent", "RustBlock/0.1");
    if let Some(token) = &options.token {
        request = request.set("Authorization", &format!("Bearer {token}"));
    }
    let response = request
        .call()
        .map_err(|error| DownloadError::Http(error.to_string()))?;
    let ranges = response
        .header("accept-ranges")
        .is_some_and(|value| value.to_ascii_lowercase().contains("bytes"));
    let length = response
        .header("content-length")
        .and_then(|value| value.parse().ok());
    Ok(HeadInfo { length, ranges })
}

fn request_range(
    url: &str,
    start: u64,
    end: u64,
    options: &DownloadOptions,
    on_bytes: &(dyn Fn(u64) + Sync),
) -> Result<Vec<u8>, DownloadError> {
    let mut request = ureq::get(url)
        .set("User-Agent", "RustBlock/0.1")
        .set("Range", &format!("bytes={start}-{end}"));
    if let Some(token) = &options.token {
        request = request.set("Authorization", &format!("Bearer {token}"));
    }
    let response = request
        .call()
        .map_err(|error| DownloadError::Http(error.to_string()))?;
    let mut reader = response.into_reader();
    let mut data = Vec::new();
    let mut buffer = vec![0u8; 256 * 1024];
    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|error| DownloadError::Io(error.to_string()))?;
        if read == 0 {
            break;
        }
        data.extend_from_slice(&buffer[..read]);
        on_bytes(read as u64);
    }
    Ok(data)
}

fn download_single(
    url: &str,
    partial: &Path,
    start: u64,
    total: u64,
    options: &DownloadOptions,
    cancel: &AtomicBool,
    on_progress: Option<&(dyn Fn(u64, u64) + Sync)>,
) -> Result<(), DownloadError> {
    let mut request = ureq::get(url).set("User-Agent", "RustBlock/0.1");
    if start > 0 {
        request = request.set("Range", &format!("bytes={start}-"));
    }
    if let Some(token) = &options.token {
        request = request.set("Authorization", &format!("Bearer {token}"));
    }
    let response = request
        .call()
        .map_err(|error| DownloadError::Http(error.to_string()))?;
    let mut reader = response.into_reader();
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .append(start > 0)
        .truncate(start == 0)
        .open(partial)
        .map_err(|error| DownloadError::Io(error.to_string()))?;
    let mut buffer = vec![0u8; 256 * 1024];
    let mut written = start;
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err(DownloadError::Cancelled);
        }
        let read = reader
            .read(&mut buffer)
            .map_err(|error| DownloadError::Io(error.to_string()))?;
        if read == 0 {
            break;
        }
        file.write_all(&buffer[..read])
            .map_err(|error| DownloadError::Io(error.to_string()))?;
        written += read as u64;
        if let Some(callback) = on_progress {
            callback(written, total.max(written));
        }
    }
    if total > 0 && written != total {
        return Err(DownloadError::Truncated {
            actual: written,
            expected: total,
        });
    }
    Ok(())
}

fn download_ranges(
    url: &str,
    partial: &Path,
    total: u64,
    connections: usize,
    options: &DownloadOptions,
    cancel: &AtomicBool,
    on_progress: Option<&(dyn Fn(u64, u64) + Sync)>,
) -> Result<(), DownloadError> {
    let chunk = total.div_ceil(connections as u64);
    let mut jobs = Vec::new();
    for index in 0..connections {
        let start = index as u64 * chunk;
        if start >= total {
            break;
        }
        let end = ((index as u64 + 1) * chunk).min(total) - 1;
        jobs.push((index, start, end));
    }
    let results: Arc<Mutex<HashMap<usize, Vec<u8>>>> = Arc::new(Mutex::new(HashMap::new()));
    let failure = Arc::new(Mutex::new(None::<String>));
    let done = Arc::new(std::sync::atomic::AtomicU64::new(0));
    std::thread::scope(|scope| {
        for (index, start, end) in &jobs {
            let results = results.clone();
            let failure = failure.clone();
            let done = Arc::clone(&done);
            let url = url.to_string();
            let options = options.clone();
            scope.spawn(move || {
                if cancel.load(Ordering::Relaxed) {
                    *failure.lock().expect("failure lock") = Some("cancelled".to_string());
                    return;
                }
                let report = |bytes: u64| {
                    let complete = done.fetch_add(bytes, Ordering::Relaxed) + bytes;
                    if let Some(callback) = on_progress {
                        callback(complete, total);
                    }
                };
                match request_range(&url, *start, *end, &options, &report) {
                    Ok(data) => {
                        results.lock().expect("results lock").insert(*index, data);
                    }
                    Err(error) => {
                        *failure.lock().expect("failure lock") = Some(error.to_string());
                    }
                }
            });
        }
    });
    if let Some(error) = failure.lock().expect("failure lock").take() {
        if error == "cancelled" {
            return Err(DownloadError::Cancelled);
        }
        return Err(DownloadError::Http(error));
    }
    let mut file = File::create(partial).map_err(|error| DownloadError::Io(error.to_string()))?;
    let results = results.lock().expect("results lock");
    for (index, start, end) in jobs {
        let data = results
            .get(&index)
            .ok_or_else(|| DownloadError::Http("missing chunk".into()))?;
        file.seek(SeekFrom::Start(start))
            .map_err(|error| DownloadError::Io(error.to_string()))?;
        file.write_all(data)
            .map_err(|error| DownloadError::Io(error.to_string()))?;
        if data.len() as u64 != end - start + 1 {
            return Err(DownloadError::Truncated {
                actual: data.len() as u64,
                expected: end - start + 1,
            });
        }
    }
    Ok(())
}

pub fn download_many(
    jobs: &[(String, PathBuf)],
    max_files: usize,
    options: &DownloadOptions,
    cancel: &AtomicBool,
) -> Vec<Result<DownloadReport, DownloadError>> {
    download_many_with_events(jobs, max_files, options, cancel, &|_| {})
}

/// Batch download with a callback for start, byte progress and completion.
/// Events from concurrently downloaded files can interleave; `index` maps each
/// event back to the input job. The returned vector preserves input order.
pub fn download_many_with_events(
    jobs: &[(String, PathBuf)],
    max_files: usize,
    options: &DownloadOptions,
    cancel: &AtomicBool,
    on_event: &(dyn Fn(BatchDownloadEvent) + Sync),
) -> Vec<Result<DownloadReport, DownloadError>> {
    let mut results: Vec<Option<Result<DownloadReport, DownloadError>>> =
        (0..jobs.len()).map(|_| None).collect();
    let indexed: Vec<_> = jobs.iter().enumerate().collect();
    for group in indexed.chunks(max_files.clamp(1, 8)) {
        std::thread::scope(|scope| {
            let handles: Vec<_> = group
                .iter()
                .map(|(index, (url, destination))| {
                    scope.spawn(move || {
                        on_event(BatchDownloadEvent::Started {
                            index: *index,
                            url: (*url).clone(),
                            destination: (*destination).clone(),
                        });
                        let result = download_url_with_progress(
                            url,
                            destination,
                            options,
                            cancel,
                            &|done, total| {
                                on_event(BatchDownloadEvent::Progress {
                                    index: *index,
                                    done,
                                    total,
                                });
                            },
                        );
                        on_event(BatchDownloadEvent::Finished {
                            index: *index,
                            result: result.clone(),
                        });
                        (*index, result)
                    })
                })
                .collect();
            for handle in handles {
                if let Ok((index, result)) = handle.join() {
                    results[index] = Some(result);
                }
            }
        });
    }
    results
        .into_iter()
        .map(|result| {
            result.unwrap_or_else(|| Err(DownloadError::Http("download worker panicked".into())))
        })
        .collect()
}

fn sha256(path: &Path) -> Result<String, DownloadError> {
    let mut file = File::open(path).map_err(|error| DownloadError::Io(error.to_string()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| DownloadError::Io(error.to_string()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Download the optional vision projector next to its model. The caller passes
/// the repository's projector URL; RustBlock does not guess multimodal
/// compatibility from a filename.
pub fn download_mmproj(
    projector_url: &str,
    model: &Path,
    options: &DownloadOptions,
    cancel: &AtomicBool,
) -> Result<DownloadReport, DownloadError> {
    let name = projector_url
        .rsplit('/')
        .next()
        .filter(|name| name.to_ascii_lowercase().ends_with(".gguf"))
        .unwrap_or("mmproj.gguf");
    let destination = model.parent().unwrap_or_else(|| Path::new(".")).join(name);
    download_url(projector_url, &destination, options, cancel)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::time::Duration;

    fn serve(mut stream: TcpStream, data: &[u8]) {
        let mut buffer = [0u8; 8192];
        let mut request = String::new();
        loop {
            let Ok(read) = stream.read(&mut buffer) else {
                return;
            };
            if read == 0 {
                break;
            }
            request.push_str(&String::from_utf8_lossy(&buffer[..read]));
            if request.contains("\r\n\r\n") || request.len() >= 8192 {
                break;
            }
        }
        let first = request.lines().next().unwrap_or_default();
        let is_head = first.starts_with("HEAD ");
        let range = request.lines().find_map(|line| {
            line.strip_prefix("Range: bytes=")
                .and_then(|value| value.split_once('-'))
        });
        if is_head {
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nAccept-Ranges: bytes\r\nConnection: close\r\n\r\n",
                data.len()
            );
            let _ = stream.write_all(response.as_bytes());
            return;
        }
        let (start, end) = range
            .and_then(|(start, end)| {
                let start = start.parse::<usize>().ok()?;
                let end = end.parse::<usize>().ok()?;
                Some((start, end.min(data.len().saturating_sub(1))))
            })
            .unwrap_or((0, data.len().saturating_sub(1)));
        let body = &data[start..=end];
        let status = if range.is_some() {
            "206 Partial Content"
        } else {
            "200 OK"
        };
        let response = format!(
            "HTTP/1.1 {status}\r\nContent-Length: {}\r\nAccept-Ranges: bytes\r\nContent-Range: bytes {}-{}/{}\r\nConnection: close\r\n\r\n",
            body.len(),
            start,
            end,
            data.len()
        );
        let _ = stream.write_all(response.as_bytes());
        let _ = stream.write_all(body);
    }

    fn server(data: Vec<u8>) -> (String, Arc<AtomicBool>, std::thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let address = listener.local_addr().unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_thread = stop.clone();
        let data = Arc::new(data);
        let handle = std::thread::spawn(move || {
            while !stop_thread.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((stream, _)) => serve(stream, &data),
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    Err(_) => break,
                }
            }
        });
        (format!("http://{address}/model.gguf"), stop, handle)
    }

    #[test]
    fn parallel_download_verifies_the_expected_hash() {
        let data = (0..524_288u32)
            .map(|value| (value % 251) as u8)
            .collect::<Vec<_>>();
        let expected = format!("{:x}", Sha256::digest(&data));
        let (url, stop, handle) = server(data.clone());
        let dir = tempfile::tempdir().unwrap();
        let destination = dir.path().join("model.gguf");
        let cancel = AtomicBool::new(false);
        let report = download_url(
            &url,
            &destination,
            &DownloadOptions {
                connections: 4,
                expected_sha256: Some(expected),
                token: None,
                ..DownloadOptions::default()
            },
            &cancel,
        )
        .unwrap();
        stop.store(true, Ordering::Relaxed);
        handle.join().unwrap();
        assert_eq!(report.path, destination);
        assert_eq!(report.bytes, data.len() as u64);
        assert_eq!(std::fs::read(&destination).unwrap(), data);
    }

    #[test]
    fn a_bad_hash_is_reported_and_leaves_no_file() {
        let data = vec![7u8; 64 * 1024];
        let (url, stop, handle) = server(data);
        let dir = tempfile::tempdir().unwrap();
        let destination = dir.path().join("model.gguf");
        let cancel = AtomicBool::new(false);
        let error = download_url(
            &url,
            &destination,
            &DownloadOptions {
                connections: 1,
                expected_sha256: Some("00".into()),
                token: None,
                ..DownloadOptions::default()
            },
            &cancel,
        )
        .unwrap_err();
        stop.store(true, Ordering::Relaxed);
        handle.join().unwrap();
        assert!(
            matches!(error, DownloadError::HashMismatch { .. }),
            "unexpected error: {error:?}"
        );
        assert!(!destination.exists());
        assert!(!destination.with_extension("part").exists());
    }

    #[test]
    fn batch_events_cover_each_file() {
        let data = (0..524_288u32)
            .map(|value| (value % 251) as u8)
            .collect::<Vec<_>>();
        let (url, stop, handle) = server(data.clone());
        let dir = tempfile::tempdir().unwrap();
        let jobs = vec![
            (url.clone(), dir.path().join("a.gguf")),
            (url, dir.path().join("b.gguf")),
        ];
        let cancel = AtomicBool::new(false);
        let started = std::sync::atomic::AtomicUsize::new(0);
        let finished = std::sync::atomic::AtomicUsize::new(0);
        let progress = std::sync::atomic::AtomicUsize::new(0);
        let results = download_many_with_events(
            &jobs,
            2,
            &DownloadOptions {
                connections: 2,
                token: None,
                ..DownloadOptions::default()
            },
            &cancel,
            &|event| match event {
                BatchDownloadEvent::Started { .. } => {
                    started.fetch_add(1, Ordering::Relaxed);
                }
                BatchDownloadEvent::Progress { .. } => {
                    progress.fetch_add(1, Ordering::Relaxed);
                }
                BatchDownloadEvent::Finished { .. } => {
                    finished.fetch_add(1, Ordering::Relaxed);
                }
            },
        );
        stop.store(true, Ordering::Relaxed);
        handle.join().unwrap();
        assert_eq!(started.load(Ordering::Relaxed), 2);
        assert_eq!(finished.load(Ordering::Relaxed), 2);
        assert!(progress.load(Ordering::Relaxed) >= 1);
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(Result::is_ok));
        assert_eq!(std::fs::read(&jobs[0].1).unwrap(), data);
        assert_eq!(std::fs::read(&jobs[1].1).unwrap(), data);
    }
}
