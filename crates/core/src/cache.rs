//! Model-download cache inventory and quota enforcement.
//!
//! The index is rebuilt from the download directory so it cannot drift from
//! the files on disk. Only model artifacts are considered; unrelated files are
//! never removed.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct CacheEntry {
    pub path: PathBuf,
    pub bytes: u64,
    pub modified: SystemTime,
}

#[derive(Debug, Clone)]
pub struct CacheStats {
    pub entries: Vec<CacheEntry>,
    pub total_bytes: u64,
    pub quota_bytes: u64,
}

impl CacheStats {
    pub fn over_quota(&self) -> bool {
        self.total_bytes > self.quota_bytes
    }
}

#[derive(Debug, Clone, Default)]
pub struct CacheTrimReport {
    pub removed: usize,
    pub freed_bytes: u64,
    pub remaining_bytes: u64,
}

fn is_model_artifact(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "gguf" | "safetensors" | "part" | "bin" | "pt" | "pth"
            )
        })
}

pub fn cache_index(dir: &Path) -> Result<Vec<CacheEntry>, String> {
    let entries =
        fs::read_dir(dir).map_err(|error| format!("cannot read {}: {error}", dir.display()))?;
    let mut index = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|error| format!("cannot read cache entry: {error}"))?;
        let path = entry.path();
        if !entry
            .file_type()
            .map_err(|error| format!("cannot inspect {}: {error}", path.display()))?
            .is_file()
            || !is_model_artifact(&path)
        {
            continue;
        }
        let metadata = entry
            .metadata()
            .map_err(|error| format!("cannot inspect {}: {error}", path.display()))?;
        index.push(CacheEntry {
            path,
            bytes: metadata.len(),
            modified: metadata.modified().unwrap_or(UNIX_EPOCH),
        });
    }
    index.sort_by(|a, b| {
        a.modified
            .cmp(&b.modified)
            .then_with(|| a.path.cmp(&b.path))
    });
    Ok(index)
}

pub fn cache_stats(dir: &Path, quota_bytes: u64) -> Result<CacheStats, String> {
    if !dir.exists() {
        return Ok(CacheStats {
            entries: Vec::new(),
            total_bytes: 0,
            quota_bytes,
        });
    }
    let entries = cache_index(dir)?;
    let total_bytes = entries
        .iter()
        .fold(0u64, |total, entry| total.saturating_add(entry.bytes));
    Ok(CacheStats {
        entries,
        total_bytes,
        quota_bytes,
    })
}

/// Remove the oldest model artifacts until the directory fits `quota_bytes`.
/// A missing directory is an empty cache, not an error.
pub fn trim_cache(dir: &Path, quota_bytes: u64) -> Result<CacheTrimReport, String> {
    if !dir.exists() {
        return Ok(CacheTrimReport::default());
    }
    let mut entries = cache_index(dir)?;
    let mut total = entries
        .iter()
        .fold(0u64, |total, entry| total.saturating_add(entry.bytes));
    let mut report = CacheTrimReport::default();
    for entry in entries.drain(..) {
        if total <= quota_bytes {
            break;
        }
        fs::remove_file(&entry.path)
            .map_err(|error| format!("cannot remove {}: {error}", entry.path.display()))?;
        total = total.saturating_sub(entry.bytes);
        report.removed += 1;
        report.freed_bytes = report.freed_bytes.saturating_add(entry.bytes);
    }
    report.remaining_bytes = total;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indexes_only_model_artifacts() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("model.gguf"), b"1234").unwrap();
        fs::write(dir.path().join("notes.txt"), b"ignore").unwrap();
        fs::write(dir.path().join("cache.part"), b"12").unwrap();
        let stats = cache_stats(dir.path(), 1024).unwrap();
        assert_eq!(stats.entries.len(), 2);
        assert_eq!(stats.total_bytes, 6);
    }

    #[test]
    fn trims_oldest_entries_to_quota() {
        let dir = tempfile::tempdir().unwrap();
        let old = dir.path().join("old.gguf");
        let new = dir.path().join("new.gguf");
        fs::write(&old, b"12345678").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        fs::write(&new, b"abcdefgh").unwrap();
        let report = trim_cache(dir.path(), 8).unwrap();
        assert_eq!(report.removed, 1);
        assert_eq!(report.remaining_bytes, 8);
        assert!(!old.exists());
        assert!(new.exists());
    }

    #[test]
    fn missing_directory_is_an_empty_cache() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("missing");
        let stats = cache_stats(&missing, 10).unwrap();
        assert_eq!(stats.total_bytes, 0);
        assert!(!stats.over_quota());
        let report = trim_cache(&missing, 10).unwrap();
        assert_eq!(report.removed, 0);
    }
}
