//! OTA updates through GitHub Releases.
//!
//! The app checks the latest release on `Clutsy/RustBlock`, compares its tag
//! with the running version and, when a newer Windows installer is published,
//! downloads it to a temporary folder and hands it to Inno Setup. The installer
//! is per-user, so the silent update does not require an elevation prompt.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use serde_json::Value;

pub const REPO: &str = "Clutsy/RustBlock";
pub const RELEASES_URL: &str = "https://github.com/Clutsy/RustBlock/releases/latest";
const USER_AGENT: &str = concat!("RustBlock/", env!("CARGO_PKG_VERSION"));
const INSTALLER_PREFIX: &str = "RustBlock-Setup-";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpdateInfo {
    pub version: String,
    pub notes: String,
    pub page_url: String,
    pub installer_url: Option<String>,
    pub installer_name: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Check {
    UpToDate,
    Available(UpdateInfo),
}

#[derive(Debug)]
pub enum UpdateEvent {
    Checked(Result<Check, String>),
    Downloaded(Result<PathBuf, String>),
}

/// Ask GitHub for the latest published release. A 404 simply means that no
/// release exists yet, which is not an error for a fresh repository.
pub fn check_latest() -> Result<Check, String> {
    let url = format!("https://api.github.com/repos/{REPO}/releases/latest");
    match ureq::get(&url)
        .set("User-Agent", USER_AGENT)
        .set("Accept", "application/vnd.github+json")
        .timeout(Duration::from_secs(20))
        .call()
    {
        Ok(response) => {
            let value: Value = response
                .into_json()
                .map_err(|error| format!("Invalid GitHub release response: {error}"))?;
            parse_release(&value, env!("CARGO_PKG_VERSION"))
        }
        Err(ureq::Error::Status(404, _)) => Ok(Check::UpToDate),
        Err(ureq::Error::Status(code, _)) => Err(format!("GitHub returned HTTP {code}.")),
        Err(error) => Err(format!("Update check failed: {error}")),
    }
}

fn parse_release(value: &Value, current: &str) -> Result<Check, String> {
    let tag = value
        .get("tag_name")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|tag| !tag.is_empty())
        .ok_or_else(|| "The GitHub release has no tag.".to_string())?;
    let version = tag
        .trim_start_matches('v')
        .trim_start_matches('V')
        .to_string();
    let page_url = value
        .get("html_url")
        .and_then(Value::as_str)
        .unwrap_or(RELEASES_URL)
        .to_string();
    let notes = value
        .get("body")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();

    let installer = value
        .get("assets")
        .and_then(Value::as_array)
        .and_then(|assets| {
            assets.iter().find_map(|asset| {
                let name = asset.get("name")?.as_str()?;
                if !(name.starts_with(INSTALLER_PREFIX) && name.ends_with(".exe")) {
                    return None;
                }
                let url = asset.get("browser_download_url")?.as_str()?;
                Some((name.to_string(), url.to_string()))
            })
        });

    if version_is_newer(&version, current) {
        Ok(Check::Available(UpdateInfo {
            version,
            notes,
            page_url,
            installer_url: installer.as_ref().map(|(_, url)| url.clone()),
            installer_name: installer.map(|(name, _)| name),
        }))
    } else {
        Ok(Check::UpToDate)
    }
}

/// Numeric semantic-version comparison. Pre-release/build suffixes are ignored,
/// which is enough for RustBlock's `v0.1.0` style tags.
pub fn version_is_newer(latest: &str, current: &str) -> bool {
    let latest = version_parts(latest);
    let current = version_parts(current);
    let len = latest.len().max(current.len());
    for index in 0..len {
        let new = latest.get(index).copied().unwrap_or(0);
        let old = current.get(index).copied().unwrap_or(0);
        if new != old {
            return new > old;
        }
    }
    false
}

fn version_parts(value: &str) -> Vec<u64> {
    value
        .trim_start_matches('v')
        .trim_start_matches('V')
        .split(['-', '+'])
        .next()
        .unwrap_or_default()
        .split('.')
        .filter_map(|part| part.parse::<u64>().ok())
        .collect()
}

pub fn download_installer(info: &UpdateInfo) -> Result<PathBuf, String> {
    let url = info
        .installer_url
        .as_deref()
        .ok_or_else(|| "This release has no Windows installer asset.".to_string())?;
    let raw_name = info
        .installer_name
        .as_deref()
        .unwrap_or("RustBlock-Setup.exe");
    let name = raw_name
        .rsplit(['\\', '/'])
        .next()
        .filter(|name| !name.is_empty())
        .unwrap_or("RustBlock-Setup.exe");
    if !name.to_ascii_lowercase().ends_with(".exe") {
        return Err("The update asset is not an executable installer.".to_string());
    }

    let folder = std::env::temp_dir().join("RustBlock");
    fs::create_dir_all(&folder)
        .map_err(|error| format!("Cannot create {}: {error}", folder.display()))?;
    let path = folder.join(name);

    let response = ureq::get(url)
        .set("User-Agent", USER_AGENT)
        .timeout(Duration::from_secs(300))
        .call()
        .map_err(|error| format!("Download failed: {error}"))?;
    let mut reader = response.into_reader();
    let mut file = fs::File::create(&path)
        .map_err(|error| format!("Cannot write {}: {error}", path.display()))?;
    let copied = io::copy(&mut reader, &mut file)
        .map_err(|error| format!("Download interrupted: {error}"))?;
    if copied < 1_000_000 {
        let _ = fs::remove_file(&path);
        return Err("The installer download is unexpectedly small.".to_string());
    }
    Ok(path)
}

/// Start the per-user Inno Setup installer. `/CLOSEAPPLICATIONS` lets it close
/// the running app, and `/RESTARTAPPLICATIONS` opens the new build afterwards.
pub fn run_installer(path: &Path) -> Result<(), String> {
    if !path.is_file() {
        return Err(format!("Installer not found: {}", path.display()));
    }
    #[cfg(windows)]
    {
        Command::new(path)
            .args([
                "/SILENT",
                "/CLOSEAPPLICATIONS",
                "/RESTARTAPPLICATIONS",
                "/NORESTART",
            ])
            .spawn()
            .map_err(|error| format!("Cannot start the installer: {error}"))?;
        Ok(())
    }
    #[cfg(not(windows))]
    {
        let _ = path;
        Err("In-app updates are only supported on Windows.".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compares_numeric_versions() {
        assert!(version_is_newer("0.2.0", "0.1.9"));
        assert!(version_is_newer("v1.0", "0.9.9"));
        assert!(version_is_newer("0.1.1", "0.1.0"));
        assert!(!version_is_newer("0.1.0", "0.1.0"));
        assert!(!version_is_newer("0.0.9", "0.1.0"));
        assert!(!version_is_newer("0.1.0-beta", "0.1.0"));
    }

    #[test]
    fn reads_the_windows_installer_asset() {
        let release = serde_json::json!({
            "tag_name": "v0.2.0",
            "name": "RustBlock 0.2.0",
            "body": "Multicore verification and OTA updates.",
            "html_url": "https://github.com/Clutsy/RustBlock/releases/tag/v0.2.0",
            "assets": [
                { "name": "RustBlock-0.2.0-portable.zip", "browser_download_url": "https://example.com/portable.zip" },
                { "name": "RustBlock-Setup-0.2.0.exe", "browser_download_url": "https://example.com/setup.exe" }
            ]
        });
        let check = parse_release(&release, "0.1.0").unwrap();
        let Check::Available(info) = check else {
            panic!("expected an available update");
        };
        assert_eq!(info.version, "0.2.0");
        assert_eq!(
            info.installer_name.as_deref(),
            Some("RustBlock-Setup-0.2.0.exe")
        );
        assert_eq!(
            info.installer_url.as_deref(),
            Some("https://example.com/setup.exe")
        );
    }

    #[test]
    fn ignores_the_current_release() {
        let release = serde_json::json!({
            "tag_name": "v0.1.0",
            "assets": []
        });
        assert_eq!(parse_release(&release, "0.1.0").unwrap(), Check::UpToDate);
    }
}
