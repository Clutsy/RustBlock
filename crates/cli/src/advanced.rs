use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};
use clap::Args;
use rustblock_core::{
    cache_stats, compare_quality, detect_hardware, diff, download_many_with_events, download_url,
    download_url_with_progress, estimate_fit, generate_with_llama_imatrix, human_bytes,
    install_ollama, merge, metadata_get, metadata_list, metadata_set, plan_budget, prune,
    quality_csv, quick_test, render_chat_template, run_kld, sha256_file, split, transcode,
    trim_cache, validate_metadata, verify_hash, write_lmstudio_instructions, BatchDownloadEvent,
    BudgetRequest, ConvertOptions, DownloadOptions, FitRequest, Hardware, ImportanceMatrix,
    VerifyMode,
};
use serde_json::json;

#[derive(Args)]
pub struct ImatrixArgs {
    /// Model used by llama-imatrix.
    pub model: PathBuf,
    /// Calibration text file.
    pub calibration: PathBuf,
    /// Destination legacy .imatrix file.
    pub output: PathBuf,
    /// Number of calibration chunks.
    #[arg(long, default_value_t = 100)]
    pub chunks: usize,
}

pub fn imatrix(args: ImatrixArgs) -> Result<()> {
    generate_with_llama_imatrix(&args.model, &args.calibration, &args.output, args.chunks)?;
    println!("wrote {}", args.output.display());
    println!("{}", rustblock_core::OVERFITTING_WARNING);
    Ok(())
}

#[derive(Args)]
pub struct QualityArgs {
    pub source: PathBuf,
    pub output: PathBuf,
    /// Optional importance matrix used for weighted SNR.
    #[arg(long)]
    pub imatrix: Option<PathBuf>,
    #[arg(long)]
    pub json: bool,
    /// Write the per-tensor CSV report.
    #[arg(long)]
    pub csv: Option<PathBuf>,
}

pub fn quality(args: QualityArgs) -> Result<()> {
    let matrix = args
        .imatrix
        .as_deref()
        .map(ImportanceMatrix::load)
        .transpose()
        .context("cannot load importance matrix")?;
    let report = compare_quality(&args.source, &args.output, matrix.as_ref())
        .context("cannot compare GGUF tensors")?;
    if let Some(path) = args.csv {
        fs::write(&path, quality_csv(&report))
            .with_context(|| format!("cannot write '{}'", path.display()))?;
    }
    if args.json {
        let tensors: Vec<_> = report
            .tensors
            .iter()
            .map(|t| {
                json!({
                    "name": t.name,
                    "source_type": t.source_type,
                    "output_type": t.output_type,
                    "elements": t.elements,
                    "rmse": t.rmse,
                    "snr_db": t.snr_db,
                    "cosine": t.cosine,
                    "weighted_snr_db": t.weighted_snr_db,
                })
            })
            .collect();
        let value = json!({
            "source": report.source,
            "output": report.output,
            "total_elements": report.total_elements,
            "mean_rmse": report.mean_rmse,
            "mean_snr_db": report.mean_snr_db,
            "mean_cosine": report.mean_cosine,
            "weighted_snr_db": report.weighted_snr_db,
            "tensors": tensors,
        });
        println!("{}", serde_json::to_string_pretty(&value)?);
        return Ok(());
    }
    println!(
        "mean RMSE {:.6}, mean SNR {:.2} dB, mean cosine {:.6}",
        report.mean_rmse, report.mean_snr_db, report.mean_cosine
    );
    if let Some(snr) = report.weighted_snr_db {
        println!("importance-weighted SNR: {snr:.2} dB");
    }
    println!("lowest-SNR tensors:");
    for tensor in report.sensitivity_ranking().into_iter().take(12) {
        println!(
            "  {:<48} {:>8.2} dB  cosine {:.5}  {} -> {}",
            tensor.name, tensor.snr_db, tensor.cosine, tensor.source_type, tensor.output_type
        );
    }
    Ok(())
}

#[derive(Args)]
pub struct FitArgs {
    pub file: PathBuf,
    /// Override the context length.
    #[arg(long)]
    pub context: Option<u64>,
    /// Bytes per KV cache element (2 = f16, 1 = q8_0-like).
    #[arg(long, default_value_t = 2)]
    pub cache_element_bytes: u64,
    /// Fixed runtime overhead in MiB.
    #[arg(long, default_value_t = 512)]
    pub overhead_mib: u64,
    /// Override detected system RAM in GiB.
    #[arg(long)]
    pub ram_gib: Option<f64>,
    /// Override detected VRAM in GiB.
    #[arg(long)]
    pub vram_gib: Option<f64>,
    #[arg(long)]
    pub json: bool,
}

pub fn fit(args: FitArgs) -> Result<()> {
    let detected = detect_hardware();
    let hardware = Hardware {
        ram_bytes: args
            .ram_gib
            .map(|gib| (gib * 1024.0 * 1024.0 * 1024.0) as u64)
            .or(detected.ram_bytes),
        vram_bytes: args
            .vram_gib
            .map(|gib| (gib * 1024.0 * 1024.0 * 1024.0) as u64)
            .or(detected.vram_bytes),
    };
    let request = FitRequest {
        context: args.context,
        cache_element_bytes: args.cache_element_bytes.max(1),
        overhead_bytes: args.overhead_mib.saturating_mul(1024 * 1024),
        hardware: Some(hardware),
    };
    let fit = estimate_fit(&args.file, request).context("cannot estimate GGUF memory use")?;
    if args.json {
        let value = json!({
            "file": fit.path,
            "architecture": fit.architecture,
            "layers": fit.layers,
            "context": fit.context,
            "kv_heads": fit.kv_heads,
            "head_bytes": fit.head_bytes,
            "weights_bytes": fit.weights_bytes,
            "kv_cache_bytes": fit.kv_cache_bytes,
            "overhead_bytes": fit.overhead_bytes,
            "total_bytes": fit.total_bytes,
            "ram_bytes": fit.hardware.ram_bytes,
            "vram_bytes": fit.hardware.vram_bytes,
            "fits_ram": fit.fits_ram,
            "fits_vram": fit.fits_vram,
            "suggested_target": fit.suggested_target.short(),
        });
        println!("{}", serde_json::to_string_pretty(&value)?);
        return Ok(());
    }
    println!("file:        {}", fit.path);
    println!("architecture:{} / {} layers", fit.architecture, fit.layers);
    println!("context:     {}", fit.context);
    println!("weights:     {}", human_bytes(fit.weights_bytes));
    println!("KV cache:    {}", human_bytes(fit.kv_cache_bytes));
    println!("overhead:    {}", human_bytes(fit.overhead_bytes));
    println!("total:       {}", human_bytes(fit.total_bytes));
    match fit.hardware.ram_bytes {
        Some(ram) => println!(
            "RAM:         {} ({})",
            human_bytes(ram),
            fit.fits_ram
                .map(|v| if v { "fits" } else { "does not fit" })
                .unwrap_or("unknown")
        ),
        None => println!("RAM:         unknown"),
    }
    match fit.hardware.vram_bytes {
        Some(vram) => println!(
            "VRAM:        {} ({})",
            human_bytes(vram),
            fit.fits_vram
                .map(|v| if v { "fits" } else { "does not fit" })
                .unwrap_or("unknown")
        ),
        None => println!("VRAM:        unknown"),
    }
    println!("suggested:   {}", fit.suggested_target.short());
    Ok(())
}

#[derive(Args)]
pub struct SplitArgs {
    pub input: PathBuf,
    pub output_dir: PathBuf,
    #[arg(long, default_value_t = 4096)]
    pub max_size_mib: u64,
    #[arg(long)]
    pub force: bool,
}

pub fn split_cmd(args: SplitArgs) -> Result<()> {
    let paths = split(
        &args.input,
        &args.output_dir,
        args.max_size_mib.saturating_mul(1024 * 1024),
        args.force,
    )?;
    for path in paths {
        println!("{}", path.display());
    }
    Ok(())
}

#[derive(Args)]
pub struct MergeArgs {
    #[arg(required = true, num_args = 1..)]
    pub inputs: Vec<PathBuf>,
    pub output: PathBuf,
    #[arg(long)]
    pub force: bool,
    #[arg(long)]
    pub delete_splits: bool,
}

pub fn merge_cmd(args: MergeArgs) -> Result<()> {
    merge(&args.inputs, &args.output, args.force, args.delete_splits)?;
    println!("merged into {}", args.output.display());
    Ok(())
}

#[derive(Args)]
pub struct MetaArgs {
    pub file: PathBuf,
    /// Read one metadata key.
    #[arg(long)]
    pub get: Option<String>,
    /// Set `key=JSON` (repeatable); requires --output.
    #[arg(long = "set", value_name = "KEY=JSON")]
    pub set: Vec<String>,
    /// Destination when setting metadata.
    #[arg(long)]
    pub output: Option<PathBuf>,
    #[arg(long)]
    pub force: bool,
    #[arg(long)]
    pub json: bool,
    /// Validate rope metadata and the chat-template structure without writing.
    #[arg(long)]
    pub validate: bool,
    /// Render tokenizer.chat_template with a JSON context object.
    #[arg(long, value_name = "JSON")]
    pub render_template: Option<PathBuf>,
}

pub fn meta(args: MetaArgs) -> Result<()> {
    if let Some(context_path) = args.render_template.as_ref() {
        if args.validate || !args.set.is_empty() || args.get.is_some() {
            anyhow::bail!("--render-template cannot be combined with --validate, --set or --get");
        }
        let raw = fs::read_to_string(context_path)
            .with_context(|| format!("cannot read '{}'", context_path.display()))?;
        let context: serde_json::Value = serde_json::from_str(&raw)
            .with_context(|| format!("invalid JSON in '{}'", context_path.display()))?;
        let rendered = render_chat_template(&args.file, &context).with_context(|| {
            format!("cannot render chat template from '{}'", args.file.display())
        })?;
        if args.json {
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({ "rendered": rendered }))?
            );
        } else {
            print!("{rendered}");
        }
        return Ok(());
    }
    if args.validate {
        if !args.set.is_empty() || args.get.is_some() {
            anyhow::bail!("--validate cannot be combined with --set or --get");
        }
        let report = validate_metadata(&args.file)?;
        if args.json {
            let issues: Vec<_> = report
                .issues
                .iter()
                .map(|issue| json!({ "key": issue.key, "message": issue.message }))
                .collect();
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "valid": report.is_ok(),
                    "issues": issues,
                }))?
            );
        } else if report.is_ok() {
            println!("metadata validation passed");
        } else {
            for issue in &report.issues {
                eprintln!("{}: {}", issue.key, issue.message);
            }
        }
        if !report.is_ok() {
            anyhow::bail!(
                "metadata validation failed with {} issue(s)",
                report.issues.len()
            );
        }
        return Ok(());
    }
    if args.set.is_empty() {
        if let Some(key) = args.get {
            match metadata_get(&args.file, &key)? {
                Some(value) => println!("{value}"),
                None => anyhow::bail!("metadata key '{key}' not found"),
            }
        } else {
            let values = metadata_list(&args.file)?;
            if args.json {
                println!("{}", serde_json::to_string_pretty(&values)?);
            } else {
                for (key, value) in values {
                    println!("{key} = {value}");
                }
            }
        }
        return Ok(());
    }
    let output = args
        .output
        .ok_or_else(|| anyhow::anyhow!("--output is required with --set"))?;
    let mut current = args.file.clone();
    for spec in &args.set {
        let (key, value) = spec
            .split_once('=')
            .ok_or_else(|| anyhow::anyhow!("invalid --set '{spec}', expected key=value"))?;
        let next = output.with_extension(format!(
            "{}.tmp",
            current
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("gguf")
        ));
        metadata_set(&current, &next, key, value, true)
            .with_context(|| format!("cannot set metadata '{key}'"))?;
        if current != args.file {
            let _ = fs::remove_file(&current);
        }
        current = next;
    }
    if current != output {
        if output.exists() {
            fs::remove_file(&output)
                .with_context(|| format!("cannot replace '{}'", output.display()))?;
        }
        fs::rename(&current, &output)
            .with_context(|| format!("cannot move result to '{}'", output.display()))?;
    }
    println!("wrote {}", output.display());
    Ok(())
}

#[derive(Args)]
pub struct PruneArgs {
    pub input: PathBuf,
    pub output: PathBuf,
    /// Regex matching tensors to remove.
    pub pattern: String,
    #[arg(long)]
    pub force: bool,
}

pub fn prune_cmd(args: PruneArgs) -> Result<()> {
    let removed = prune(&args.input, &args.output, &args.pattern, args.force)?;
    println!("removed {removed} tensors; wrote {}", args.output.display());
    Ok(())
}

#[derive(Args)]
pub struct DiffArgs {
    pub a: PathBuf,
    pub b: PathBuf,
    #[arg(long)]
    pub json: bool,
}

pub fn diff_cmd(args: DiffArgs) -> Result<()> {
    let report = diff(&args.a, &args.b)?;
    if args.json {
        let value = json!({
            "metadata_only_in_a": report.metadata_only_in_a,
            "metadata_only_in_b": report.metadata_only_in_b,
            "metadata_changed": report.metadata_changed,
            "tensors_only_in_a": report.tensors_only_in_a,
            "tensors_only_in_b": report.tensors_only_in_b,
            "tensor_changes": report.tensor_changes,
        });
        println!("{}", serde_json::to_string_pretty(&value)?);
    } else if report.is_empty() {
        println!("no differences");
    } else {
        for (label, values) in [
            ("metadata only in A", &report.metadata_only_in_a),
            ("metadata only in B", &report.metadata_only_in_b),
            ("metadata changed", &report.metadata_changed),
            ("tensors only in A", &report.tensors_only_in_a),
            ("tensors only in B", &report.tensors_only_in_b),
            ("tensor layout changed", &report.tensor_changes),
        ] {
            if !values.is_empty() {
                println!("{label}:");
                for value in values {
                    println!("  {value}");
                }
            }
        }
    }
    Ok(())
}

#[derive(Args)]
pub struct HashArgs {
    pub file: PathBuf,
    /// Expected SHA-256, for example an LFS oid.
    #[arg(long)]
    pub expected: Option<String>,
    #[arg(long)]
    pub json: bool,
}

pub fn hash(args: HashArgs) -> Result<()> {
    let actual = sha256_file(&args.file)?;
    let matches = args
        .expected
        .as_deref()
        .map(|expected| verify_hash(&args.file, expected))
        .transpose()?;
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "file": args.file,
                "sha256": actual,
                "expected": args.expected,
                "matches": matches,
            }))?
        );
    } else {
        println!("{actual}");
        if let Some(matches) = matches {
            println!(
                "{}",
                if matches {
                    "hash matches"
                } else {
                    "hash mismatch"
                }
            );
        }
    }
    if matches == Some(false) {
        anyhow::bail!("SHA-256 mismatch");
    }
    Ok(())
}

#[derive(Args)]
pub struct SafetensorsArgs {
    pub input_dir: PathBuf,
    pub output: PathBuf,
    #[arg(long)]
    pub script: Option<PathBuf>,
    #[arg(long, default_value = "f16")]
    pub outtype: String,
    /// Convert natively for supported LLaMA/Qwen BPE checkpoints.
    #[arg(long)]
    pub native: bool,
    #[arg(long)]
    pub force: bool,
}

pub fn from_safetensors(args: SafetensorsArgs) -> Result<()> {
    if args.native {
        let report =
            rustblock_core::convert_llama_safetensors(&args.input_dir, &args.output, args.force)
                .context("native safetensors conversion failed")?;
        println!(
            "wrote {} ({} layers, {} tensors, {} tokens, file_type {})",
            report.output, report.layers, report.tensors, report.tokenizer_tokens, report.file_type
        );
        return Ok(());
    }
    rustblock_core::safetensors::convert_with_hf_script(
        &args.input_dir,
        &args.output,
        args.script.as_deref(),
        &args.outtype,
    )
    .context("Hugging Face conversion failed")?;
    println!("wrote {}", args.output.display());
    Ok(())
}

#[derive(Args)]
pub struct DequantSafetensorsArgs {
    /// Single .safetensors file to dequantize.
    pub input: PathBuf,
    pub output: PathBuf,
    #[arg(long)]
    pub force: bool,
    #[arg(long)]
    pub json: bool,
}

pub fn dequant_safetensors(args: DequantSafetensorsArgs) -> Result<()> {
    let report = rustblock_core::dequantize_safetensors_file(&args.input, &args.output, args.force)
        .context("native safetensors dequantization failed")?;
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "input": report.input,
                "output": report.output,
                "tensors_in": report.tensors_in,
                "tensors_out": report.tensors_out,
                "fp8_tensors": report.fp8_tensors,
                "awq_tensors": report.awq_tensors,
            }))?
        );
    } else {
        println!(
            "wrote {} ({} tensors: {} AWQ, {} FP8; BF16)",
            report.output, report.tensors_out, report.awq_tensors, report.fp8_tensors
        );
    }
    Ok(())
}

#[derive(Args)]
pub struct DownloadArgs {
    /// Direct HTTP(S) URL of a GGUF file.
    pub url: String,
    /// Destination file path.
    pub output: PathBuf,
    /// Parallel range connections (1-16).
    #[arg(long, default_value_t = 16)]
    pub connections: usize,
    /// Optional SHA-256 (for example a Hugging Face LFS oid).
    #[arg(long)]
    pub sha256: Option<String>,
    /// Optional vision projector URL to download next to the model.
    #[arg(long)]
    pub mmproj: Option<String>,
    #[arg(long)]
    pub force: bool,
    #[arg(long)]
    pub json: bool,
    /// Emit one JSON object per line for each lifecycle/progress event.
    #[arg(long)]
    pub events: bool,
}

pub fn download_cmd(args: DownloadArgs) -> Result<()> {
    if args.events && args.json {
        anyhow::bail!("use either --events or --json, not both");
    }
    let options = DownloadOptions {
        connections: args.connections,
        force: args.force,
        expected_sha256: args.sha256,
        ..DownloadOptions::default()
    };
    let cancel = std::sync::atomic::AtomicBool::new(false);
    if args.events {
        println!(
            "{}",
            json!({"event": "start", "url": &args.url, "destination": &args.output})
        );
        let last = AtomicU64::new(u64::MAX);
        let result = download_url_with_progress(
            &args.url,
            &args.output,
            &options,
            &cancel,
            &|done, total| {
                let percent = if total > 0 {
                    done.saturating_mul(100) / total
                } else {
                    0
                };
                if last.swap(percent, Ordering::Relaxed) != percent {
                    println!(
                        "{}",
                        json!({"event": "progress", "done": done, "total": total, "percent": percent})
                    );
                }
            },
        );
        match result {
            Ok(report) => {
                println!(
                    "{}",
                    json!({"event": "done", "path": &report.path, "bytes": report.bytes, "sha256": &report.sha256, "resumed": report.resumed})
                );
            }
            Err(error) => {
                println!(
                    "{}",
                    json!({"event": "error", "message": error.to_string()})
                );
                return Err(error.into());
            }
        }
        if let Some(url) = args.mmproj.as_deref() {
            println!(
                "{}",
                json!({"event": "start", "kind": "mmproj", "url": url})
            );
            match rustblock_core::download_mmproj(url, &args.output, &options, &cancel) {
                Ok(projector) => println!(
                    "{}",
                    json!({"event": "done", "kind": "mmproj", "path": &projector.path, "bytes": projector.bytes, "sha256": &projector.sha256})
                ),
                Err(error) => {
                    println!(
                        "{}",
                        json!({"event": "error", "kind": "mmproj", "message": error.to_string()})
                    );
                    return Err(error.into());
                }
            }
        }
        return Ok(());
    }
    let report =
        download_url(&args.url, &args.output, &options, &cancel).context("download failed")?;
    let projector = if let Some(url) = args.mmproj {
        Some(
            rustblock_core::download_mmproj(&url, &args.output, &options, &cancel)
                .context("projector download failed")?,
        )
    } else {
        None
    };
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "path": report.path,
                "bytes": report.bytes,
                "sha256": report.sha256,
                "resumed": report.resumed,
                "mmproj": projector.map(|p| p.path),
            }))?
        );
    } else {
        println!(
            "downloaded {} ({})",
            report.path.display(),
            human_bytes(report.bytes)
        );
        println!("sha256: {}", report.sha256);
        if let Some(projector) = projector {
            println!("projector: {}", projector.path.display());
        }
    }
    Ok(())
}

#[derive(Args)]
pub struct DownloadManyArgs {
    /// Text file with one `URL<TAB>DESTINATION` job per line.
    pub jobs: PathBuf,
    #[arg(long, default_value_t = 16)]
    pub connections: usize,
    #[arg(long, default_value_t = 8)]
    pub max_files: usize,
    #[arg(long)]
    pub json: bool,
    /// Emit one JSON object per line for start, progress and completion.
    #[arg(long)]
    pub events: bool,
}

pub fn download_many_cmd(args: DownloadManyArgs) -> Result<()> {
    if args.events && args.json {
        anyhow::bail!("use either --events or --json, not both");
    }
    let text = fs::read_to_string(&args.jobs)
        .with_context(|| format!("cannot read '{}'", args.jobs.display()))?;
    let mut jobs = Vec::new();
    for (line_no, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (url, path) = line
            .split_once('\t')
            .or_else(|| line.split_once(' '))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "{}:{}: expected URL<TAB>PATH",
                    args.jobs.display(),
                    line_no + 1
                )
            })?;
        jobs.push((url.trim().to_string(), PathBuf::from(path.trim())));
    }
    if jobs.is_empty() {
        anyhow::bail!("no download jobs");
    }
    let options = DownloadOptions {
        connections: args.connections,
        ..DownloadOptions::default()
    };
    let cancel = std::sync::atomic::AtomicBool::new(false);
    if args.events {
        let last = std::sync::Mutex::new(std::collections::HashMap::<usize, u64>::new());
        let results = download_many_with_events(
            &jobs,
            args.max_files,
            &options,
            &cancel,
            &|event| match event {
                BatchDownloadEvent::Started {
                    index,
                    url,
                    destination,
                } => println!(
                    "{}",
                    json!({"event": "start", "index": index, "url": url, "destination": destination})
                ),
                BatchDownloadEvent::Progress { index, done, total } => {
                    let percent = if total > 0 {
                        done.saturating_mul(100) / total
                    } else {
                        0
                    };
                    let mut seen = last.lock().expect("progress lock");
                    if seen.get(&index).copied() == Some(percent) {
                        return;
                    }
                    seen.insert(index, percent);
                    drop(seen);
                    println!(
                        "{}",
                        json!({"event": "progress", "index": index, "done": done, "total": total, "percent": percent})
                    );
                }
                BatchDownloadEvent::Finished { index, result } => match result {
                    Ok(report) => println!(
                        "{}",
                        json!({"event": "done", "index": index, "path": &report.path, "bytes": report.bytes, "sha256": &report.sha256, "resumed": report.resumed})
                    ),
                    Err(error) => println!(
                        "{}",
                        json!({"event": "error", "index": index, "message": error.to_string()})
                    ),
                },
            },
        );
        let failures = results.iter().filter(|result| result.is_err()).count();
        println!(
            "{}",
            json!({"event": "summary", "files": results.len(), "failed": failures})
        );
        if failures > 0 {
            anyhow::bail!("{failures} download(s) failed");
        }
        return Ok(());
    }
    let results = rustblock_core::download_many(&jobs, args.max_files, &options, &cancel);
    let mut failures = 0usize;
    for result in &results {
        match result {
            Ok(report) => {
                if args.json {
                    println!(
                        "{}",
                        serde_json::to_string(
                            &json!({"ok": true, "path": report.path, "bytes": report.bytes, "sha256": report.sha256, "resumed": report.resumed})
                        )?
                    );
                } else {
                    println!(
                        "ok {} ({})",
                        report.path.display(),
                        human_bytes(report.bytes)
                    );
                }
            }
            Err(error) => {
                failures += 1;
                if args.json {
                    println!(
                        "{}",
                        serde_json::to_string(&json!({"ok": false, "error": error.to_string()}))?
                    );
                } else {
                    println!("error: {error}");
                }
            }
        }
    }
    if failures > 0 {
        anyhow::bail!("{failures} download(s) failed");
    }
    Ok(())
}

#[derive(Args)]
pub struct CacheArgs {
    /// Download/cache directory to inspect.
    pub directory: PathBuf,
    /// Keep at most this many GiB of model artifacts.
    #[arg(long, default_value_t = 50)]
    pub quota_gib: u64,
    /// Only inspect; do not delete anything.
    #[arg(long)]
    pub dry_run: bool,
    #[arg(long)]
    pub json: bool,
}

pub fn cache_cmd(args: CacheArgs) -> Result<()> {
    let quota_bytes = args.quota_gib.saturating_mul(1024 * 1024 * 1024);
    let before = cache_stats(&args.directory, quota_bytes)
        .map_err(|error| anyhow::anyhow!(error))
        .context("cannot inspect cache")?;
    let report = if args.dry_run {
        None
    } else {
        Some(
            trim_cache(&args.directory, quota_bytes)
                .map_err(|error| anyhow::anyhow!(error))
                .context("cannot trim cache")?,
        )
    };
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "directory": args.directory,
                "quota_bytes": quota_bytes,
                "files_before": before.entries.len(),
                "bytes_before": before.total_bytes,
                "over_quota_before": before.over_quota(),
                "dry_run": args.dry_run,
                "removed": report.as_ref().map(|r| r.removed),
                "freed_bytes": report.as_ref().map(|r| r.freed_bytes),
                "remaining_bytes": report.as_ref().map(|r| r.remaining_bytes),
            }))?
        );
    } else {
        println!(
            "{} file(s), {}{}",
            before.entries.len(),
            human_bytes(before.total_bytes),
            if before.over_quota() {
                " (over quota)"
            } else {
                ""
            }
        );
        if let Some(report) = report {
            println!(
                "removed {} file(s), freed {}; {} remain",
                report.removed,
                human_bytes(report.freed_bytes),
                human_bytes(report.remaining_bytes)
            );
        }
    }
    Ok(())
}

#[derive(Args)]
pub struct InstallArgs {
    pub model: PathBuf,
    /// Local Ollama model name (or label for the LM Studio hand-off).
    #[arg(long, default_value = "rustblock")]
    pub name: String,
    /// Folder where the Modelfile/instructions are written.
    #[arg(long, default_value = "./rustblock-runtime")]
    pub directory: PathBuf,
    /// Write LM Studio import instructions instead of Ollama.
    #[arg(long)]
    pub lmstudio: bool,
    /// Run `ollama create`; without this flag only the Modelfile is written.
    #[arg(long)]
    pub run: bool,
    #[arg(long)]
    pub json: bool,
}

pub fn install_cmd(args: InstallArgs) -> Result<()> {
    let report = if args.lmstudio {
        write_lmstudio_instructions(&args.model, &args.directory)?
    } else {
        install_ollama(&args.model, &args.name, &args.directory, args.run)?
    };
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "runtime": report.runtime,
                "model": report.model,
                "instructions": report.instructions,
                "command": report.command,
            }))?
        );
    } else {
        println!(
            "{} hand-off written: {}",
            report.runtime,
            report.instructions.display()
        );
        if let Some(command) = report.command {
            println!("next: {command}");
        }
    }
    Ok(())
}

#[derive(Args)]
pub struct QuickTestArgs {
    pub model: PathBuf,
    #[arg(long, default_value = "Hello")]
    pub prompt: String,
    #[arg(long, default_value_t = 32)]
    pub tokens: u32,
    #[arg(long)]
    pub json: bool,
}

pub fn quicktest_cmd(args: QuickTestArgs) -> Result<()> {
    let report = quick_test(&args.model, &args.prompt, args.tokens)?;
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "command": report.command,
                "output": report.output,
            }))?
        );
    } else {
        println!("{}", report.command);
        print!("{}", report.output);
    }
    Ok(())
}

#[derive(Args)]
pub struct BudgetArgs {
    pub file: PathBuf,
    /// Available file/memory budget in GiB.
    #[arg(long)]
    pub budget_gb: Option<f64>,
    #[arg(long, default_value_t = 4096)]
    pub context: u64,
    #[arg(long, default_value_t = 512)]
    pub overhead_mib: u64,
    /// Also convert to the selected target at this path.
    #[arg(long)]
    pub output: Option<PathBuf>,
    #[arg(long)]
    pub force: bool,
    #[arg(long)]
    pub json: bool,
}

pub fn budget(args: BudgetArgs) -> Result<()> {
    let plan = plan_budget(
        &args.file,
        BudgetRequest {
            budget_bytes: args
                .budget_gb
                .map(|gib| (gib.max(0.0) * 1024.0 * 1024.0 * 1024.0) as u64),
            context: args.context.max(1),
            overhead_bytes: args.overhead_mib.saturating_mul(1024 * 1024),
            hardware: Some(detect_hardware()),
        },
    )
    .context("cannot plan a quantization budget")?;
    if args.json {
        let candidates: Vec<_> = plan
            .candidates
            .iter()
            .map(|candidate| {
                json!({
                    "target": candidate.target.short(),
                    "weights_bytes": candidate.weights_bytes,
                    "kv_cache_bytes": candidate.kv_cache_bytes,
                    "total_bytes": candidate.total_bytes,
                    "fits_budget": candidate.fits_budget,
                    "fits_ram": candidate.fits_ram,
                    "fits_vram": candidate.fits_vram,
                })
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "file": plan.path,
                "context": plan.context,
                "budget_bytes": plan.budget_bytes,
                "selected": plan.selected.short(),
                "selected_total_bytes": plan.selected_total_bytes,
                "reason": plan.reason,
                "candidates": candidates,
            }))?
        );
    } else {
        println!("selected: {} ({})", plan.selected.short(), plan.reason);
        println!("estimated: {}", human_bytes(plan.selected_total_bytes));
        for candidate in &plan.candidates {
            println!(
                "  {:<7} {:>10}  budget={:?} ram={:?} vram={:?}",
                candidate.target.short(),
                human_bytes(candidate.total_bytes),
                candidate.fits_budget,
                candidate.fits_ram,
                candidate.fits_vram
            );
        }
    }
    if let Some(output) = args.output {
        let report = transcode(
            &args.file,
            &output,
            plan.selected,
            ConvertOptions {
                force: args.force,
                dry_run: false,
                verify: VerifyMode::Structure,
                allow_requantize: false,
            },
            None,
        )?;
        println!(
            "converted {} -> {} ({})",
            report.input.display(),
            report.output.display(),
            human_bytes(report.new_file_size)
        );
    }
    Ok(())
}

#[derive(Args)]
pub struct KldArgs {
    pub model: PathBuf,
    pub calibration: PathBuf,
    #[arg(long, default_value_t = 100)]
    pub chunks: u64,
    /// Reference logits file to compare against (adds --kl-divergence).
    #[arg(long)]
    pub base: Option<PathBuf>,
    /// Record reference logits into this file instead of comparing.
    #[arg(long)]
    pub record: Option<PathBuf>,
    #[arg(long)]
    pub json: bool,
}

pub fn kld(args: KldArgs) -> Result<()> {
    if args.base.is_some() && args.record.is_some() {
        anyhow::bail!("use either --base or --record, not both");
    }
    let report = run_kld(
        &args.model,
        &args.calibration,
        args.chunks,
        args.base.as_deref(),
        args.record.as_deref(),
    )?;
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "command": report.command,
                "stdout": report.stdout,
                "stderr": report.stderr,
            }))?
        );
    } else {
        println!("{}", report.command);
        print!("{}", report.stdout);
        eprint!("{}", report.stderr);
    }
    Ok(())
}
