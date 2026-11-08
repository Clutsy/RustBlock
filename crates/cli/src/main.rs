use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use rustblock_core::{
    convert, effective_tensor_types, exact_name_pattern, source_supported, transcode_with_imatrix,
    transcode_with_overrides, type_name, ConvertOptions, Gguf, ImportanceMatrix, Phase, Progress,
    Target, TensorOverride, VerifyMode,
};
use serde_json::json;

mod advanced;

#[derive(Parser)]
#[command(
    name = "rustblock",
    version,
    about = "Native GGUF conversion toolkit: reblock, float conversion, quantization"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Show GGUF metadata, tensor types and conversion eligibility.
    Inspect {
        /// GGUF file to inspect.
        file: PathBuf,
        /// Print machine-readable JSON.
        #[arg(long)]
        json: bool,
    },
    /// Verify a GGUF without writing anything.
    ///
    /// For PQ2_0 files this performs the full lossless split check; for other
    /// files it validates the structure only.
    Verify {
        /// GGUF file to verify.
        file: PathBuf,
        /// Skip the full PQ2_0 bit-exact check (structure only).
        #[arg(long)]
        quick: bool,
        /// Print machine-readable JSON on stdout.
        #[arg(long)]
        json: bool,
    },
    /// Convert a GGUF model to another GGUF layout.
    Convert {
        /// Input GGUF.
        input: PathBuf,
        /// Output GGUF.
        output: PathBuf,
        /// Target layout (default: lossless PQ2_0 -> Q2_0 reblock).
        #[arg(long, value_enum, default_value_t = TargetArg::Reblock)]
        to: TargetArg,
        /// Overwrite an existing output file.
        #[arg(long)]
        force: bool,
        /// Validate and estimate without writing anything.
        #[arg(long)]
        dry_run: bool,
        /// Post-write verification level.
        #[arg(long, value_enum, default_value_t = VerifyArg::Structure)]
        verify: VerifyArg,
        /// Override the type of tensors matching REGEX, e.g. `attn_v=q5_k`.
        /// May be repeated; the first matching rule wins.
        #[arg(long = "tensor-type", value_name = "REGEX=TYPE")]
        tensor_types: Vec<String>,
        /// Read `REGEX=TYPE` rules from a file (one or more per line, # comments).
        #[arg(long = "tensor-type-file", value_name = "FILE")]
        tensor_type_file: Option<PathBuf>,
        /// Write the effective per-tensor type map before converting.
        #[arg(long = "dump-tensor-types", value_name = "FILE")]
        dump_tensor_types: Option<PathBuf>,
        /// Allow requantizing an already-quantized source (lossy).
        #[arg(long)]
        allow_requantize: bool,
        /// Apply importance values from a legacy .imatrix or GGUF imatrix.
        #[arg(long)]
        imatrix: Option<PathBuf>,
        /// Top fraction of tensors promoted to a higher-precision type.
        #[arg(long, default_value_t = 0.2)]
        imatrix_percentile: f32,
        /// Print machine-readable JSON on stdout.
        #[arg(long)]
        json: bool,
    },
    /// Generate an importance matrix with an installed llama-imatrix.
    Imatrix(advanced::ImatrixArgs),
    /// Compare source and output tensors with RMSE, SNR and cosine metrics.
    Quality(advanced::QualityArgs),
    /// Choose the best target for a size/context budget, optionally converting.
    Budget(advanced::BudgetArgs),
    /// Run llama-perplexity for KL-divergence recording or comparison.
    Kld(advanced::KldArgs),
    /// Estimate weights and KV-cache memory for a context length.
    Fit(advanced::FitArgs),
    /// Split a GGUF into numbered shards.
    Split(advanced::SplitArgs),
    /// Merge numbered GGUF shards.
    Merge(advanced::MergeArgs),
    /// Read or edit GGUF metadata.
    Meta(advanced::MetaArgs),
    /// Remove tensors matching a regex.
    Prune(advanced::PruneArgs),
    /// Compare two GGUF files without loading tensor payloads.
    Diff(advanced::DiffArgs),
    /// SHA-256 and LFS oid verification.
    Hash(advanced::HashArgs),
    /// Bridge to Hugging Face's convert_hf_to_gguf.py.
    FromSafetensors(advanced::SafetensorsArgs),
    /// Dequantize native FP8/AWQ safetensors to a BF16 safetensors file.
    DequantSafetensors(advanced::DequantSafetensorsArgs),
    /// Download a GGUF with resumable parallel ranges and optional SHA-256.
    Download(advanced::DownloadArgs),
    /// Download up to 8 files at once from a JSON-lines/TAB task file.
    DownloadMany(advanced::DownloadManyArgs),
    /// Inspect the model cache and enforce a quota by removing oldest files.
    Cache(advanced::CacheArgs),
    /// Write an Ollama Modelfile or LM Studio import instructions.
    Install(advanced::InstallArgs),
    /// Smoke-test a GGUF with an installed llama-cli.
    QuickTest(advanced::QuickTestArgs),
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum VerifyArg {
    None,
    Structure,
    Full,
}

impl From<VerifyArg> for VerifyMode {
    fn from(v: VerifyArg) -> Self {
        match v {
            VerifyArg::None => VerifyMode::None,
            VerifyArg::Structure => VerifyMode::Structure,
            VerifyArg::Full => VerifyMode::Full,
        }
    }
}

#[derive(Copy, Clone, Debug, ValueEnum)]
pub(crate) enum TargetArg {
    Reblock,
    F32,
    F16,
    Bf16,
    #[value(name = "q8_0")]
    Q8_0,
    #[value(name = "q5_0")]
    Q5_0,
    #[value(name = "q5_1")]
    Q5_1,
    #[value(name = "q4_0")]
    Q4_0,
    #[value(name = "q4_1")]
    Q4_1,
    #[value(name = "q2_k")]
    Q2K,
    #[value(name = "q3_k")]
    Q3KM,
    #[value(name = "q3_k_m")]
    Q3KMPreset,
    #[value(name = "q4_k")]
    Q4KM,
    #[value(name = "q4_k_m")]
    Q4KMPreset,
    #[value(name = "q5_k")]
    Q5KM,
    #[value(name = "q5_k_m")]
    Q5KMPreset,
    #[value(name = "q3_k_s")]
    Q3KS,
    #[value(name = "q4_k_s")]
    Q4KS,
    #[value(name = "q5_k_s")]
    Q5KS,
    #[value(name = "q6_k")]
    Q6K,
    #[value(name = "tq1_0")]
    Tq1_0,
    #[value(name = "tq2_0")]
    Tq2_0,
    #[value(name = "mxfp4")]
    Mxfp4,
    #[value(name = "nvfp4")]
    Nvfp4,
    #[value(name = "q1_0")]
    Q1_0,
}

impl From<TargetArg> for Target {
    fn from(v: TargetArg) -> Self {
        match v {
            TargetArg::Reblock => Target::Reblock,
            TargetArg::F32 => Target::F32,
            TargetArg::F16 => Target::F16,
            TargetArg::Bf16 => Target::Bf16,
            TargetArg::Q8_0 => Target::Q8_0,
            TargetArg::Q5_0 => Target::Q5_0,
            TargetArg::Q5_1 => Target::Q5_1,
            TargetArg::Q4_0 => Target::Q4_0,
            TargetArg::Q4_1 => Target::Q4_1,
            TargetArg::Q2K => Target::Q2K,
            TargetArg::Q3KM => Target::Q3KM,
            TargetArg::Q3KMPreset => Target::Q3KM,
            TargetArg::Q4KM => Target::Q4KM,
            TargetArg::Q4KMPreset => Target::Q4KM,
            TargetArg::Q5KM => Target::Q5KM,
            TargetArg::Q5KMPreset => Target::Q5KM,
            TargetArg::Q3KS => Target::Q3K,
            TargetArg::Q4KS => Target::Q4K,
            TargetArg::Q5KS => Target::Q5K,
            TargetArg::Q6K => Target::Q6K,
            TargetArg::Tq1_0 => Target::Tq1_0,
            TargetArg::Tq2_0 => Target::Tq2_0,
            TargetArg::Mxfp4 => Target::Mxfp4,
            TargetArg::Nvfp4 => Target::Nvfp4,
            TargetArg::Q1_0 => Target::Q1_0,
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Inspect { file, json } => inspect(&file, json),
        Command::Verify { file, quick, json } => verify_cmd(&file, quick, json),
        Command::Convert {
            input,
            output,
            to,
            force,
            dry_run,
            verify,
            tensor_types,
            tensor_type_file,
            dump_tensor_types,
            allow_requantize,
            imatrix,
            imatrix_percentile,
            json,
        } => {
            let mut overrides = Vec::new();
            for spec in &tensor_types {
                overrides.push(
                    TensorOverride::parse(spec)
                        .with_context(|| format!("invalid --tensor-type '{spec}'"))?,
                );
            }
            if let Some(path) = &tensor_type_file {
                overrides.extend(
                    TensorOverride::load_file(path)
                        .with_context(|| format!("cannot read '{}'", path.display()))?,
                );
            }
            let matrix = imatrix
                .as_deref()
                .map(ImportanceMatrix::load)
                .transpose()
                .context("cannot load importance matrix")?;
            convert_cmd(ConvertCli {
                input: &input,
                output: &output,
                target: to.into(),
                force,
                dry_run,
                verify: verify.into(),
                allow_requantize,
                overrides: &overrides,
                dump_tensor_types: dump_tensor_types.as_deref(),
                imatrix: matrix.as_ref(),
                imatrix_percentile,
                as_json: json,
            })
        }
        Command::Imatrix(args) => advanced::imatrix(args),
        Command::Quality(args) => advanced::quality(args),
        Command::Budget(args) => advanced::budget(args),
        Command::Kld(args) => advanced::kld(args),
        Command::Fit(args) => advanced::fit(args),
        Command::Split(args) => advanced::split_cmd(args),
        Command::Merge(args) => advanced::merge_cmd(args),
        Command::Meta(args) => advanced::meta(args),
        Command::Prune(args) => advanced::prune_cmd(args),
        Command::Diff(args) => advanced::diff_cmd(args),
        Command::Hash(args) => advanced::hash(args),
        Command::FromSafetensors(args) => advanced::from_safetensors(args),
        Command::DequantSafetensors(args) => advanced::dequant_safetensors(args),
        Command::Download(args) => advanced::download_cmd(args),
        Command::DownloadMany(args) => advanced::download_many_cmd(args),
        Command::Cache(args) => advanced::cache_cmd(args),
        Command::Install(args) => advanced::install_cmd(args),
        Command::QuickTest(args) => advanced::quicktest_cmd(args),
    }
}

fn inspect(file: &Path, as_json: bool) -> Result<()> {
    let g = Gguf::open(file).with_context(|| format!("cannot read '{}'", file.display()))?;

    let pq2 = g.pq2_count();
    let q2 = g
        .tensors
        .iter()
        .filter(|t| t.ty == rustblock_core::TY_Q2_0)
        .count();
    let ptq1 = g
        .tensors
        .iter()
        .filter(|t| t.ty == rustblock_core::TY_PTQ1_0)
        .count();
    let reblockable = pq2 > 0 && ptq1 == 0;
    let native_convert = ptq1 == 0 && g.tensors.iter().all(|t| source_supported(t.ty));
    let histogram: Vec<_> = g
        .type_histogram()
        .into_iter()
        .map(|(ty, count)| (ty, type_name(ty).to_string(), count))
        .collect();

    if as_json {
        let value = json!({
            "file": file,
            "version": g.version,
            "size": g.file_len,
            "alignment": g.alignment,
            "tensors": g.tensors.len(),
            "pq2_0": pq2,
            "ptq1_0": ptq1,
            "already_q2_0": q2 == g.tensors.len(),
            "file_type": g.file_type,
            "hadamard_metadata": g.has_hadamard_version,
            "mtp_metadata": g.has_mtp_metadata,
            "prism_metadata_entries": g.prism_metadata_signature().len(),
            "convertible": reblockable,
            "reblock": reblockable,
            "native_convert": native_convert,
            "types": histogram.iter().map(|(ty, name, count)| {
                json!({"id": ty, "name": name, "tensors": count})
            }).collect::<Vec<_>>(),
        });
        println!("{}", serde_json::to_string_pretty(&value)?);
        return Ok(());
    }

    println!("file:      {}", file.display());
    println!("size:      {}", human_bytes(g.file_len));
    println!("gguf:      v{}", g.version);
    println!("alignment: {} bytes", g.alignment);
    println!(
        "tensors:   {} (PQ2_0: {}, PTQ1_0: {})",
        g.tensors.len(),
        pq2,
        ptq1
    );
    println!(
        "file_type: {}",
        g.file_type
            .map(|v| v.to_string())
            .unwrap_or_else(|| "absent".to_string())
    );
    println!(
        "hadamard:  {}",
        if g.has_hadamard_version {
            "prism.hadamard.* metadata present"
        } else {
            "not found"
        }
    );
    println!(
        "mtp:       {}",
        if g.has_mtp_metadata {
            "prism.mtp.* metadata present (preserved, not interpreted)"
        } else {
            "not found"
        }
    );
    println!(
        "reblock:   {}",
        match (reblockable, ptq1 > 0) {
            (true, _) => "yes (lossless PQ2_0 -> Q2_0)",
            (false, true) => "no (PTQ1_0 has no lossless upstream type)",
            (false, false) => "no (no PQ2_0 tensors found)",
        }
    );
    println!(
        "native:    {}",
        match (native_convert, ptq1 > 0) {
            (true, _) => {
                "yes (19 targets: float, Q4_0..Q6_K, TQ1_0/TQ2_0, MXFP4/NVFP4, Q1_0)"
            }
            (false, true) => "no (PTQ1_0 is not supported)",
            (false, false) => "no (file contains an unsupported tensor type)",
        }
    );
    println!("types:");
    for (ty, name, count) in histogram {
        println!("  {ty:>4}  {name:<12} {count} tensors");
    }
    Ok(())
}

fn verify_cmd(file: &Path, quick: bool, as_json: bool) -> Result<()> {
    let started = std::time::Instant::now();
    let g = Gguf::open(file).with_context(|| format!("cannot read '{}'", file.display()))?;
    let pq2 = g.pq2_count();
    let ptq1 = g
        .tensors
        .iter()
        .filter(|t| t.ty == rustblock_core::TY_PTQ1_0)
        .count();
    let convertible = pq2 > 0 && ptq1 == 0;

    let mut report = None;
    if convertible && !quick {
        let pb = ProgressBar::new(0);
        pb.set_style(
            ProgressStyle::with_template("{spinner:.green} {msg} {bytes}/{total_bytes} ({eta})")
                .unwrap()
                .progress_chars("=>-"),
        );
        let mut progress = |p: &Progress| {
            pb.set_length(p.bytes_total.max(1));
            pb.set_position(p.bytes_done);
            pb.set_message(format!(
                "verifying [{}/{}] {}",
                p.tensor_index + 1,
                p.tensors_total,
                p.tensor_name
            ));
        };
        let dummy = file.with_extension("verify.gguf");
        let r = convert(
            file,
            &dummy,
            ConvertOptions {
                dry_run: true,
                verify: VerifyMode::Full,
                ..Default::default()
            },
            Some(&mut progress),
        )
        .with_context(|| format!("verification of '{}' failed", file.display()))?;
        pb.finish_and_clear();
        report = Some(r);
    }

    let elapsed = started.elapsed();
    if as_json {
        let value = json!({
            "file": file,
            "status": "ok",
            "version": g.version,
            "size": g.file_len,
            "alignment": g.alignment,
            "tensors": g.tensors.len(),
            "pq2_0": pq2,
            "ptq1_0": ptq1,
            "file_type": g.file_type,
            "hadamard_metadata": g.has_hadamard_version,
            "convertible": convertible,
            "full_check": report.is_some(),
            "new_file_size": report.as_ref().map(|r| r.new_file_size),
            "elapsed_seconds": elapsed.as_secs_f64(),
        });
        println!("{}", serde_json::to_string_pretty(&value)?);
        return Ok(());
    }

    println!("file:      {}", file.display());
    println!(
        "structure: OK ({} tensors, GGUF v{}, alignment {})",
        g.tensors.len(),
        g.version,
        g.alignment
    );
    if convertible {
        if let Some(report) = &report {
            println!(
                "lossless:  OK ({} PQ2_0 tensors, {} -> {} data)",
                report.tensors_pq2,
                human_bytes(report.old_data_size),
                human_bytes(report.new_data_size)
            );
        } else {
            println!("lossless:  skipped (--quick)");
        }
    } else if ptq1 > 0 {
        println!("lossless:  not supported (PTQ1_0 has no lossless upstream type)");
    } else {
        println!("lossless:  not needed (no PQ2_0 tensors found)");
    }
    println!("elapsed:   {}", human_duration(elapsed));
    Ok(())
}

struct ConvertCli<'a> {
    input: &'a Path,
    output: &'a Path,
    target: Target,
    force: bool,
    dry_run: bool,
    verify: VerifyMode,
    allow_requantize: bool,
    overrides: &'a [TensorOverride],
    dump_tensor_types: Option<&'a Path>,
    imatrix: Option<&'a ImportanceMatrix>,
    imatrix_percentile: f32,
    as_json: bool,
}

fn convert_cmd(args: ConvertCli<'_>) -> Result<()> {
    let ConvertCli {
        input,
        output,
        target,
        force,
        dry_run,
        verify,
        allow_requantize,
        overrides,
        dump_tensor_types,
        imatrix,
        imatrix_percentile,
        as_json,
    } = args;
    if target == Target::Reblock && (!overrides.is_empty() || dump_tensor_types.is_some()) {
        anyhow::bail!("tensor-type options cannot be used with the lossless reblock target");
    }
    let opts = ConvertOptions {
        force,
        dry_run,
        verify,
        allow_requantize,
    };

    if let Some(path) = dump_tensor_types {
        let g = Gguf::open(input).with_context(|| format!("cannot read '{}'", input.display()))?;
        let mut text = String::new();
        for (name, ty) in effective_tensor_types(&g.tensors, target, overrides) {
            text.push_str(&exact_name_pattern(&name));
            text.push('=');
            text.push_str(&type_name(ty).to_ascii_lowercase());
            text.push('\n');
        }
        fs::write(path, text).with_context(|| format!("cannot write '{}'", path.display()))?;
    }

    let pb = ProgressBar::new(0);
    pb.set_style(
        ProgressStyle::with_template("{spinner:.green} {msg} {bytes}/{total_bytes} ({eta})")
            .unwrap()
            .progress_chars("=>-"),
    );

    let mut progress = |p: &Progress| {
        pb.set_length(p.bytes_total.max(1));
        pb.set_position(p.bytes_done);
        pb.set_message(format!(
            "{} [{}/{}] {}",
            match p.phase {
                Phase::Converting => "converting",
                Phase::Verifying => "verifying",
            },
            p.tensor_index + 1,
            p.tensors_total,
            p.tensor_name
        ));
    };

    let report = if let Some(matrix) = imatrix {
        transcode_with_imatrix(
            input,
            output,
            target,
            opts,
            overrides,
            matrix,
            imatrix_percentile,
            Some(&mut progress),
        )
        .with_context(|| format!("conversion of '{}' failed", input.display()))?
    } else {
        transcode_with_overrides(input, output, target, opts, overrides, Some(&mut progress))
            .with_context(|| format!("conversion of '{}' failed", input.display()))?
    };
    pb.finish_and_clear();

    if as_json {
        let value = json!({
            "input": report.input,
            "output": report.output,
            "dry_run": report.dry_run,
            "target": target.short(),
            "tensors_total": report.tensors_total,
            "tensors_pq2_0": report.tensors_pq2,
            "old_file_size": report.old_file_size,
            "new_file_size": report.new_file_size,
            "old_data_size": report.old_data_size,
            "new_data_size": report.new_data_size,
            "verify": format!("{:?}", report.verified).to_lowercase(),
            "elapsed_seconds": report.elapsed.as_secs_f64(),
        });
        println!("{}", serde_json::to_string_pretty(&value)?);
        return Ok(());
    }

    if report.dry_run {
        println!("dry run: no file written");
    } else {
        println!(
            "converted: {} -> {}",
            report.input.display(),
            report.output.display()
        );
    }
    println!("target:    {}", target.short());
    if target == Target::Reblock {
        println!(
            "tensors:   {} ({} PQ2_0 -> Q2_0)",
            report.tensors_total, report.tensors_pq2
        );
    } else {
        println!("tensors:   {}", report.tensors_total);
    }
    println!(
        "data:      {} -> {}",
        human_bytes(report.old_data_size),
        human_bytes(report.new_data_size)
    );
    println!(
        "file:      {} -> {}",
        human_bytes(report.old_file_size),
        human_bytes(report.new_file_size)
    );
    println!("verify:    {:?}", report.verified);
    println!("elapsed:   {}", human_duration(report.elapsed));
    Ok(())
}

fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.2} {}", UNITS[unit])
    }
}

fn human_duration(d: Duration) -> String {
    if d.as_secs() >= 1 {
        format!("{:.1} s", d.as_secs_f64())
    } else {
        format!("{} ms", d.as_millis())
    }
}
