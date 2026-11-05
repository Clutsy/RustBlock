use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use rustblock_core::{
    convert, source_supported, transcode, type_name, ConvertOptions, Gguf, Phase, Progress, Target,
    VerifyMode,
};
use serde_json::json;

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
        /// Print machine-readable JSON on stdout.
        #[arg(long)]
        json: bool,
    },
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
enum TargetArg {
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
    Q3K,
    #[value(name = "q4_k")]
    Q4K,
    #[value(name = "q5_k")]
    Q5K,
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
            TargetArg::Q3K => Target::Q3K,
            TargetArg::Q4K => Target::Q4K,
            TargetArg::Q5K => Target::Q5K,
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
            json,
        } => convert_cmd(
            &input,
            &output,
            to.into(),
            force,
            dry_run,
            verify.into(),
            json,
        ),
    }
}

fn inspect(file: &Path, as_json: bool) -> Result<()> {
    let g = Gguf::open(file).with_context(|| format!("cannot read '{}'", file.display()))?;

    let pq2 = g.pq2_count();
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
            "file_type": g.file_type,
            "hadamard_metadata": g.has_hadamard_version,
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

fn convert_cmd(
    input: &Path,
    output: &Path,
    target: Target,
    force: bool,
    dry_run: bool,
    verify: VerifyMode,
    as_json: bool,
) -> Result<()> {
    let opts = ConvertOptions {
        force,
        dry_run,
        verify,
    };

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

    let report = transcode(input, output, target, opts, Some(&mut progress))
        .with_context(|| format!("conversion of '{}' failed", input.display()))?;
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
