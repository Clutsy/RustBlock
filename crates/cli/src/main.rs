use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use serde_json::json;
use ternary_reblock_core::{convert, type_name, ConvertOptions, Gguf, Progress, VerifyMode};

#[derive(Parser)]
#[command(
    name = "ternary-reblock",
    version,
    about = "Lossless GGUF reblocker: Prism PQ2_0 -> upstream Q2_0"
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
    /// Convert a PQ2_0 GGUF into a standard Q2_0 GGUF.
    Convert {
        /// Input GGUF (PQ2_0).
        input: PathBuf,
        /// Output GGUF (Q2_0).
        output: PathBuf,
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

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Inspect { file, json } => inspect(&file, json),
        Command::Convert {
            input,
            output,
            force,
            dry_run,
            verify,
            json,
        } => convert_cmd(&input, &output, force, dry_run, verify.into(), json),
    }
}

fn inspect(file: &Path, as_json: bool) -> Result<()> {
    let g = Gguf::open(file).with_context(|| format!("cannot read '{}'", file.display()))?;

    let pq2 = g.pq2_count();
    let ptq1 = g
        .tensors
        .iter()
        .filter(|t| t.ty == ternary_reblock_core::TY_PTQ1_0)
        .count();
    let convertible = pq2 > 0 && ptq1 == 0;
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
            "convertible": convertible,
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
        "convert:   {}",
        match (convertible, ptq1 > 0) {
            (true, _) => "yes (lossless PQ2_0 -> Q2_0)",
            (false, true) => "no (PTQ1_0 has no lossless upstream type)",
            (false, false) => "no (no PQ2_0 tensors found)",
        }
    );
    println!("types:");
    for (ty, name, count) in histogram {
        println!("  {ty:>4}  {name:<12} {count} tensors");
    }
    Ok(())
}

fn convert_cmd(
    input: &Path,
    output: &Path,
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
            "[{}/{}] {}",
            p.tensor_index + 1,
            p.tensors_total,
            p.tensor_name
        ));
    };

    let report = convert(input, output, opts, Some(&mut progress))
        .with_context(|| format!("conversion of '{}' failed", input.display()))?;
    pb.finish_and_clear();

    if as_json {
        let value = json!({
            "input": report.input,
            "output": report.output,
            "dry_run": report.dry_run,
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
    println!(
        "tensors:   {} ({} PQ2_0 -> Q2_0)",
        report.tensors_total, report.tensors_pq2
    );
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
