//! Differential parity harness against a pinned `llama-quantize` build.
//!
//! The roadmap correctness gate requires proving that the RustBlock recipe
//! engine produces the same per-tensor type map as llama.cpp for the
//! deterministic targets. A synthetic F16 GGUF with representative tensor
//! names is quantized by the reference binary and the resulting type map is
//! compared with [`effective_tensor_types`].
//!
//! The test is `#[ignore]` on purpose: it needs an external build. Point
//! `RUSTBLOCK_LLAMA_QUANTIZE` at a pinned `llama-quantize` binary and run:
//!
//! ```text
//! RUSTBLOCK_LLAMA_QUANTIZE=/path/to/llama-quantize \
//!     cargo test -p rustblock-core --test quantize_parity -- --ignored --nocapture
//! ```
//!
//! Without the variable the test reports a skip instead of failing, so a plain
//! `cargo test --workspace` stays green on machines without llama.cpp.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use rustblock_core::gguf::Gguf;
use rustblock_core::{effective_tensor_types, Target};

const TY_F16: u32 = 1;
const TY_F32: u32 = 0;
const ALIGNMENT: u32 = 32;

/// Eight layers so the first/last eighth and the every-third-layer rules are
/// actually exercised. Transcode order matters: llama-quantize reads the tensor
/// table as written.
const LAYERS: usize = 8;

fn fixture_tensors() -> Vec<(String, Vec<u64>, u32)> {
    let mut tensors = vec![("token_embd.weight".to_string(), vec![256, 4], TY_F16)];
    for layer in 0..LAYERS {
        tensors.push((format!("blk.{layer}.attn_norm.weight"), vec![256], TY_F32));
        for name in [
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_output",
            "ffn_gate",
            "ffn_up",
            "ffn_down",
        ] {
            tensors.push((format!("blk.{layer}.{name}.weight"), vec![256, 4], TY_F16));
        }
    }
    tensors.push(("output.weight".to_string(), vec![256, 4], TY_F16));
    tensors
}

fn push_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn push_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn push_str(buf: &mut Vec<u8>, s: &str) {
    push_u64(buf, s.len() as u64);
    buf.extend_from_slice(s.as_bytes());
}

fn kv_str(buf: &mut Vec<u8>, key: &str, value: &str) {
    push_str(buf, key);
    push_u32(buf, 8);
    push_str(buf, value);
}

fn kv_u32(buf: &mut Vec<u8>, key: &str, value: u32) {
    push_str(buf, key);
    push_u32(buf, 4);
    push_u32(buf, value);
}

fn kv_f32(buf: &mut Vec<u8>, key: &str, value: f32) {
    push_str(buf, key);
    push_u32(buf, 6);
    buf.extend_from_slice(&value.to_le_bytes());
}

fn element_count(dims: &[u64]) -> u64 {
    dims.iter().fold(1u64, |acc, d| acc.saturating_mul(*d))
}

fn byte_len(ty: u32, elements: u64) -> u64 {
    match ty {
        TY_F16 => elements * 2,
        _ => elements * 4,
    }
}

fn align_up(value: u64, alignment: u64) -> u64 {
    value.div_ceil(alignment) * alignment
}

/// A minimal but valid GGUF v3 F16 model, enough for llama-quantize to apply
/// the llama.cpp recipe rules without a real checkpoint.
fn write_fixture(path: &Path) {
    let tensors = fixture_tensors();
    let mut header = Vec::new();
    header.extend_from_slice(b"GGUF");
    push_u32(&mut header, 3);
    push_u64(&mut header, tensors.len() as u64);
    push_u64(&mut header, 12);
    kv_str(&mut header, "general.architecture", "llama");
    kv_str(&mut header, "general.name", "rustblock-parity-fixture");
    kv_u32(&mut header, "general.alignment", ALIGNMENT);
    kv_u32(&mut header, "general.file_type", TY_F16);

    kv_u32(&mut header, "llama.block_count", LAYERS as u32);
    kv_u32(&mut header, "llama.context_length", 512);
    kv_u32(&mut header, "llama.embedding_length", 256);
    kv_u32(&mut header, "llama.feed_forward_length", 1024);
    kv_u32(&mut header, "llama.attention.head_count", 2);
    kv_u32(&mut header, "llama.attention.head_count_kv", 2);
    kv_f32(
        &mut header,
        "llama.attention.layer_norm_rms_epsilon",
        0.00001,
    );
    // n_rot must equal embedding_length / head_count for this tiny model.
    kv_u32(&mut header, "llama.rope.dimension_count", 128);

    let mut offset = 0u64;
    let mut data_len = 0u64;
    for (name, dims, ty) in &tensors {
        push_str(&mut header, name);
        push_u32(&mut header, dims.len() as u32);
        for dim in dims {
            push_u64(&mut header, *dim);
        }
        push_u32(&mut header, *ty);
        push_u64(&mut header, offset);
        let size = byte_len(*ty, element_count(dims));
        offset = align_up(offset + size, ALIGNMENT as u64);
        data_len = offset;
    }

    let pad = align_up(header.len() as u64, ALIGNMENT as u64) as usize;
    header.resize(pad, 0);
    let mut bytes = header;
    bytes.resize(pad + data_len as usize, 0);
    fs::write(path, bytes).expect("write parity fixture");
}

fn reference_binary() -> Option<PathBuf> {
    let value = std::env::var_os("RUSTBLOCK_LLAMA_QUANTIZE")?;
    let path = PathBuf::from(value);
    if path.as_os_str().is_empty() {
        return None;
    }
    Some(path)
}

#[test]
#[ignore = "needs a pinned llama-quantize build; set RUSTBLOCK_LLAMA_QUANTIZE"]
fn tensor_type_maps_match_llama_quantize() {
    let Some(binary) = reference_binary() else {
        eprintln!(
            "skipped: RUSTBLOCK_LLAMA_QUANTIZE is not set, so no reference llama-quantize build was used"
        );
        return;
    };

    let dir = tempfile::tempdir().expect("temp dir");
    let source = dir.path().join("fixture-f16.gguf");
    write_fixture(&source);
    let source_gguf = Gguf::open(&source).expect("fixture must be a readable GGUF");

    let targets = [
        Target::Q2K,
        Target::Q3KM,
        Target::Q4KM,
        Target::Q5KM,
        Target::Q6K,
        Target::Q3K,
        Target::Q4K,
        Target::Q5K,
        Target::Q8_0,
    ];

    let mut failures = Vec::new();
    for target in targets {
        let reference = dir.path().join(format!("ref-{}.gguf", target.short()));
        let output = Command::new(&binary)
            .arg(&source)
            .arg(&reference)
            .arg(target.short())
            .output()
            .unwrap_or_else(|error| panic!("cannot run {}: {error}", binary.display()));
        if !output.status.success() {
            failures.push(format!(
                "{}: llama-quantize failed: {}",
                target.short(),
                String::from_utf8_lossy(&output.stderr).trim()
            ));
            continue;
        }

        let expected = effective_tensor_types(&source_gguf.tensors, target, &[]);
        let produced = Gguf::open(&reference).expect("llama-quantize output must be readable");
        let mut produced_map: Vec<(String, u32)> = produced
            .tensors
            .iter()
            .map(|tensor| (tensor.name.clone(), tensor.ty))
            .collect();
        produced_map.sort();
        let mut expected_map = expected;
        expected_map.sort();

        if expected_map != produced_map {
            let mut diff = String::new();
            for ((ename, ety), (pname, pty)) in expected_map.iter().zip(produced_map.iter()) {
                if ename != pname || ety != pty {
                    diff.push_str(&format!("\n  {ename}: rustblock {ety} vs llama.cpp {pty}"));
                }
            }
            failures.push(format!("{}: type map mismatch{diff}", target.short()));
        }
    }

    assert!(
        failures.is_empty(),
        "llama-quantize parity failed:\n{}",
        failures.join("\n")
    );
}
