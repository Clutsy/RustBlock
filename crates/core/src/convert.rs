//! Lossless Prism PQ2_0 -> upstream Q2_0 reblocking.
//!
//! The two formats use the same 2-bit ternary codec; only the block grouping
//! differs. A 128-element PQ2_0 block (`d: f16`, 32 code bytes) is rewritten
//! as two 64-element Q2_0 blocks that share the same `d` and contain the same
//! codes split in half. No float round-trip is performed, so the represented
//! values are bit-identical by construction.
//!
//! Tensor offsets are recalculated because a PQ2_0 block is 34 bytes while a
//! Q2_0 block is 18 bytes. The GGUF header is copied verbatim and only
//! fixed-width fields are patched (tensor type, tensor offset, file_type), so
//! all metadata - including the `prism.hadamard.*` keys - is preserved.

use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use half::f16;
use tempfile::NamedTempFile;
use thiserror::Error;

use crate::gguf::{align_up, patch_u32, patch_u64, Gguf, GgufError};
use crate::types::{TY_PQ2_0, TY_Q2_0};

/// `LLAMA_FTYPE_MOSTLY_Q2_0` from llama.h (not the same namespace as GGML types).
pub const LLAMA_FTYPE_MOSTLY_Q2_0: u32 = 41;

const IO_BUF_SIZE: usize = 8 * 1024 * 1024;
const REBLOCK_CHUNK_BLOCKS: usize = IO_BUF_SIZE / 34;

#[derive(Debug, Error)]
pub enum ConvertError {
    #[error(transparent)]
    Gguf(#[from] GgufError),
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error("no PQ2_0 tensors found in the input; nothing to convert")]
    NoPq2,
    #[error(
        "PTQ1_0 is not supported: it has no lossless upstream type \
         (converting it would require lossy requantization)"
    )]
    Ptq1,
    #[error("tensor '{name}' has unsupported type {ty}")]
    UnsupportedTensor { name: String, ty: u32 },
    #[error("input and output refer to the same file")]
    SameFile,
    #[error("output file already exists (use --force to overwrite): {0}")]
    OutputExists(PathBuf),
    #[error("size overflow while planning the output layout")]
    Overflow,
    #[error("verification failed: {0}")]
    Verify(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifyMode {
    /// Only parse the output to make sure it is structurally readable.
    None,
    /// Also compare the patched header and the tensor table.
    Structure,
    /// Also decode every converted element and compare it bit-for-bit.
    Full,
}

#[derive(Debug, Clone, Copy)]
pub struct ConvertOptions {
    pub force: bool,
    pub dry_run: bool,
    pub verify: VerifyMode,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        Self {
            force: false,
            dry_run: false,
            verify: VerifyMode::Structure,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Progress {
    pub tensor_index: usize,
    pub tensor_name: String,
    pub tensors_total: usize,
    pub bytes_done: u64,
    pub bytes_total: u64,
}

#[derive(Debug)]
pub struct Report {
    pub input: PathBuf,
    pub output: PathBuf,
    pub tensors_total: usize,
    pub tensors_pq2: usize,
    pub old_file_size: u64,
    pub new_file_size: u64,
    pub old_data_size: u64,
    pub new_data_size: u64,
    pub elapsed: Duration,
    pub dry_run: bool,
    pub verified: VerifyMode,
}

#[derive(Debug, Clone, Copy)]
struct Plan {
    index: usize,
    new_offset: u64,
    new_size: u64,
}

/// Converts `input` (PQ2_0) into `output` (Q2_0).
///
/// The source file is opened read-only and never modified. The output is
/// written to a temporary file in the destination directory and atomically
/// renamed into place after verification.
pub fn convert(
    input: &Path,
    output: &Path,
    opts: ConvertOptions,
    mut on_progress: Option<&mut dyn FnMut(&Progress)>,
) -> Result<Report, ConvertError> {
    let started = Instant::now();
    let g = Gguf::open(input)?;

    if g.has_ptq1() {
        return Err(ConvertError::Ptq1);
    }
    let pq2_count = g.pq2_count();
    if pq2_count == 0 {
        return Err(ConvertError::NoPq2);
    }
    for t in &g.tensors {
        if t.size.is_none() {
            return Err(ConvertError::UnsupportedTensor {
                name: t.name.clone(),
                ty: t.ty,
            });
        }
    }

    let plans = plan_layout(&g)?;
    let old_data_size = g.file_len - g.data_start;
    let new_data_size = plans
        .iter()
        .map(|p| p.new_offset + p.new_size)
        .max()
        .unwrap_or(0);
    let new_file_size = g
        .data_start
        .checked_add(new_data_size)
        .ok_or(ConvertError::Overflow)?;

    if opts.dry_run {
        if opts.verify == VerifyMode::Full {
            verify_source_full(&g, input)?;
        }
        return Ok(Report {
            input: input.to_path_buf(),
            output: output.to_path_buf(),
            tensors_total: g.tensors.len(),
            tensors_pq2: pq2_count,
            old_file_size: g.file_len,
            new_file_size,
            old_data_size,
            new_data_size,
            elapsed: started.elapsed(),
            dry_run: true,
            verified: opts.verify,
        });
    }

    if paths_same(input, output)? {
        return Err(ConvertError::SameFile);
    }
    if output.exists() && !opts.force {
        return Err(ConvertError::OutputExists(output.to_path_buf()));
    }

    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if !parent.exists() {
        fs::create_dir_all(parent)?;
    }

    let mut tmp = NamedTempFile::new_in(parent)?;
    let mut header = g.header.clone();

    for plan in &plans {
        let t = &g.tensors[plan.index];
        patch_u64(&mut header, t.offset_pos, plan.new_offset)?;
        if t.ty == TY_PQ2_0 {
            patch_u32(&mut header, t.type_pos, TY_Q2_0)?;
        }
    }
    if let Some(pos) = g.file_type_pos {
        patch_u32(&mut header, pos, LLAMA_FTYPE_MOSTLY_Q2_0)?;
    }

    tmp.write_all(&header)?;

    let mut src = File::open(input)?;
    let mut written: u64 = 0;

    for plan in &plans {
        if plan.new_offset < written {
            return Err(ConvertError::Verify(
                "internal error: tensor layout is not monotonic".into(),
            ));
        }
        write_zeros(&mut tmp, plan.new_offset - written)?;

        let t = &g.tensors[plan.index];
        src.seek(SeekFrom::Start(g.data_start + t.offset))?;
        let tensor_size = t.size.expect("checked above");

        if t.ty == TY_PQ2_0 {
            let blocks = tensor_size / 34;
            let mut remaining = blocks;
            let mut inbuf = vec![0u8; REBLOCK_CHUNK_BLOCKS * 34];
            let mut outbuf = vec![0u8; REBLOCK_CHUNK_BLOCKS * 36];
            let mut done = plan.new_offset;

            while remaining > 0 {
                let n = remaining.min(REBLOCK_CHUNK_BLOCKS as u64) as usize;
                src.read_exact(&mut inbuf[..n * 34])?;
                for i in 0..n {
                    reblock_block(
                        &inbuf[i * 34..i * 34 + 34],
                        &mut outbuf[i * 36..i * 36 + 36],
                    );
                }
                tmp.write_all(&outbuf[..n * 36])?;
                remaining -= n as u64;
                done += (n * 36) as u64;
                emit_progress(&mut on_progress, plan, &g, done, new_data_size);
            }
        } else {
            let mut remaining = tensor_size;
            let mut buf = vec![0u8; IO_BUF_SIZE];
            let mut done = plan.new_offset;
            while remaining > 0 {
                let n = remaining.min(IO_BUF_SIZE as u64) as usize;
                src.read_exact(&mut buf[..n])?;
                tmp.write_all(&buf[..n])?;
                remaining -= n as u64;
                done += n as u64;
                emit_progress(&mut on_progress, plan, &g, done, new_data_size);
            }
        }

        written = plan.new_offset + plan.new_size;
    }

    tmp.flush()?;
    tmp.as_file().sync_all()?;

    let out_meta = tmp.as_file().metadata()?;
    if out_meta.len() != new_file_size {
        return Err(ConvertError::Verify(format!(
            "output size is {} bytes, expected {}",
            out_meta.len(),
            new_file_size
        )));
    }

    let g_out = Gguf::open(tmp.path())?;
    if opts.verify != VerifyMode::None {
        verify_structure(&g, &g_out, &plans)?;
    }
    if opts.verify == VerifyMode::Full {
        verify_full(&g, &g_out, input, tmp.path())?;
    }

    if output.exists() {
        if !opts.force {
            return Err(ConvertError::OutputExists(output.to_path_buf()));
        }
        fs::remove_file(output)?;
    }
    tmp.persist(output).map_err(|e| ConvertError::Io(e.error))?;

    Ok(Report {
        input: input.to_path_buf(),
        output: output.to_path_buf(),
        tensors_total: g.tensors.len(),
        tensors_pq2: pq2_count,
        old_file_size: g.file_len,
        new_file_size,
        old_data_size,
        new_data_size,
        elapsed: started.elapsed(),
        dry_run: false,
        verified: opts.verify,
    })
}

fn emit_progress(
    cb: &mut Option<&mut dyn FnMut(&Progress)>,
    plan: &Plan,
    g: &Gguf,
    bytes_done: u64,
    bytes_total: u64,
) {
    if let Some(cb) = cb.as_mut() {
        cb(&Progress {
            tensor_index: plan.index,
            tensor_name: g.tensors[plan.index].name.clone(),
            tensors_total: g.tensors.len(),
            bytes_done,
            bytes_total,
        });
    }
}

fn plan_layout(g: &Gguf) -> Result<Vec<Plan>, ConvertError> {
    let mut order: Vec<usize> = (0..g.tensors.len()).collect();
    order.sort_by_key(|&i| g.tensors[i].offset);

    let mut cursor: u64 = 0;
    let mut plans = Vec::with_capacity(order.len());

    for index in order {
        let t = &g.tensors[index];
        let size = t.size.ok_or_else(|| ConvertError::UnsupportedTensor {
            name: t.name.clone(),
            ty: t.ty,
        })?;

        let new_size = if t.ty == TY_PQ2_0 {
            if size % 34 != 0 {
                return Err(ConvertError::Verify(format!(
                    "tensor '{}' has a PQ2_0 size that is not a multiple of 34",
                    t.name
                )));
            }
            (size / 34).checked_mul(36).ok_or(ConvertError::Overflow)?
        } else {
            size
        };

        let new_offset = align_up(cursor, g.alignment).ok_or(ConvertError::Overflow)?;
        cursor = new_offset
            .checked_add(new_size)
            .ok_or(ConvertError::Overflow)?;
        plans.push(Plan {
            index,
            new_offset,
            new_size,
        });
    }

    Ok(plans)
}

/// Rewrites one 34-byte PQ2_0 block into two 18-byte Q2_0 blocks.
fn reblock_block(src: &[u8], dst: &mut [u8]) {
    debug_assert_eq!(src.len(), 34);
    debug_assert_eq!(dst.len(), 36);
    // Block 1: same d, first 64 codes.
    dst[0..18].copy_from_slice(&src[0..18]);
    // Block 2: same d, second 64 codes.
    dst[18..20].copy_from_slice(&src[0..2]);
    dst[20..36].copy_from_slice(&src[18..34]);
}

fn write_zeros<W: Write>(w: &mut W, mut n: u64) -> io::Result<()> {
    const ZEROS: [u8; 8192] = [0u8; 8192];
    while n > 0 {
        let k = n.min(ZEROS.len() as u64) as usize;
        w.write_all(&ZEROS[..k])?;
        n -= k as u64;
    }
    Ok(())
}

fn paths_same(input: &Path, output: &Path) -> io::Result<bool> {
    let input = fs::canonicalize(input)?;
    if output.exists() {
        return Ok(input == fs::canonicalize(output)?);
    }

    let out_abs = if output.is_absolute() {
        output.to_path_buf()
    } else {
        std::env::current_dir()?.join(output)
    };
    let parent = out_abs.parent().unwrap_or_else(|| Path::new("."));
    let Some(name) = out_abs.file_name() else {
        return Ok(false);
    };
    let parent = fs::canonicalize(parent)?;
    Ok(input == parent.join(name))
}

fn decode_codes(src: &[u8], count: usize, out: &mut [f32]) {
    debug_assert!(out.len() >= count);
    let d = f16::from_bits(u16::from_le_bytes([src[0], src[1]])).to_f32();
    for j in 0..count {
        let byte = src[2 + j / 4];
        let q = (byte >> ((j % 4) * 2)) & 0b11;
        out[j] = (q as i32 - 1) as f32 * d;
    }
}

fn verify_structure(input: &Gguf, output: &Gguf, plans: &[Plan]) -> Result<(), ConvertError> {
    let mut expected = input.header.clone();
    for plan in plans {
        let t = &input.tensors[plan.index];
        patch_u64(&mut expected, t.offset_pos, plan.new_offset)?;
        if t.ty == TY_PQ2_0 {
            patch_u32(&mut expected, t.type_pos, TY_Q2_0)?;
        }
    }
    if let Some(pos) = input.file_type_pos {
        patch_u32(&mut expected, pos, LLAMA_FTYPE_MOSTLY_Q2_0)?;
    }
    if output.header != expected {
        return Err(ConvertError::Verify(
            "output header differs from the expected patched header".into(),
        ));
    }

    if output.version != input.version
        || output.alignment != input.alignment
        || output.data_start != input.data_start
        || output.tensors.len() != input.tensors.len()
    {
        return Err(ConvertError::Verify(
            "output GGUF parameters do not match the input".into(),
        ));
    }

    let mut by_index: Vec<Option<&Plan>> = vec![None; plans.len()];
    for p in plans {
        by_index[p.index] = Some(p);
    }

    for (i, t_in) in input.tensors.iter().enumerate() {
        let p = by_index[i].ok_or_else(|| {
            ConvertError::Verify(format!("missing layout plan for tensor '{}'", t_in.name))
        })?;
        let t_out = &output.tensors[i];
        if t_in.name != t_out.name || t_in.dims != t_out.dims {
            return Err(ConvertError::Verify(format!(
                "tensor name/shape mismatch at index {i}"
            )));
        }
        let expected_ty = if t_in.ty == TY_PQ2_0 {
            TY_Q2_0
        } else {
            t_in.ty
        };
        if t_out.ty != expected_ty {
            return Err(ConvertError::Verify(format!(
                "tensor '{}' has type {}, expected {}",
                t_in.name, t_out.ty, expected_ty
            )));
        }
        if t_out.offset != p.new_offset || t_out.size != Some(p.new_size) {
            return Err(ConvertError::Verify(format!(
                "tensor '{}' has wrong offset/size after conversion",
                t_in.name
            )));
        }
    }

    if output.pq2_count() != 0 {
        return Err(ConvertError::Verify(
            "output still contains PQ2_0 tensors".into(),
        ));
    }
    if input.file_type.is_some() && output.file_type != Some(LLAMA_FTYPE_MOSTLY_Q2_0) {
        return Err(ConvertError::Verify(format!(
            "output general.file_type is {:?}, expected {}",
            output.file_type, LLAMA_FTYPE_MOSTLY_Q2_0
        )));
    }
    if output.has_hadamard_version != input.has_hadamard_version {
        return Err(ConvertError::Verify(
            "prism.hadamard.version metadata was not preserved".into(),
        ));
    }

    let last_end = plans
        .iter()
        .map(|p| p.new_offset + p.new_size)
        .max()
        .unwrap_or(0);
    if output.file_len != input.data_start + last_end {
        return Err(ConvertError::Verify(format!(
            "output file length is {} bytes, expected {}",
            output.file_len,
            input.data_start + last_end
        )));
    }

    Ok(())
}

fn verify_source_full(g: &Gguf, input: &Path) -> Result<(), ConvertError> {
    let mut src = File::open(input)?;
    let mut inbuf = vec![0u8; REBLOCK_CHUNK_BLOCKS * 34];

    for t in g.tensors.iter().filter(|t| t.ty == TY_PQ2_0) {
        let blocks = t.size.expect("checked earlier") / 34;
        src.seek(SeekFrom::Start(g.data_start + t.offset))?;
        let mut remaining = blocks;
        while remaining > 0 {
            let n = remaining.min(REBLOCK_CHUNK_BLOCKS as u64) as usize;
            src.read_exact(&mut inbuf[..n * 34])?;
            for i in 0..n {
                compare_reblocked_block(&inbuf[i * 34..i * 34 + 34], &t.name)?;
            }
            remaining -= n as u64;
        }
    }
    Ok(())
}

fn compare_reblocked_block(src: &[u8], tensor: &str) -> Result<(), ConvertError> {
    let mut old = [0f32; 128];
    decode_codes(src, 128, &mut old);

    let mut first = [0f32; 64];
    decode_codes(&src[0..18], 64, &mut first);

    let mut second_block = [0u8; 18];
    second_block[0..2].copy_from_slice(&src[0..2]);
    second_block[2..18].copy_from_slice(&src[18..34]);
    let mut second = [0f32; 64];
    decode_codes(&second_block, 64, &mut second);

    for j in 0..64 {
        if old[j].to_bits() != first[j].to_bits() {
            return Err(ConvertError::Verify(format!(
                "dry-run mismatch in tensor '{tensor}' at element {j}"
            )));
        }
        if old[64 + j].to_bits() != second[j].to_bits() {
            return Err(ConvertError::Verify(format!(
                "dry-run mismatch in tensor '{tensor}' at element {}",
                64 + j
            )));
        }
    }
    Ok(())
}

fn verify_full(
    g_in: &Gguf,
    g_out: &Gguf,
    in_path: &Path,
    out_path: &Path,
) -> Result<(), ConvertError> {
    let mut src = File::open(in_path)?;
    let mut dst = File::open(out_path)?;

    let mut old_buf = vec![0u8; REBLOCK_CHUNK_BLOCKS * 34];
    let mut new_buf = vec![0u8; REBLOCK_CHUNK_BLOCKS * 36];
    let mut raw_a = vec![0u8; IO_BUF_SIZE];
    let mut raw_b = vec![0u8; IO_BUF_SIZE];

    for (i, t_in) in g_in.tensors.iter().enumerate() {
        let t_out = &g_out.tensors[i];
        if t_in.name != t_out.name {
            return Err(ConvertError::Verify(format!(
                "tensor order mismatch at index {i}"
            )));
        }

        src.seek(SeekFrom::Start(g_in.data_start + t_in.offset))?;
        dst.seek(SeekFrom::Start(g_out.data_start + t_out.offset))?;

        let size_in = t_in.size.expect("checked earlier");
        if t_in.ty == TY_PQ2_0 {
            let blocks = size_in / 34;
            let mut remaining = blocks;
            let mut element_base: u64 = 0;
            while remaining > 0 {
                let n = remaining.min(REBLOCK_CHUNK_BLOCKS as u64) as usize;
                src.read_exact(&mut old_buf[..n * 34])?;
                dst.read_exact(&mut new_buf[..n * 36])?;
                for b in 0..n {
                    compare_reblocked_pair(
                        &old_buf[b * 34..b * 34 + 34],
                        &new_buf[b * 36..b * 36 + 36],
                        &t_in.name,
                        element_base,
                    )?;
                    element_base += 128;
                }
                remaining -= n as u64;
            }
        } else {
            let size_out = t_out.size.expect("checked earlier");
            if size_in != size_out {
                return Err(ConvertError::Verify(format!(
                    "tensor '{}' size changed unexpectedly",
                    t_in.name
                )));
            }
            let mut remaining = size_in;
            while remaining > 0 {
                let n = remaining.min(IO_BUF_SIZE as u64) as usize;
                src.read_exact(&mut raw_a[..n])?;
                dst.read_exact(&mut raw_b[..n])?;
                if raw_a[..n] != raw_b[..n] {
                    return Err(ConvertError::Verify(format!(
                        "raw tensor '{}' differs after conversion",
                        t_in.name
                    )));
                }
                remaining -= n as u64;
            }
        }
    }

    Ok(())
}

fn compare_reblocked_pair(
    old: &[u8],
    new: &[u8],
    tensor: &str,
    element_base: u64,
) -> Result<(), ConvertError> {
    let mut decoded_old = [0f32; 128];
    decode_codes(old, 128, &mut decoded_old);

    let mut decoded_first = [0f32; 64];
    decode_codes(&new[0..18], 64, &mut decoded_first);
    let mut decoded_second = [0f32; 64];
    decode_codes(&new[18..36], 64, &mut decoded_second);

    for j in 0..64 {
        if decoded_old[j].to_bits() != decoded_first[j].to_bits() {
            return Err(ConvertError::Verify(format!(
                "value mismatch in tensor '{tensor}' at element {}",
                element_base + j as u64
            )));
        }
        if decoded_old[64 + j].to_bits() != decoded_second[j].to_bits() {
            return Err(ConvertError::Verify(format!(
                "value mismatch in tensor '{tensor}' at element {}",
                element_base + 64 + j as u64
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn reblock_splits_codes_in_order() {
        let mut src = [0u8; 34];
        src[0] = 0x00;
        src[1] = 0x3c; // f16 1.0
        for (i, b) in src[2..].iter_mut().enumerate() {
            *b = i as u8;
        }
        let mut dst = [0u8; 36];
        reblock_block(&src, &mut dst);
        assert_eq!(&dst[0..2], &src[0..2]);
        assert_eq!(&dst[2..18], &src[2..18]);
        assert_eq!(&dst[18..20], &src[0..2]);
        assert_eq!(&dst[20..36], &src[18..34]);
    }

    proptest! {
        #[test]
        fn reblock_preserves_decoded_values(
            d_bits in any::<u16>(),
            codes in proptest::collection::vec(any::<u8>(), 32),
        ) {
            let mut src = [0u8; 34];
            src[0..2].copy_from_slice(&d_bits.to_le_bytes());
            src[2..34].copy_from_slice(&codes);
            let mut dst = [0u8; 36];
            reblock_block(&src, &mut dst);

            let mut old = [0f32; 128];
            decode_codes(&src, 128, &mut old);
            let mut first = [0f32; 64];
            decode_codes(&dst[0..18], 64, &mut first);
            let mut second = [0f32; 64];
            decode_codes(&dst[18..36], 64, &mut second);

            for j in 0..64 {
                prop_assert_eq!(old[j].to_bits(), first[j].to_bits());
                prop_assert_eq!(old[64 + j].to_bits(), second[j].to_bits());
            }
        }
    }
}
