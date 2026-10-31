//! Minimal, dependency-free GGUF v3 reader and header patcher.
//!
//! The reader keeps the raw header bytes (including the padding up to the
//! tensor-data section) so callers can patch fixed-width fields in place and
//! write them back without re-encoding any metadata.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

use thiserror::Error;

use crate::types::{type_info, TY_PQ2_0, TY_PTQ1_0};

pub const DEFAULT_ALIGNMENT: u64 = 32;

const MAX_STRING_LEN: u64 = 256 * 1024 * 1024;
const MAX_ARRAY_COUNT: u64 = 1 << 30;
const MAX_TENSORS: u64 = 1_000_000;
const MAX_KV: u64 = 1_000_000;
const MAX_DIMS: u32 = 8;
const MAX_HEADER_BYTES: u64 = 4 * 1024 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum GgufError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("not a GGUF file (bad magic)")]
    BadMagic,
    #[error("unsupported GGUF version {0}; only version 3 is supported")]
    UnsupportedVersion(u32),
    #[error("malformed GGUF: {0}")]
    Malformed(String),
    #[error("unsupported metadata value type {0}")]
    UnsupportedValueType(u32),
    #[error("tensor '{name}' has unsupported type {ty}")]
    UnsupportedTensorType { name: String, ty: u32 },
    #[error("tensor '{name}' is invalid: {msg}")]
    InvalidTensor { name: String, msg: String },
    #[error("tensor '{name}' is misaligned or out of bounds")]
    BadTensorOffset { name: String },
    #[error("tensor data overlaps: '{a}' and '{b}'")]
    OverlappingTensors { a: String, b: String },
    #[error("invalid GGUF alignment {0}")]
    InvalidAlignment(u64),
}

#[derive(Debug, Clone)]
pub struct KvEntry {
    pub key: String,
    pub value_type: u32,
    /// Absolute file offset of the value payload (after key and type).
    pub value_pos: u64,
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ty: u32,
    pub offset: u64,
    /// Absolute file offset of the type field (u32) in the header.
    pub type_pos: u64,
    /// Absolute file offset of the offset field (u64) in the header.
    pub offset_pos: u64,
    /// On-disk byte size, or `None` when the type layout is unknown.
    pub size: Option<u64>,
}

impl TensorInfo {
    pub fn element_count(&self) -> u64 {
        self.dims.iter().fold(1u64, |acc, d| acc.saturating_mul(*d))
    }
}

#[derive(Debug)]
pub struct Gguf {
    pub version: u32,
    pub file_len: u64,
    pub data_start: u64,
    pub alignment: u64,
    /// Raw bytes of the header, including padding up to `data_start`.
    pub header: Vec<u8>,
    pub kv: Vec<KvEntry>,
    pub tensors: Vec<TensorInfo>,
    pub file_type: Option<u32>,
    pub file_type_pos: Option<u64>,
    pub has_hadamard_version: bool,
}

impl Gguf {
    pub fn open(path: &Path) -> Result<Self, GgufError> {
        let mut file = File::open(path)?;
        let file_len = file.metadata()?.len();
        let mut r = Reader::new(&mut file);

        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != b"GGUF" {
            return Err(GgufError::BadMagic);
        }
        let version = r.u32()?;
        if version != 3 {
            return Err(GgufError::UnsupportedVersion(version));
        }
        let tensor_count = r.u64()?;
        if tensor_count == 0 {
            return Err(GgufError::Malformed("file contains no tensors".into()));
        }
        if tensor_count > MAX_TENSORS {
            return Err(GgufError::Malformed(format!(
                "unreasonable tensor count {tensor_count}"
            )));
        }
        let kv_count = r.u64()?;
        if kv_count > MAX_KV {
            return Err(GgufError::Malformed(format!(
                "unreasonable metadata entry count {kv_count}"
            )));
        }

        let mut kv = Vec::with_capacity(kv_count.min(4096) as usize);
        let mut alignment = DEFAULT_ALIGNMENT;
        let mut alignment_seen = false;
        let mut file_type = None;
        let mut file_type_pos = None;
        let mut has_hadamard_version = false;

        for _ in 0..kv_count {
            let key = r.string()?;
            let value_type = r.u32()?;
            let value_pos = r.pos;

            if value_type == 4 {
                // UINT32: read the payload so we can pick up alignment/file_type.
                let value = r.u32()?;
                match key.as_str() {
                    "general.alignment" => {
                        if alignment_seen {
                            return Err(GgufError::Malformed(
                                "duplicate metadata key 'general.alignment'".into(),
                            ));
                        }
                        alignment = value as u64;
                        alignment_seen = true;
                    }
                    "general.file_type" => {
                        if file_type.is_some() {
                            return Err(GgufError::Malformed(
                                "duplicate metadata key 'general.file_type'".into(),
                            ));
                        }
                        file_type = Some(value);
                        file_type_pos = Some(value_pos);
                    }
                    "prism.hadamard.version" => has_hadamard_version = true,
                    _ => {}
                }
            } else {
                if key == "general.alignment" || key == "general.file_type" {
                    return Err(GgufError::Malformed(format!(
                        "metadata '{key}' must be UINT32"
                    )));
                }
                walk_value(&mut r, value_type)?;
                if key == "prism.hadamard.version" {
                    has_hadamard_version = true;
                }
            }

            kv.push(KvEntry {
                key,
                value_type,
                value_pos,
            });
        }

        if alignment == 0 || !alignment.is_power_of_two() || alignment > (1 << 24) {
            return Err(GgufError::InvalidAlignment(alignment));
        }

        let mut tensors = Vec::with_capacity(tensor_count.min(65536) as usize);
        for _ in 0..tensor_count {
            let name = r.string()?;
            let n_dims = r.u32()?;
            if n_dims == 0 || n_dims > MAX_DIMS {
                return Err(GgufError::InvalidTensor {
                    name,
                    msg: format!("invalid dimension count {n_dims}"),
                });
            }
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                let d = r.u64()?;
                if d == 0 {
                    return Err(GgufError::InvalidTensor {
                        name,
                        msg: "zero-sized dimension".into(),
                    });
                }
                dims.push(d);
            }

            let type_pos = r.pos;
            let ty = r.u32()?;
            let offset_pos = r.pos;
            let offset = r.u64()?;

            let elems = dims
                .iter()
                .try_fold(1u64, |acc, d| acc.checked_mul(*d))
                .ok_or_else(|| GgufError::InvalidTensor {
                    name: name.clone(),
                    msg: "element count overflow".into(),
                })?;

            let size = match type_info(ty) {
                Some(info) => {
                    if elems % info.block != 0 {
                        return Err(GgufError::InvalidTensor {
                            name: name.clone(),
                            msg: format!(
                                "element count {elems} is not a multiple of block size {}",
                                info.block
                            ),
                        });
                    }
                    Some(
                        (elems / info.block)
                            .checked_mul(info.bytes)
                            .ok_or_else(|| GgufError::InvalidTensor {
                                name: name.clone(),
                                msg: "tensor byte size overflow".into(),
                            })?,
                    )
                }
                None => None,
            };

            tensors.push(TensorInfo {
                name,
                dims,
                ty,
                offset,
                type_pos,
                offset_pos,
                size,
            });
        }

        let header_end = r.pos;

        let data_start = align_up(header_end, alignment)
            .ok_or_else(|| GgufError::Malformed("tensor data start offset overflow".into()))?;
        if data_start > file_len {
            return Err(GgufError::Malformed(format!(
                "tensor data start {data_start} is beyond end of file ({file_len})"
            )));
        }
        if data_start > MAX_HEADER_BYTES {
            return Err(GgufError::Malformed(format!(
                "header is unreasonably large ({data_start} bytes)"
            )));
        }

        let mut header = vec![0u8; data_start as usize];
        file.seek(SeekFrom::Start(0))?;
        file.read_exact(&mut header)?;

        let data_len = file_len - data_start;
        let mut known: Vec<usize> = tensors
            .iter()
            .enumerate()
            .filter(|(_, t)| t.size.is_some())
            .map(|(i, _)| i)
            .collect();

        for &i in &known {
            let t = &tensors[i];
            let size = t.size.unwrap();
            if t.offset % alignment != 0 {
                return Err(GgufError::BadTensorOffset {
                    name: t.name.clone(),
                });
            }
            let end = t
                .offset
                .checked_add(size)
                .ok_or_else(|| GgufError::BadTensorOffset {
                    name: t.name.clone(),
                })?;
            if end > data_len {
                return Err(GgufError::BadTensorOffset {
                    name: t.name.clone(),
                });
            }
        }

        known.sort_by_key(|&i| tensors[i].offset);
        for pair in known.windows(2) {
            let a = &tensors[pair[0]];
            let b = &tensors[pair[1]];
            if a.offset + a.size.unwrap() > b.offset {
                return Err(GgufError::OverlappingTensors {
                    a: a.name.clone(),
                    b: b.name.clone(),
                });
            }
        }

        Ok(Self {
            version,
            file_len,
            data_start,
            alignment,
            header,
            kv,
            tensors,
            file_type,
            file_type_pos,
            has_hadamard_version,
        })
    }

    pub fn pq2_count(&self) -> usize {
        self.tensors.iter().filter(|t| t.ty == TY_PQ2_0).count()
    }

    pub fn has_ptq1(&self) -> bool {
        self.tensors.iter().any(|t| t.ty == TY_PTQ1_0)
    }

    pub fn type_histogram(&self) -> BTreeMap<u32, usize> {
        let mut map = BTreeMap::new();
        for t in &self.tensors {
            *map.entry(t.ty).or_insert(0) += 1;
        }
        map
    }

    pub fn tensor_data_offset(&self, t: &TensorInfo) -> u64 {
        self.data_start + t.offset
    }
}

/// Aligns `value` up to the next multiple of `alignment`.
pub fn align_up(value: u64, alignment: u64) -> Option<u64> {
    if alignment == 0 {
        return None;
    }
    let rem = value % alignment;
    if rem == 0 {
        Some(value)
    } else {
        value.checked_add(alignment - rem)
    }
}

pub fn patch_u32(header: &mut [u8], pos: u64, value: u32) -> Result<(), GgufError> {
    let p = usize::try_from(pos)
        .map_err(|_| GgufError::Malformed("header offset does not fit in memory".into()))?;
    let end = p
        .checked_add(4)
        .ok_or_else(|| GgufError::Malformed("header offset overflow".into()))?;
    if end > header.len() {
        return Err(GgufError::Malformed(
            "patch position is outside the header".into(),
        ));
    }
    header[p..end].copy_from_slice(&value.to_le_bytes());
    Ok(())
}

pub fn patch_u64(header: &mut [u8], pos: u64, value: u64) -> Result<(), GgufError> {
    let p = usize::try_from(pos)
        .map_err(|_| GgufError::Malformed("header offset does not fit in memory".into()))?;
    let end = p
        .checked_add(8)
        .ok_or_else(|| GgufError::Malformed("header offset overflow".into()))?;
    if end > header.len() {
        return Err(GgufError::Malformed(
            "patch position is outside the header".into(),
        ));
    }
    header[p..end].copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn fixed_value_size(ty: u32) -> Option<u64> {
    match ty {
        0 | 1 | 7 => Some(1), // UINT8, INT8, BOOL
        2..=3 => Some(2),     // UINT16, INT16
        4..=6 => Some(4),     // UINT32, INT32, FLOAT32
        10..=12 => Some(8),   // UINT64, INT64, FLOAT64
        _ => None,
    }
}

fn walk_value<R: Read>(r: &mut Reader<R>, ty: u32) -> Result<(), GgufError> {
    match ty {
        0 | 1 | 7 => r.skip(1),
        2..=3 => r.skip(2),
        4..=6 => r.skip(4),
        8 => {
            let len = r.u64()?;
            if len > MAX_STRING_LEN {
                return Err(GgufError::Malformed("metadata string too long".into()));
            }
            r.skip(len)
        }
        9 => {
            let elem_ty = r.u32()?;
            let count = r.u64()?;
            if count > MAX_ARRAY_COUNT {
                return Err(GgufError::Malformed("metadata array too large".into()));
            }
            if elem_ty == 8 {
                for _ in 0..count {
                    let len = r.u64()?;
                    if len > MAX_STRING_LEN {
                        return Err(GgufError::Malformed("metadata string too long".into()));
                    }
                    r.skip(len)?;
                }
            } else if elem_ty == 9 {
                return Err(GgufError::Malformed(
                    "nested metadata arrays are not allowed".into(),
                ));
            } else {
                let sz =
                    fixed_value_size(elem_ty).ok_or(GgufError::UnsupportedValueType(elem_ty))?;
                let total = sz
                    .checked_mul(count)
                    .ok_or_else(|| GgufError::Malformed("metadata array size overflow".into()))?;
                r.skip(total)?;
            }
            Ok(())
        }
        10..=12 => r.skip(8),
        _ => Err(GgufError::UnsupportedValueType(ty)),
    }
}

struct Reader<R> {
    inner: R,
    pos: u64,
}

impl<R: Read> Reader<R> {
    fn new(inner: R) -> Self {
        Self { inner, pos: 0 }
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> Result<(), GgufError> {
        self.inner.read_exact(buf)?;
        self.pos += buf.len() as u64;
        Ok(())
    }

    fn u32(&mut self) -> Result<u32, GgufError> {
        let mut b = [0u8; 4];
        self.read_exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    fn u64(&mut self) -> Result<u64, GgufError> {
        let mut b = [0u8; 8];
        self.read_exact(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }

    fn string(&mut self) -> Result<String, GgufError> {
        let len = self.u64()?;
        if len > MAX_STRING_LEN {
            return Err(GgufError::Malformed("metadata string too long".into()));
        }
        let mut buf = vec![0u8; len as usize];
        self.read_exact(&mut buf)?;
        String::from_utf8(buf)
            .map_err(|e| GgufError::Malformed(format!("invalid UTF-8 string: {e}")))
    }

    fn skip(&mut self, n: u64) -> Result<(), GgufError> {
        let mut taken = (&mut self.inner).take(n);
        let copied = io::copy(&mut taken, &mut io::sink())?;
        if copied != n {
            return Err(GgufError::Io(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "unexpected end of file",
            )));
        }
        self.pos += n;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alignment_rounds_up() {
        assert_eq!(align_up(0, 32), Some(0));
        assert_eq!(align_up(1, 32), Some(32));
        assert_eq!(align_up(31, 32), Some(32));
        assert_eq!(align_up(32, 32), Some(32));
        assert_eq!(align_up(33, 32), Some(64));
        assert_eq!(align_up(u64::MAX, 32), None);
    }
}
