//! Native dequantization for safetensors checkpoints.
//!
//! This is deliberately narrower than a full safetensors -> GGUF converter.
//! It reads an existing `.safetensors` file, dequantizes FP8 and AutoAWQ 4-bit
//! GEMM tensors to BF16, and writes another `.safetensors` file. The resulting
//! checkpoint can then be converted by the optional `convert_hf_to_gguf.py`
//! bridge, which still owns Hugging Face architecture/name mapping.
//!
//! AWQ layout follows AutoAWQ GEMM: `qweight` is `[in, out/8]` packed I32,
//! `qzeros` is `[in/group, out/8]`, `scales` is `[in/group, out]`, and the
//! eight nibbles use `AWQ_REVERSE_ORDER`.

use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use half::bf16;
use serde_json::{json, Map, Value};
use tempfile::NamedTempFile;

const MAX_HEADER: u64 = 128 * 1024 * 1024;
const AWQ_REVERSE_ORDER: [usize; 8] = [0, 4, 1, 5, 2, 6, 3, 7];

#[derive(Debug, thiserror::Error)]
pub enum NativeError {
    #[error("I/O error: {0}")]
    Io(String),
    #[error("invalid safetensors header: {0}")]
    Header(String),
    #[error("tensor '{name}' uses unsupported dtype {dtype}")]
    UnsupportedDtype { name: String, dtype: String },
    #[error("tensor '{name}' is malformed: {msg}")]
    Tensor { name: String, msg: String },
    #[error("AWQ tensor '{name}' is malformed: {msg}")]
    Awq { name: String, msg: String },
}

#[derive(Debug, Clone)]
pub(crate) struct Entry {
    pub(crate) dtype: String,
    pub(crate) shape: Vec<u64>,
    pub(crate) start: u64,
    pub(crate) end: u64,
}

#[derive(Debug, Clone)]
pub struct NativeReport {
    pub input: String,
    pub output: String,
    pub tensors_in: usize,
    pub tensors_out: usize,
    pub fp8_tensors: usize,
    pub awq_tensors: usize,
}

pub(crate) struct Reader {
    file: File,
    data_start: u64,
    pub(crate) entries: BTreeMap<String, Entry>,
}

impl Reader {
    pub(crate) fn open(path: &Path) -> Result<Self, NativeError> {
        let mut file = File::open(path).map_err(|error| NativeError::Io(error.to_string()))?;
        let mut len = [0u8; 8];
        file.read_exact(&mut len)
            .map_err(|error| NativeError::Header(error.to_string()))?;
        let header_len = u64::from_le_bytes(len);
        if header_len == 0 || header_len > MAX_HEADER {
            return Err(NativeError::Header(format!(
                "unreasonable header length {header_len}"
            )));
        }
        let mut raw = vec![0u8; header_len as usize];
        file.read_exact(&mut raw)
            .map_err(|error| NativeError::Header(error.to_string()))?;
        let value: Value =
            serde_json::from_slice(&raw).map_err(|error| NativeError::Header(error.to_string()))?;
        let object = value
            .as_object()
            .ok_or_else(|| NativeError::Header("header is not an object".into()))?;
        let data_start = 8 + header_len;
        let file_len = file
            .metadata()
            .map_err(|error| NativeError::Io(error.to_string()))?
            .len();
        let mut entries = BTreeMap::new();
        for (name, item) in object {
            if name == "__metadata__" {
                continue;
            }
            let dtype = item
                .get("dtype")
                .and_then(Value::as_str)
                .ok_or_else(|| NativeError::Header(format!("'{name}' has no dtype")))?
                .to_string();
            let shape = item
                .get("shape")
                .and_then(Value::as_array)
                .ok_or_else(|| NativeError::Header(format!("'{name}' has no shape")))?
                .iter()
                .map(|v| {
                    v.as_u64().ok_or_else(|| {
                        NativeError::Header(format!("'{name}' has an invalid shape"))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let offsets = item
                .get("data_offsets")
                .and_then(Value::as_array)
                .ok_or_else(|| NativeError::Header(format!("'{name}' has no data_offsets")))?;
            if offsets.len() != 2 {
                return Err(NativeError::Header(format!(
                    "'{name}' has invalid data_offsets"
                )));
            }
            let start = offsets[0]
                .as_u64()
                .ok_or_else(|| NativeError::Header(format!("'{name}' has an invalid offset")))?;
            let end = offsets[1]
                .as_u64()
                .ok_or_else(|| NativeError::Header(format!("'{name}' has an invalid offset")))?;
            if end < start || data_start.saturating_add(end) > file_len {
                return Err(NativeError::Header(format!(
                    "'{name}' data range is outside the file"
                )));
            }
            entries.insert(
                name.clone(),
                Entry {
                    dtype,
                    shape,
                    start,
                    end,
                },
            );
        }
        Ok(Self {
            file,
            data_start,
            entries,
        })
    }

    pub(crate) fn bytes(&mut self, name: &str) -> Result<Vec<u8>, NativeError> {
        let entry = self.entries.get(name).ok_or_else(|| NativeError::Tensor {
            name: name.to_string(),
            msg: "missing tensor".into(),
        })?;
        let len = entry.end - entry.start;
        if len > usize::MAX as u64 {
            return Err(NativeError::Tensor {
                name: name.to_string(),
                msg: "tensor is too large".into(),
            });
        }
        let mut data = vec![0u8; len as usize];
        self.file
            .seek(SeekFrom::Start(self.data_start + entry.start))
            .map_err(|error| NativeError::Io(error.to_string()))?;
        self.file
            .read_exact(&mut data)
            .map_err(|error| NativeError::Io(error.to_string()))?;
        Ok(data)
    }

    pub(crate) fn shape(&self, name: &str) -> Option<&[u64]> {
        self.entries.get(name).map(|entry| entry.shape.as_slice())
    }
}

pub(crate) fn element_count(shape: &[u64]) -> Result<usize, NativeError> {
    shape.iter().try_fold(1usize, |acc, dim| {
        usize::try_from(*dim)
            .ok()
            .and_then(|dim| acc.checked_mul(dim))
            .ok_or_else(|| NativeError::Header("tensor shape is too large".into()))
    })
}

pub(crate) fn decode_floats(
    name: &str,
    dtype: &str,
    raw: &[u8],
    count: usize,
) -> Result<Vec<f32>, NativeError> {
    let width = match dtype {
        "F32" => 4,
        "F16" | "BF16" => 2,
        "F8_E4M3" | "F8_E5M2" | "F8_E4M3FN" => 1,
        _ => 0,
    };
    if width == 0 {
        return Err(NativeError::UnsupportedDtype {
            name: name.to_string(),
            dtype: dtype.to_string(),
        });
    }
    if raw.len() != count.saturating_mul(width) {
        return Err(NativeError::Tensor {
            name: name.to_string(),
            msg: format!("expected {} bytes, got {}", count * width, raw.len()),
        });
    }
    match dtype {
        "F32" => Ok(raw
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four bytes")))
            .collect()),
        "F16" => Ok(raw
            .chunks_exact(2)
            .map(|chunk| half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
            .collect()),
        "BF16" => Ok(raw
            .chunks_exact(2)
            .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
            .collect()),
        "F8_E4M3" | "F8_E4M3FN" => Ok(raw.iter().map(|byte| fp8_e4m3(*byte)).collect()),
        "F8_E5M2" => Ok(raw.iter().map(|byte| fp8_e5m2(*byte)).collect()),
        _ => unreachable!("dtype was checked above"),
    }
}

pub(crate) fn decode_u32(name: &str, raw: &[u8]) -> Result<Vec<u32>, NativeError> {
    if !raw.len().is_multiple_of(4) {
        return Err(NativeError::Tensor {
            name: name.to_string(),
            msg: "I32 tensor is not a multiple of four bytes".into(),
        });
    }
    Ok(raw
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four bytes")))
        .collect())
}

fn fp8_e4m3(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = ((byte >> 3) & 0x0f) as i32;
    let mant = (byte & 0x07) as f32;
    if exp == 0 {
        sign * (mant / 8.0) * 2f32.powi(-6)
    } else if exp == 15 && mant == 7.0 {
        // E4M3FN has no infinities: 448 is the largest finite value and only
        // a full exponent plus a full mantissa encodes NaN.
        f32::NAN
    } else {
        sign * (1.0 + mant / 8.0) * 2f32.powi(exp - 7)
    }
}

fn fp8_e5m2(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = ((byte >> 2) & 0x1f) as i32;
    let mant = (byte & 0x03) as f32;
    if exp == 0 {
        sign * (mant / 4.0) * 2f32.powi(-14)
    } else if exp == 31 {
        if mant == 0.0 {
            sign * f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        sign * (1.0 + mant / 4.0) * 2f32.powi(exp - 15)
    }
}

pub(crate) fn awq_dequant(
    name: &str,
    qweight: &[u32],
    qzeros: Option<&[u32]>,
    scales: &[f32],
    qweight_shape: &[u64],
    qzeros_shape: &[u64],
    scales_shape: &[u64],
) -> Result<(Vec<u64>, Vec<u8>), NativeError> {
    if qweight_shape.len() != 2
        || scales_shape.len() != 2
        || (qzeros.is_some() && qzeros_shape.len() != 2)
    {
        return Err(NativeError::Awq {
            name: name.to_string(),
            msg: "qweight/qzeros/scales must be two-dimensional".into(),
        });
    }
    let in_features = usize::try_from(qweight_shape[0]).map_err(|_| NativeError::Awq {
        name: name.to_string(),
        msg: "in_features does not fit".into(),
    })?;
    let packed = usize::try_from(qweight_shape[1]).map_err(|_| NativeError::Awq {
        name: name.to_string(),
        msg: "packed width does not fit".into(),
    })?;
    let out_features = packed.checked_mul(8).ok_or_else(|| NativeError::Awq {
        name: name.to_string(),
        msg: "out_features overflow".into(),
    })?;
    let groups = usize::try_from(scales_shape[0]).map_err(|_| NativeError::Awq {
        name: name.to_string(),
        msg: "group count does not fit".into(),
    })?;
    if in_features == 0 || out_features == 0 || groups == 0 || in_features % groups != 0 {
        return Err(NativeError::Awq {
            name: name.to_string(),
            msg: "invalid in/out/group dimensions".into(),
        });
    }
    if scales_shape[1] as usize != out_features
        || qweight.len() != in_features * packed
        || scales.len() != groups * out_features
        || qzeros.is_some_and(|zeros| {
            zeros.len() != groups * packed
                || qzeros_shape[0] as usize != groups
                || qzeros_shape[1] as usize != packed
        })
    {
        return Err(NativeError::Awq {
            name: name.to_string(),
            msg: "tensor shapes do not describe one AWQ matrix".into(),
        });
    }
    let group_size = in_features / groups;
    // HF stores Linear.weight as [out_features, in_features]. AWQ qweight is
    // packed as [in_features, out_features/8], so transpose while unpacking.
    let mut out = Vec::with_capacity(in_features * out_features * 2);
    for col in 0..out_features {
        for row in 0..in_features {
            let group = row / group_size;
            let block = col / 8;
            let position = col % 8;
            let shift = AWQ_REVERSE_ORDER[position] * 4;
            let q = (qweight[row * packed + block] >> shift) & 0x0f;
            let z = qzeros
                .map(|zeros| (zeros[group * packed + block] >> shift) & 0x0f)
                .unwrap_or(0);
            let scale = scales[group * out_features + col];
            let value = (q as f32 - z as f32) * scale;
            out.extend_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
        }
    }
    Ok((vec![out_features as u64, in_features as u64], out))
}

fn read_i32_tensor(reader: &mut Reader, name: &str) -> Result<Vec<u32>, NativeError> {
    let entry = reader
        .entries
        .get(name)
        .ok_or_else(|| NativeError::Tensor {
            name: name.to_string(),
            msg: "missing tensor".into(),
        })?
        .clone();
    if entry.dtype != "I32" {
        return Err(NativeError::UnsupportedDtype {
            name: name.to_string(),
            dtype: entry.dtype,
        });
    }
    decode_u32(name, &reader.bytes(name)?)
}

fn read_float_tensor(reader: &mut Reader, name: &str) -> Result<Vec<f32>, NativeError> {
    let entry = reader
        .entries
        .get(name)
        .ok_or_else(|| NativeError::Tensor {
            name: name.to_string(),
            msg: "missing tensor".into(),
        })?
        .clone();
    let count = element_count(&entry.shape)?;
    decode_floats(name, &entry.dtype, &reader.bytes(name)?, count)
}

/// Dequantize an FP8/AWQ safetensors checkpoint into a BF16 safetensors file.
/// Plain F32/F16/BF16 tensors are copied as BF16 too.
pub fn dequantize_file(
    input: &Path,
    output: &Path,
    force: bool,
) -> Result<NativeReport, NativeError> {
    if input == output {
        return Err(NativeError::Io(
            "input and output must be different files".into(),
        ));
    }
    if output.exists() && !force {
        return Err(NativeError::Io(format!(
            "output already exists: {}",
            output.display()
        )));
    }
    let mut reader = Reader::open(input)?;
    let mut awq_prefixes = std::collections::BTreeSet::new();
    for name in reader.entries.keys() {
        if let Some(prefix) = name.strip_suffix(".qweight") {
            if reader.entries.contains_key(&format!("{prefix}.scales")) {
                awq_prefixes.insert(prefix.to_string());
            }
        }
    }
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(|error| NativeError::Io(error.to_string()))?;
    let mut payload =
        NamedTempFile::new_in(parent).map_err(|error| NativeError::Io(error.to_string()))?;
    let mut out_entries = Map::new();
    let mut cursor = 0u64;
    let mut tensors_out = 0usize;
    let mut fp8_tensors = 0usize;
    let mut awq_tensors = 0usize;
    let mut emitted = std::collections::BTreeSet::new();

    for name in reader.entries.keys().cloned().collect::<Vec<_>>() {
        if name.ends_with(".qzeros") || name.ends_with(".scales") {
            continue;
        }
        let (out_name, shape, data, is_awq, is_fp8) =
            if let Some(prefix) = name.strip_suffix(".qweight") {
                if !awq_prefixes.contains(prefix) {
                    continue;
                }
                let qweight_shape = reader.shape(&name).unwrap_or(&[]).to_vec();
                let qzeros_name = format!("{prefix}.qzeros");
                let qzeros = if reader.entries.contains_key(&qzeros_name) {
                    Some(read_i32_tensor(&mut reader, &qzeros_name)?)
                } else {
                    None
                };
                let qzeros_shape = reader.shape(&qzeros_name).unwrap_or(&[]).to_vec();
                let scales_name = format!("{prefix}.scales");
                let scales_shape = reader.shape(&scales_name).unwrap_or(&[]).to_vec();
                let scales = read_float_tensor(&mut reader, &scales_name)?;
                let qweight = read_i32_tensor(&mut reader, &name)?;
                let (shape, data) = awq_dequant(
                    prefix,
                    &qweight,
                    qzeros.as_deref(),
                    &scales,
                    &qweight_shape,
                    &qzeros_shape,
                    &scales_shape,
                )?;
                (format!("{prefix}.weight"), shape, data, true, false)
            } else {
                let entry = reader.entries[&name].clone();
                let count = element_count(&entry.shape)?;
                let floats = decode_floats(&name, &entry.dtype, &reader.bytes(&name)?, count)?;
                let mut data = Vec::with_capacity(floats.len() * 2);
                for value in floats {
                    data.extend_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
                }
                let fp8 = entry.dtype.starts_with("F8_");
                (name.clone(), entry.shape, data, false, fp8)
            };
        if !emitted.insert(out_name.clone()) {
            return Err(NativeError::Header(format!(
                "duplicate output tensor '{out_name}'"
            )));
        }
        payload
            .write_all(&data)
            .map_err(|error| NativeError::Io(error.to_string()))?;
        let end = cursor + data.len() as u64;
        out_entries.insert(
            out_name,
            json!({
                "dtype": "BF16",
                "shape": shape,
                "data_offsets": [cursor, end],
            }),
        );
        cursor = end;
        tensors_out += 1;
        awq_tensors += is_awq as usize;
        fp8_tensors += is_fp8 as usize;
    }
    let header = Value::Object(out_entries);
    let mut raw_header =
        serde_json::to_vec(&header).map_err(|error| NativeError::Header(error.to_string()))?;
    while !raw_header.len().is_multiple_of(8) {
        raw_header.push(b' ');
    }
    let tmp_output = output.with_extension("tmp.safetensors");
    let mut out = File::create(&tmp_output).map_err(|error| NativeError::Io(error.to_string()))?;
    out.write_all(&(raw_header.len() as u64).to_le_bytes())
        .map_err(|error| NativeError::Io(error.to_string()))?;
    out.write_all(&raw_header)
        .map_err(|error| NativeError::Io(error.to_string()))?;
    payload
        .as_file_mut()
        .seek(SeekFrom::Start(0))
        .map_err(|error| NativeError::Io(error.to_string()))?;
    std::io::copy(payload.as_file_mut(), &mut out)
        .map_err(|error| NativeError::Io(error.to_string()))?;
    out.flush()
        .map_err(|error| NativeError::Io(error.to_string()))?;
    if output.exists() {
        fs::remove_file(output).map_err(|error| NativeError::Io(error.to_string()))?;
    }
    fs::rename(&tmp_output, output).map_err(|error| NativeError::Io(error.to_string()))?;
    Ok(NativeReport {
        input: input.display().to_string(),
        output: output.display().to_string(),
        tensors_in: reader.entries.len(),
        tensors_out,
        fp8_tensors,
        awq_tensors,
    })
}

/// A quick inventory used by the CLI before a native dequantization.
pub fn inspect_file(path: &Path) -> Result<HashMap<String, (String, Vec<u64>)>, NativeError> {
    let reader = Reader::open(path)?;
    Ok(reader
        .entries
        .into_iter()
        .map(|(name, entry)| (name, (entry.dtype, entry.shape)))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp8_e4m3_decodes_one_and_negative_one() {
        assert!((fp8_e4m3(0x38) - 1.0).abs() < 1e-6);
        assert!((fp8_e4m3(0xb8) + 1.0).abs() < 1e-6);
        assert_eq!(fp8_e4m3(0x7e), 448.0);
        assert_eq!(fp8_e4m3(0xfe), -448.0);
        assert!(fp8_e4m3(0x7f).is_nan());
    }

    #[test]
    fn awq_reverses_nibble_order() {
        let qweight = [0x8765_4321u32];
        let scales = [1.0f32; 8];
        let (shape, data) =
            awq_dequant("test", &qweight, None, &scales, &[1, 1], &[1, 1], &[1, 8]).unwrap();
        assert_eq!(shape, vec![8, 1]);
        let values = data
            .chunks_exact(2)
            .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
            .collect::<Vec<_>>();
        assert_eq!(values, vec![1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0]);
    }

    #[test]
    fn awq_applies_packed_zero_points() {
        let qweight = [0x8765_4321u32];
        let qzeros = [0x1111_1111u32];
        let scales = [1.0f32; 8];
        let (shape, data) = awq_dequant(
            "test",
            &qweight,
            Some(&qzeros),
            &scales,
            &[1, 1],
            &[1, 1],
            &[1, 8],
        )
        .unwrap();
        assert_eq!(shape, vec![8, 1]);
        let values = data
            .chunks_exact(2)
            .map(|chunk| bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32())
            .collect::<Vec<_>>();
        assert_eq!(values, vec![0.0, 4.0, 1.0, 5.0, 2.0, 6.0, 3.0, 7.0]);
    }

    #[test]
    fn dequantizes_a_tiny_fp8_file() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in.safetensors");
        let output = dir.path().join("out.safetensors");
        let mut header = br#"{"x":{"dtype":"F8_E4M3","shape":[2],"data_offsets":[0,2]}}"#.to_vec();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut file = File::create(&input).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(&header).unwrap();
        file.write_all(&[0x38, 0xb8]).unwrap();
        drop(file);

        let report = dequantize_file(&input, &output, false).unwrap();
        assert_eq!(report.fp8_tensors, 1);
        assert_eq!(report.tensors_out, 1);
        let mut out = File::open(&output).unwrap();
        let mut len = [0u8; 8];
        out.read_exact(&mut len).unwrap();
        let mut raw = vec![0u8; u64::from_le_bytes(len) as usize];
        out.read_exact(&mut raw).unwrap();
        let value: Value = serde_json::from_slice(&raw).unwrap();
        assert_eq!(value["x"]["dtype"], "BF16");
        assert_eq!(value["x"]["shape"], json!([2]));
    }
}
