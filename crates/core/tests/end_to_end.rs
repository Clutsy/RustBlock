use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use rustblock_core::gguf::{align_up, Gguf};
use rustblock_core::{
    convert, ConvertError, ConvertOptions, VerifyMode, TY_PQ2_0, TY_PTQ1_0, TY_Q2_0,
};

const PQ2_BLOCK: usize = 34;
const Q2_BLOCK: usize = 18;

struct TestTensor {
    name: &'static str,
    dims: Vec<u64>,
    ty: u32,
    data: Vec<u8>,
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
    push_u32(buf, 8); // GGUF_METADATA_VALUE_TYPE_STRING
    push_str(buf, value);
}

fn kv_u32(buf: &mut Vec<u8>, key: &str, value: u32) {
    push_str(buf, key);
    push_u32(buf, 4); // GGUF_METADATA_VALUE_TYPE_UINT32
    push_u32(buf, value);
}

fn build_gguf(path: &Path, tensors: &[TestTensor], alignment: u32, file_type: Option<u32>) {
    let mut header = Vec::new();
    header.extend_from_slice(b"GGUF");
    push_u32(&mut header, 3);
    push_u64(&mut header, tensors.len() as u64);

    let kv_count = 2 + file_type.is_some() as u64 + 1; // architecture, alignment, optional file_type, hadamard
    push_u64(&mut header, kv_count);

    kv_str(&mut header, "general.architecture", "test");
    kv_u32(&mut header, "general.alignment", alignment);
    if let Some(ft) = file_type {
        kv_u32(&mut header, "general.file_type", ft);
    }
    kv_u32(&mut header, "prism.hadamard.version", 1);

    let align = alignment as u64;
    let mut offsets = Vec::new();
    let mut cursor = 0u64;
    for t in tensors {
        let off = align_up(cursor, align).unwrap();
        offsets.push(off);
        cursor = off + t.data.len() as u64;
    }

    for (t, off) in tensors.iter().zip(&offsets) {
        push_str(&mut header, t.name);
        push_u32(&mut header, t.dims.len() as u32);
        for d in &t.dims {
            push_u64(&mut header, *d);
        }
        push_u32(&mut header, t.ty);
        push_u64(&mut header, *off);
    }

    let pad = ((align - header.len() as u64 % align) % align) as usize;
    header.extend(std::iter::repeat_n(0u8, pad));

    let mut file = fs::File::create(path).unwrap();
    file.write_all(&header).unwrap();

    let mut written = 0u64;
    for (t, off) in tensors.iter().zip(&offsets) {
        assert!(*off >= written);
        file.write_all(&vec![0u8; (*off - written) as usize])
            .unwrap();
        file.write_all(&t.data).unwrap();
        written = *off + t.data.len() as u64;
    }
    file.flush().unwrap();
}

fn pq2_data() -> Vec<u8> {
    let mut data = Vec::new();
    for block in 0..2 {
        // d = 0.5 (f16 bits 0x3800)
        data.extend_from_slice(&0x3800u16.to_le_bytes());
        for i in 0..32usize {
            data.push((block * 37 + i * 11) as u8);
        }
    }
    data
}

fn f32_data() -> Vec<u8> {
    let values = [1.0f32, -2.0, 3.5, 0.0, 0.25, -0.125, 7.0, 42.0];
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn decode_codes(src: &[u8], count: usize, out: &mut [f32]) {
    let d = half::f16::from_bits(u16::from_le_bytes([src[0], src[1]])).to_f32();
    for j in 0..count {
        let byte = src[2 + j / 4];
        let q = (byte >> ((j % 4) * 2)) & 0b11;
        out[j] = (q as i32 - 1) as f32 * d;
    }
}

#[test]
fn end_to_end_is_lossless_and_preserves_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in.gguf");
    let output = dir.path().join("out.gguf");

    let pq2 = pq2_data();
    let f32_bytes = f32_data();
    let tensors = vec![
        TestTensor {
            name: "ternary.weight",
            dims: vec![256],
            ty: TY_PQ2_0,
            data: pq2.clone(),
        },
        TestTensor {
            name: "norm.weight",
            dims: vec![8],
            ty: 0, // F32
            data: f32_bytes.clone(),
        },
    ];
    build_gguf(&input, &tensors, 32, Some(141));
    let source_before = fs::read(&input).unwrap();

    let report = convert(
        &input,
        &output,
        ConvertOptions {
            verify: VerifyMode::Full,
            ..Default::default()
        },
        None,
    )
    .unwrap();

    assert_eq!(report.tensors_total, 2);
    assert_eq!(report.tensors_pq2, 1);
    assert_eq!(report.new_file_size, fs::metadata(&output).unwrap().len());
    assert_eq!(fs::read(&input).unwrap(), source_before);

    let g = Gguf::open(&output).unwrap();
    assert_eq!(g.alignment, 32);
    assert_eq!(g.file_type, Some(41));
    assert!(g.has_hadamard_version);
    assert_eq!(g.tensors[0].ty, TY_Q2_0);
    assert_eq!(g.tensors[0].dims, vec![256]);
    assert_eq!(g.tensors[0].size, Some(4 * Q2_BLOCK as u64));
    assert_eq!(g.tensors[1].ty, 0);
    assert_eq!(g.tensors[1].offset, 96); // 4 Q2 blocks = 72 bytes, aligned to 32
    assert_eq!(g.tensors[1].dims, vec![8]);

    // Compare every decoded element with the source layout.
    let mut file = fs::File::open(&output).unwrap();
    file.seek(SeekFrom::Start(g.data_start + g.tensors[0].offset))
        .unwrap();
    let mut out_data = vec![0u8; g.tensors[0].size.unwrap() as usize];
    file.read_exact(&mut out_data).unwrap();

    for block in 0..4usize {
        let mut got = [0f32; 64];
        decode_codes(
            &out_data[block * Q2_BLOCK..(block + 1) * Q2_BLOCK],
            64,
            &mut got,
        );

        let src_block = block / 2; // two Q2 blocks per PQ2 block
        let mut expected = [0f32; 128];
        decode_codes(
            &pq2[src_block * PQ2_BLOCK..(src_block + 1) * PQ2_BLOCK],
            128,
            &mut expected,
        );
        let base = (block % 2) * 64;
        for j in 0..64 {
            assert_eq!(got[j].to_bits(), expected[base + j].to_bits());
        }
    }

    file.seek(SeekFrom::Start(g.data_start + g.tensors[1].offset))
        .unwrap();
    let mut raw = vec![0u8; 32];
    file.read_exact(&mut raw).unwrap();
    assert_eq!(raw, f32_bytes);
}

#[test]
fn refuses_ptq1() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in.gguf");
    let tensors = vec![TestTensor {
        name: "w",
        dims: vec![128],
        ty: TY_PTQ1_0,
        data: vec![0u8; 28],
    }];
    build_gguf(&input, &tensors, 32, Some(141));

    let err = convert(
        &input,
        &dir.path().join("out.gguf"),
        ConvertOptions::default(),
        None,
    )
    .unwrap_err();
    assert!(matches!(err, ConvertError::Ptq1), "got {err:?}");
}

#[test]
fn refuses_when_no_pq2_tensors() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in.gguf");
    let tensors = vec![TestTensor {
        name: "w",
        dims: vec![8],
        ty: 0,
        data: f32_data(),
    }];
    build_gguf(&input, &tensors, 32, Some(0));

    let err = convert(
        &input,
        &dir.path().join("out.gguf"),
        ConvertOptions::default(),
        None,
    )
    .unwrap_err();
    assert!(matches!(err, ConvertError::NoPq2), "got {err:?}");
}

#[test]
fn dry_run_writes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in.gguf");
    let output = dir.path().join("out.gguf");
    let tensors = vec![TestTensor {
        name: "w",
        dims: vec![256],
        ty: TY_PQ2_0,
        data: pq2_data(),
    }];
    build_gguf(&input, &tensors, 32, Some(141));

    let report = convert(
        &input,
        &output,
        ConvertOptions {
            dry_run: true,
            verify: VerifyMode::Full,
            ..Default::default()
        },
        None,
    )
    .unwrap();

    assert!(report.dry_run);
    assert!(!output.exists());
    assert!(report.new_file_size > report.old_file_size);
}

#[test]
fn honours_custom_alignment() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in.gguf");
    let output = dir.path().join("out.gguf");
    let tensors = vec![
        TestTensor {
            name: "w",
            dims: vec![256],
            ty: TY_PQ2_0,
            data: pq2_data(),
        },
        TestTensor {
            name: "b",
            dims: vec![8],
            ty: 0,
            data: f32_data(),
        },
    ];
    build_gguf(&input, &tensors, 64, Some(141));

    convert(
        &input,
        &output,
        ConvertOptions {
            verify: VerifyMode::Structure,
            ..Default::default()
        },
        None,
    )
    .unwrap();

    let g = Gguf::open(&output).unwrap();
    assert_eq!(g.alignment, 64);
    for t in &g.tensors {
        assert_eq!(t.offset % 64, 0);
    }
}

#[test]
fn refuses_to_overwrite_without_force() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in.gguf");
    let output = dir.path().join("out.gguf");
    let tensors = vec![TestTensor {
        name: "w",
        dims: vec![256],
        ty: TY_PQ2_0,
        data: pq2_data(),
    }];
    build_gguf(&input, &tensors, 32, Some(141));
    fs::write(&output, b"existing").unwrap();

    let err = convert(&input, &output, ConvertOptions::default(), None).unwrap_err();
    assert!(matches!(err, ConvertError::OutputExists(_)), "got {err:?}");
    assert_eq!(fs::read(&output).unwrap(), b"existing");
}
