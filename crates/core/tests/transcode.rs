use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use half::f16;
use rustblock_core::gguf::Gguf;
use rustblock_core::{
    effective_tensor_types, transcode, ConvertError, ConvertOptions, Target, TensorOverride,
    VerifyMode, TY_PQ2_0, TY_Q2_0,
};

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

fn build_gguf(
    path: &Path,
    name: &str,
    dims: &[u64],
    ty: u32,
    data: &[u8],
    alignment: u32,
    file_type: u32,
) {
    let mut header = Vec::new();
    header.extend_from_slice(b"GGUF");
    push_u32(&mut header, 3);
    push_u64(&mut header, 1);
    push_u64(&mut header, 3);
    kv_str(&mut header, "general.architecture", "test");
    kv_u32(&mut header, "general.alignment", alignment);
    kv_u32(&mut header, "general.file_type", file_type);

    push_str(&mut header, name);
    push_u32(&mut header, dims.len() as u32);
    for d in dims {
        push_u64(&mut header, *d);
    }
    push_u32(&mut header, ty);
    push_u64(&mut header, 0);

    let align = alignment as u64;
    let pad = ((align - header.len() as u64 % align) % align) as usize;
    header.extend(std::iter::repeat_n(0u8, pad));

    let mut file = fs::File::create(path).unwrap();
    file.write_all(&header).unwrap();
    file.write_all(data).unwrap();
    file.flush().unwrap();
}

struct TensorSpec<'a> {
    name: &'a str,
    dims: &'a [u64],
    ty: u32,
    data: &'a [u8],
}

fn build_multi_gguf(path: &Path, tensors: &[TensorSpec<'_>], alignment: u32, file_type: u32) {
    let align = alignment as u64;
    let mut offsets = Vec::with_capacity(tensors.len());
    let mut cursor = 0u64;
    for tensor in tensors {
        cursor = cursor.div_ceil(align) * align;
        offsets.push(cursor);
        cursor += tensor.data.len() as u64;
    }

    let mut header = Vec::new();
    header.extend_from_slice(b"GGUF");
    push_u32(&mut header, 3);
    push_u64(&mut header, tensors.len() as u64);
    push_u64(&mut header, 3);
    kv_str(&mut header, "general.architecture", "test");
    kv_u32(&mut header, "general.alignment", alignment);
    kv_u32(&mut header, "general.file_type", file_type);
    for (tensor, offset) in tensors.iter().zip(offsets.iter()) {
        push_str(&mut header, tensor.name);
        push_u32(&mut header, tensor.dims.len() as u32);
        for dim in tensor.dims {
            push_u64(&mut header, *dim);
        }
        push_u32(&mut header, tensor.ty);
        push_u64(&mut header, *offset);
    }
    let data_start = (header.len() as u64).div_ceil(align) * align;
    header.resize(data_start as usize, 0);
    header.resize(data_start as usize + cursor as usize, 0);
    for (tensor, offset) in tensors.iter().zip(offsets.iter()) {
        let start = data_start as usize + *offset as usize;
        header[start..start + tensor.data.len()].copy_from_slice(tensor.data);
    }
    fs::write(path, header).unwrap();
}

fn float_values(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 * 0.017).sin() * 2.5) + (i as f32 * 0.001))
        .collect()
}

fn f16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| f16::from_f32(*v).to_bits().to_le_bytes())
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn pq2_bytes() -> Vec<u8> {
    let mut data = Vec::new();
    for block in 0..2 {
        data.extend_from_slice(&0x3800u16.to_le_bytes());
        for i in 0..32 {
            data.push((block * 41 + i * 13) as u8);
        }
    }
    data
}

#[test]
fn f32_to_f16_is_exact_at_half_precision() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in-f32.gguf");
    let output = dir.path().join("out-f16.gguf");
    let values = float_values(512);
    build_gguf(&input, "w", &[256, 2], 0, &f32_bytes(&values), 32, 0);
    let source_before = fs::read(&input).unwrap();

    let report = transcode(
        &input,
        &output,
        Target::F16,
        ConvertOptions {
            verify: VerifyMode::Structure,
            ..Default::default()
        },
        None,
    )
    .unwrap();

    assert_eq!(report.new_file_size, fs::metadata(&output).unwrap().len());
    assert_eq!(fs::read(&input).unwrap(), source_before);

    let g = Gguf::open(&output).unwrap();
    assert_eq!(g.tensors[0].ty, 1);
    assert_eq!(g.tensors[0].size, Some(1024));
    assert_eq!(g.file_type, Some(1));

    let mut file = fs::File::open(&output).unwrap();
    file.seek(SeekFrom::Start(g.data_start)).unwrap();
    let mut raw = vec![0u8; 1024];
    file.read_exact(&mut raw).unwrap();
    for (i, value) in values.iter().enumerate() {
        let got = f16::from_bits(u16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]]));
        assert_eq!(got.to_bits(), f16::from_f32(*value).to_bits());
    }
}

#[test]
fn q4k_quantization_adds_quantization_version() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in-f16.gguf");
    let output = dir.path().join("out-q4k.gguf");
    let values = float_values(512);
    build_gguf(&input, "w", &[256, 2], 1, &f16_bytes(&values), 32, 1);
    let source_before = fs::read(&input).unwrap();

    let report = transcode(
        &input,
        &output,
        Target::Q4K,
        ConvertOptions {
            verify: VerifyMode::Structure,
            ..Default::default()
        },
        None,
    )
    .unwrap();

    assert_eq!(report.new_file_size, fs::metadata(&output).unwrap().len());
    assert_eq!(fs::read(&input).unwrap(), source_before);

    let g = Gguf::open(&output).unwrap();
    assert_eq!(g.tensors[0].ty, 12);
    assert_eq!(g.tensors[0].size, Some(288));
    assert_eq!(g.file_type, Some(14));
    assert!(
        g.kv.iter()
            .any(|kv| kv.key == "general.quantization_version"),
        "quantization_version metadata must be present"
    );
}

#[test]
fn sensitive_tensors_get_extra_precision_on_every_model() {
    let dir = tempfile::tempdir().unwrap();
    let values = float_values(512);
    for name in [
        "output.weight",
        "token_embd.weight",
        "blk.0.attn_v.weight",
        "transformer.word_embeddings.weight",
    ] {
        let input = dir.path().join(format!("{}.gguf", name.replace('.', "_")));
        let output = dir
            .path()
            .join(format!("{}-q4k.gguf", name.replace('.', "_")));
        build_gguf(&input, name, &[256, 2], 1, &f16_bytes(&values), 32, 1);

        transcode(
            &input,
            &output,
            Target::Q4K,
            ConvertOptions {
                verify: VerifyMode::Full,
                ..Default::default()
            },
            None,
        )
        .unwrap_or_else(|e| panic!("{name} failed: {e}"));

        let g = Gguf::open(&output).unwrap();
        assert_eq!(g.tensors[0].ty, 14, "{name} should be protected as Q6_K");
        assert_eq!(g.file_type, Some(14));
    }
}

#[test]
fn ternary_targets_are_not_mixed_with_protected_precision() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("output.gguf");
    let output = dir.path().join("output-tq2.gguf");
    let values = float_values(512);
    build_gguf(
        &input,
        "output.weight",
        &[256, 2],
        1,
        &f16_bytes(&values),
        32,
        1,
    );

    transcode(
        &input,
        &output,
        Target::Tq2_0,
        ConvertOptions {
            verify: VerifyMode::Full,
            ..Default::default()
        },
        None,
    )
    .unwrap();

    let g = Gguf::open(&output).unwrap();
    assert_eq!(g.tensors[0].ty, 35);
}

#[test]
fn k_quant_targets_keep_their_protected_layout() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("output.gguf");
    let values = float_values(512);
    build_gguf(
        &input,
        "output.weight",
        &[256, 2],
        1,
        &f16_bytes(&values),
        32,
        1,
    );
    for (target, expected) in [
        (Target::Q2K, 14),
        (Target::Q3K, 13),
        (Target::Q4K, 14),
        (Target::Q5K, 14),
    ] {
        let output = dir
            .path()
            .join(format!("protected-{}.gguf", target.short()));
        transcode(
            &input,
            &output,
            target,
            ConvertOptions {
                verify: VerifyMode::Full,
                ..Default::default()
            },
            None,
        )
        .unwrap_or_else(|e| panic!("{target:?} protected conversion failed: {e}"));
        let g = Gguf::open(&output).unwrap();
        assert_eq!(g.tensors[0].ty, expected, "{target:?}");
    }
}

#[test]
fn q8_0_conversion_writes_blocks() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in-f32.gguf");
    let output = dir.path().join("out-q8.gguf");
    let values = float_values(128);
    build_gguf(&input, "w", &[32, 4], 0, &f32_bytes(&values), 32, 0);

    transcode(
        &input,
        &output,
        Target::Q8_0,
        ConvertOptions {
            verify: VerifyMode::Structure,
            ..Default::default()
        },
        None,
    )
    .unwrap();

    let g = Gguf::open(&output).unwrap();
    assert_eq!(g.tensors[0].ty, 8);
    assert_eq!(g.tensors[0].size, Some(136));
    assert_eq!(g.file_type, Some(7));
    assert!(g
        .kv
        .iter()
        .any(|kv| kv.key == "general.quantization_version"));
}

#[test]
fn reblock_through_transcode_is_lossless() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in-pq2.gguf");
    let output = dir.path().join("out-q2.gguf");
    let data = pq2_bytes();
    build_gguf(&input, "w", &[256], TY_PQ2_0, &data, 32, 141);
    let source_before = fs::read(&input).unwrap();

    transcode(
        &input,
        &output,
        Target::Reblock,
        ConvertOptions {
            verify: VerifyMode::Full,
            ..Default::default()
        },
        None,
    )
    .unwrap();

    assert_eq!(fs::read(&input).unwrap(), source_before);
    let g = Gguf::open(&output).unwrap();
    assert_eq!(g.tensors[0].ty, TY_Q2_0);
    assert_eq!(g.tensors[0].size, Some(72));
    assert_eq!(g.file_type, Some(41));
}

#[test]
fn alignment_is_preserved_by_quantized_output() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in.gguf");
    let output = dir.path().join("out.gguf");
    let values = float_values(512);
    build_gguf(&input, "w", &[256, 2], 1, &f16_bytes(&values), 64, 1);

    transcode(
        &input,
        &output,
        Target::Q5K,
        ConvertOptions {
            verify: VerifyMode::Structure,
            ..Default::default()
        },
        None,
    )
    .unwrap();

    let g = Gguf::open(&output).unwrap();
    assert_eq!(g.alignment, 64);
    assert_eq!(g.tensors[0].offset % 64, 0);
    assert_eq!(g.tensors[0].ty, 13);
    assert_eq!(g.tensors[0].size, Some(352));
    assert_eq!(g.file_type, Some(16));
}

#[test]
fn source_is_never_modified_on_error() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in.gguf");
    let output = dir.path().join("out.gguf");
    let values = float_values(256);
    build_gguf(&input, "w", &[256, 1], 1, &f16_bytes(&values), 32, 1);
    let source_before = fs::read(&input).unwrap();
    fs::write(&output, b"existing").unwrap();

    let error = transcode(
        &input,
        &output,
        Target::Q4K,
        ConvertOptions::default(),
        None,
    )
    .unwrap_err();

    assert!(matches!(
        error,
        rustblock_core::ConvertError::OutputExists(_)
    ));
    assert_eq!(fs::read(&input).unwrap(), source_before);
    assert_eq!(fs::read(&output).unwrap(), b"existing");
}

#[test]
fn every_native_target_produces_a_valid_gguf() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in-f32.gguf");
    let values = float_values(512);
    build_gguf(&input, "w", &[256, 2], 0, &f32_bytes(&values), 32, 0);
    let source_before = fs::read(&input).unwrap();

    for target in [
        Target::F32,
        Target::F16,
        Target::Bf16,
        Target::Q8_0,
        Target::Q5_0,
        Target::Q5_1,
        Target::Q4_0,
        Target::Q4_1,
        Target::Q2K,
        Target::Q3K,
        Target::Q4K,
        Target::Q5K,
        Target::Q6K,
        Target::Tq1_0,
        Target::Tq2_0,
        Target::Mxfp4,
        Target::Nvfp4,
        Target::Q1_0,
    ] {
        let output = dir.path().join(format!("out-{}.gguf", target.short()));
        transcode(
            &input,
            &output,
            target,
            ConvertOptions {
                verify: VerifyMode::Structure,
                ..Default::default()
            },
            None,
        )
        .unwrap_or_else(|e| panic!("{target:?} failed: {e}"));

        let g = Gguf::open(&output).unwrap();
        assert_eq!(g.tensors[0].ty, target.type_id().unwrap(), "{target:?}");
        assert_eq!(g.file_type, target.file_type(), "{target:?}");
        assert!(g.tensors[0].size.unwrap() > 0, "{target:?}");
    }
    assert_eq!(fs::read(&input).unwrap(), source_before);
}

#[test]
fn every_native_target_survives_full_verification() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in-f32.gguf");
    let values = float_values(512);
    build_gguf(&input, "w", &[256, 2], 0, &f32_bytes(&values), 32, 0);

    for target in [
        Target::F32,
        Target::F16,
        Target::Bf16,
        Target::Q8_0,
        Target::Q5_0,
        Target::Q5_1,
        Target::Q4_0,
        Target::Q4_1,
        Target::Q2K,
        Target::Q3K,
        Target::Q4K,
        Target::Q5K,
        Target::Q6K,
        Target::Tq1_0,
        Target::Tq2_0,
        Target::Mxfp4,
        Target::Nvfp4,
        Target::Q1_0,
    ] {
        let output = dir.path().join(format!("full-{}.gguf", target.short()));
        transcode(
            &input,
            &output,
            target,
            ConvertOptions {
                verify: VerifyMode::Full,
                ..Default::default()
            },
            None,
        )
        .unwrap_or_else(|e| panic!("{target:?} failed full verification: {e}"));
    }
}

#[test]
fn unaligned_rows_fall_back_to_float_instead_of_failing() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in-f32.gguf");
    let output = dir.path().join("out-ternary.gguf");
    let values = float_values(51);
    build_gguf(&input, "w", &[17, 3], 0, &f32_bytes(&values), 32, 0);

    transcode(
        &input,
        &output,
        Target::Tq1_0,
        ConvertOptions {
            verify: VerifyMode::Full,
            ..Default::default()
        },
        None,
    )
    .unwrap();

    let g = Gguf::open(&output).unwrap();
    assert_eq!(g.tensors[0].ty, 1);
    assert_eq!(g.tensors[0].size, Some(102));
    assert_eq!(g.file_type, Some(36));
}

#[test]
fn large_tensor_parallel_chunks_verify() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("large-f32.gguf");
    let output = dir.path().join("large-q5k.gguf");
    let values = float_values(4096 * 512);
    build_gguf(&input, "w", &[4096, 512], 0, &f32_bytes(&values), 32, 0);

    transcode(
        &input,
        &output,
        Target::Q5K,
        ConvertOptions {
            verify: VerifyMode::Full,
            ..Default::default()
        },
        None,
    )
    .unwrap();

    let g = Gguf::open(&output).unwrap();
    assert_eq!(g.tensors[0].ty, 13);
    assert_eq!(g.tensors[0].element_count(), 4096 * 512);
}

#[test]
fn mixed_preset_end_to_end_writes_the_recipe_map_and_file_type() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("mixed-in.gguf");
    let output = dir.path().join("mixed-out.gguf");
    let attn_v = f16_bytes(&float_values(256));
    let ffn_down = f16_bytes(&float_values(256));
    let output_weight = f16_bytes(&float_values(256));
    let norm = f32_bytes(&float_values(256));
    let tensors = [
        TensorSpec {
            name: "blk.0.attn_v.weight",
            dims: &[256, 1],
            ty: 1,
            data: &attn_v,
        },
        TensorSpec {
            name: "blk.0.ffn_down.weight",
            dims: &[256, 1],
            ty: 1,
            data: &ffn_down,
        },
        TensorSpec {
            name: "blk.0.attn_norm.weight",
            dims: &[256],
            ty: 0,
            data: &norm,
        },
        TensorSpec {
            name: "output.weight",
            dims: &[256, 1],
            ty: 1,
            data: &output_weight,
        },
    ];
    build_multi_gguf(&input, &tensors, 32, 1);

    let g = Gguf::open(&input).unwrap();
    let map = effective_tensor_types(&g.tensors, Target::Q4KM, &[]);
    assert_eq!(map[0].1, 12, "single attention value should stay Q4_K");
    assert_eq!(map[1].1, 12, "single ffn_down should stay Q4_K");
    assert_eq!(map[2].1, 0, "norm keeps its source type");
    assert_eq!(map[3].1, 14, "output uses Q6_K");

    transcode(
        &input,
        &output,
        Target::Q4KM,
        ConvertOptions {
            verify: VerifyMode::Full,
            ..Default::default()
        },
        None,
    )
    .unwrap();

    let out = Gguf::open(&output).unwrap();
    let ty = |name: &str| {
        out.tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .unwrap()
            .ty
    };
    assert_eq!(ty("blk.0.attn_v.weight"), 12);
    assert_eq!(ty("blk.0.ffn_down.weight"), 12);
    assert_eq!(ty("blk.0.attn_norm.weight"), 0);
    assert_eq!(ty("output.weight"), 14);
    assert_eq!(out.file_type, Some(15));
    assert!(out
        .kv
        .iter()
        .any(|kv| kv.key == "general.quantization_version"));
}

#[test]
fn explicit_override_wins_over_the_mixed_recipe() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("override-in.gguf");
    let output = dir.path().join("override-out.gguf");
    let values = f16_bytes(&float_values(512));
    build_gguf(&input, "blk.0.attn_v.weight", &[256, 2], 1, &values, 32, 1);
    let rule = TensorOverride::parse(r"^blk\.0\.attn_v\.weight$=q4_k").unwrap();

    rustblock_core::transcode_with_overrides(
        &input,
        &output,
        Target::Q4KM,
        ConvertOptions {
            verify: VerifyMode::Full,
            ..Default::default()
        },
        &[rule],
        None,
    )
    .unwrap();

    let out = Gguf::open(&output).unwrap();
    assert_eq!(out.tensors[0].ty, 12);
}

#[test]
fn requantization_requires_an_explicit_opt_in() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("float.gguf");
    let first = dir.path().join("q4.gguf");
    let second = dir.path().join("q5.gguf");
    let values = f16_bytes(&float_values(512));
    build_gguf(&input, "w", &[256, 2], 1, &values, 32, 1);
    transcode(&input, &first, Target::Q4K, ConvertOptions::default(), None).unwrap();

    let error = transcode(
        &first,
        &second,
        Target::Q5K,
        ConvertOptions::default(),
        None,
    )
    .unwrap_err();
    assert!(matches!(error, ConvertError::Requantize { .. }));
    assert!(!second.exists());

    transcode(
        &first,
        &second,
        Target::Q5K,
        ConvertOptions {
            allow_requantize: true,
            ..Default::default()
        },
        None,
    )
    .unwrap();
    assert_eq!(Gguf::open(&second).unwrap().tensors[0].ty, 13);
}
