use std::fs::File;
use std::path::Path;

use pmetal_gguf::imatrix::IMatrix;
use pmetal_gguf::{GgufBuilder, MetadataValue};
use rustblock_core::{
    compare_quality, diff, estimate_fit, merge, plan_budget, render_chat_template, sha256_file,
    split, transcode, validate_metadata, BudgetRequest, ConvertOptions, FitRequest, Hardware,
    ImportanceMatrix, Target, VerifyMode,
};

fn build_model(path: &Path) {
    let mut builder = GgufBuilder::new();
    builder.add_metadata(
        "general.architecture",
        MetadataValue::String("llama".to_string()),
    );
    builder.add_metadata("llama.block_count", MetadataValue::Uint32(2));
    builder.add_metadata("llama.embedding_length", MetadataValue::Uint32(64));
    builder.add_metadata("llama.attention.head_count", MetadataValue::Uint32(4));
    builder.add_metadata("llama.attention.head_count_kv", MetadataValue::Uint32(2));
    builder.add_metadata("llama.context_length", MetadataValue::Uint32(2048));
    builder.add_f32_tensor(
        "a.weight",
        vec![8],
        vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0],
    );
    builder.add_f32_tensor(
        "b.weight",
        vec![8],
        vec![0.5, 0.25, -0.25, 0.75, -0.75, 1.5, -1.5, 2.5],
    );
    let mut file = File::create(path).unwrap();
    builder.write(&mut file).unwrap();
    file.sync_all().unwrap();
}

fn build_metadata_model(path: &Path, extra: &[(&str, MetadataValue)]) {
    let mut builder = GgufBuilder::new();
    builder.add_metadata(
        "general.architecture",
        MetadataValue::String("llama".to_string()),
    );
    builder.add_metadata("llama.block_count", MetadataValue::Uint32(2));
    builder.add_metadata("llama.embedding_length", MetadataValue::Uint32(64));
    builder.add_metadata("llama.attention.head_count", MetadataValue::Uint32(4));
    builder.add_metadata("llama.attention.head_count_kv", MetadataValue::Uint32(2));
    for (key, value) in extra {
        builder.add_metadata(*key, value.clone());
    }
    builder.add_f32_tensor("a.weight", vec![8], vec![1.0; 8]);
    let mut file = File::create(path).unwrap();
    builder.write(&mut file).unwrap();
    file.sync_all().unwrap();
}

#[test]
fn quality_fit_and_ops_work_on_a_real_gguf() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("source.gguf");
    let output = dir.path().join("output.gguf");
    build_model(&source);

    transcode(
        &source,
        &output,
        Target::F16,
        ConvertOptions {
            force: true,
            dry_run: false,
            verify: VerifyMode::Structure,
            allow_requantize: true,
        },
        None,
    )
    .unwrap();

    let report = compare_quality(&source, &output, None).unwrap();
    assert_eq!(report.tensors.len(), 2);
    assert!(report.mean_snr_db > 60.0);
    assert!(report.mean_cosine > 0.9999);

    let fit = estimate_fit(
        &output,
        FitRequest {
            context: Some(128),
            overhead_bytes: 0,
            ..FitRequest::default()
        },
    )
    .unwrap();
    assert!(fit.weights_bytes > 0);
    assert!(fit.kv_cache_bytes > 0);

    let shards = split(&output, &dir.path().join("shards"), 16, true).unwrap();
    assert!(shards.len() >= 2);
    let merged = dir.path().join("merged.gguf");
    merge(&shards, &merged, true, false).unwrap();
    let report = diff(&output, &merged).unwrap();
    assert!(report.tensor_changes.is_empty());
    assert!(report.tensors_only_in_a.is_empty());
    assert!(report.tensors_only_in_b.is_empty());
    assert_eq!(sha256_file(&output).unwrap(), sha256_file(&merged).unwrap());

    let plan = plan_budget(
        &output,
        BudgetRequest {
            budget_bytes: Some(1 << 40),
            context: 128,
            overhead_bytes: 0,
            hardware: None,
        },
    )
    .unwrap();
    assert_eq!(plan.candidates.len(), 12);
    assert_eq!(plan.selected, Target::F16);
    assert!(plan.selected_total_bytes > 0);

    let hardware_plan = plan_budget(
        &output,
        BudgetRequest {
            budget_bytes: None,
            context: 128,
            overhead_bytes: 0,
            hardware: Some(Hardware {
                ram_bytes: Some(1 << 40),
                vram_bytes: None,
            }),
        },
    )
    .unwrap();
    assert_eq!(hardware_plan.selected, Target::F16);

    let hardware_small = plan_budget(
        &output,
        BudgetRequest {
            budget_bytes: None,
            context: 128,
            overhead_bytes: 0,
            hardware: Some(Hardware {
                ram_bytes: Some(1),
                vram_bytes: None,
            }),
        },
    )
    .unwrap();
    let hardware_smallest = hardware_small
        .candidates
        .iter()
        .min_by_key(|candidate| candidate.total_bytes)
        .unwrap();
    assert_eq!(hardware_small.selected, hardware_smallest.target);

    let tight = plan_budget(
        &output,
        BudgetRequest {
            budget_bytes: Some(1),
            context: 128,
            overhead_bytes: 0,
            hardware: None,
        },
    )
    .unwrap();
    let smallest = tight
        .candidates
        .iter()
        .min_by_key(|candidate| candidate.total_bytes)
        .unwrap();
    assert_eq!(tight.selected, smallest.target);
    assert_eq!(tight.selected_total_bytes, smallest.total_bytes);
}

#[test]
fn fit_handles_sliding_mla_and_ssm_metadata() {
    let dir = tempfile::tempdir().unwrap();
    let sliding = dir.path().join("sliding.gguf");
    build_metadata_model(
        &sliding,
        &[
            (
                "llama.attention.head_count_kv",
                MetadataValue::Array(vec![MetadataValue::Uint32(2), MetadataValue::Uint32(2)]),
            ),
            (
                "llama.attention.key_length",
                MetadataValue::Array(vec![MetadataValue::Uint32(8), MetadataValue::Uint32(8)]),
            ),
            (
                "llama.attention.value_length",
                MetadataValue::Array(vec![MetadataValue::Uint32(8), MetadataValue::Uint32(8)]),
            ),
            ("llama.attention.sliding_window", MetadataValue::Uint32(32)),
            (
                "llama.attention.sliding_window_pattern",
                MetadataValue::Array(vec![MetadataValue::Uint32(1), MetadataValue::Uint32(0)]),
            ),
        ],
    );
    let fit = estimate_fit(
        &sliding,
        FitRequest {
            context: Some(128),
            overhead_bytes: 0,
            ..FitRequest::default()
        },
    )
    .unwrap();
    assert!(fit.dense_kv_bytes > 0);
    assert!(fit.sliding_kv_bytes > 0);
    assert!(fit.sliding_kv_bytes < fit.dense_kv_bytes);

    let mla = dir.path().join("mla.gguf");
    build_metadata_model(
        &mla,
        &[
            ("llama.attention.kv_lora_rank", MetadataValue::Uint32(16)),
            ("llama.attention.qk_rope_head_dim", MetadataValue::Uint32(8)),
        ],
    );
    let fit = estimate_fit(
        &mla,
        FitRequest {
            context: Some(128),
            overhead_bytes: 0,
            ..FitRequest::default()
        },
    )
    .unwrap();
    assert!(fit.mla_kv_bytes > 0);
    assert_eq!(fit.dense_kv_bytes, 0);
    assert_eq!(fit.sliding_kv_bytes, 0);

    let ssm = dir.path().join("ssm.gguf");
    build_metadata_model(
        &ssm,
        &[
            ("llama.ssm.inner_size", MetadataValue::Uint32(64)),
            ("llama.ssm.state_size", MetadataValue::Uint32(16)),
            ("llama.ssm.conv_kernel", MetadataValue::Uint32(4)),
            ("llama.ssm.time_step_rank", MetadataValue::Uint32(8)),
        ],
    );
    let fit = estimate_fit(
        &ssm,
        FitRequest {
            context: Some(128),
            overhead_bytes: 0,
            ..FitRequest::default()
        },
    )
    .unwrap();
    assert!(fit.ssm_state_bytes > 0);
}

#[test]
fn legacy_imatrix_selects_high_importance_tensors() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.imatrix");
    let mut raw = IMatrix::new();
    raw.insert("a.weight".to_string(), vec![10.0; 8], 1);
    raw.insert("b.weight".to_string(), vec![1.0; 8], 1);
    raw.save(&path).unwrap();

    let matrix = ImportanceMatrix::load(&path).unwrap();
    let gguf = rustblock_core::Gguf::open(&dir.path().join("missing.gguf"));
    assert!(gguf.is_err());

    let tensors = [
        rustblock_core::TensorInfo {
            name: "a.weight".to_string(),
            dims: vec![8],
            ty: 0,
            offset: 0,
            type_pos: 0,
            offset_pos: 0,
            size: Some(32),
        },
        rustblock_core::TensorInfo {
            name: "b.weight".to_string(),
            dims: vec![8],
            ty: 0,
            offset: 32,
            type_pos: 0,
            offset_pos: 0,
            size: Some(32),
        },
    ];
    let overrides = matrix.type_overrides(&tensors, Target::Q2K, 0.5).unwrap();
    assert_eq!(overrides.len(), 1);
}

#[test]
fn ollama_handoff_writes_a_modelfile_without_running_anything() {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("model.gguf");
    std::fs::write(&model, b"placeholder").unwrap();
    let report = rustblock_core::install_ollama(
        &model,
        "rustblock-test",
        &dir.path().join("runtime"),
        false,
    )
    .unwrap();
    let text = std::fs::read_to_string(report.instructions).unwrap();
    assert!(text.contains("FROM"));
    assert!(text.contains("model.gguf"));
}

#[test]
fn metadata_validation_accepts_a_balanced_template_and_valid_rope() {
    let dir = tempfile::tempdir().unwrap();
    let valid = dir.path().join("valid.gguf");
    build_metadata_model(
        &valid,
        &[
            (
                "tokenizer.chat_template",
                MetadataValue::String(
                    "{% for message in messages %}{% if message['role'] == 'user' %}{{ message['content'] }}{% endif %}{% endfor %}"
                        .to_string(),
                ),
            ),
            ("llama.rope.freq_base", MetadataValue::Uint32(10_000)),
            ("llama.rope.scaling.type", MetadataValue::String("linear".into())),
            ("llama.rope.scaling.factor", MetadataValue::Float32(2.0)),
            ("llama.rope.dimension_count", MetadataValue::Uint32(16)),
        ],
    );
    let report = validate_metadata(&valid).unwrap();
    assert!(report.is_ok(), "unexpected issues: {:?}", report.issues);
}

#[test]
fn metadata_validation_reports_bad_template_and_rope() {
    let dir = tempfile::tempdir().unwrap();
    let invalid = dir.path().join("invalid.gguf");
    build_metadata_model(
        &invalid,
        &[
            (
                "tokenizer.chat_template",
                MetadataValue::String("{% if message %}{{ message['content'] }}".to_string()),
            ),
            ("llama.rope.freq_base", MetadataValue::Uint32(0)),
            (
                "llama.rope.scaling.type",
                MetadataValue::String("unknown".into()),
            ),
            ("llama.rope.scaling.factor", MetadataValue::Float32(0.0)),
        ],
    );
    let report = validate_metadata(&invalid).unwrap();
    assert!(!report.is_ok());
    assert!(report
        .issues
        .iter()
        .any(|issue| issue.key == "tokenizer.chat_template"));
    assert!(report
        .issues
        .iter()
        .any(|issue| issue.key == "llama.rope.freq_base"));
    assert!(report
        .issues
        .iter()
        .any(|issue| issue.key == "llama.rope.scaling.type"));
}

#[test]
fn metadata_validation_compiles_jinja_expressions() {
    let dir = tempfile::tempdir().unwrap();
    let invalid = dir.path().join("invalid-expression.gguf");
    build_metadata_model(
        &invalid,
        &[(
            "tokenizer.chat_template",
            MetadataValue::String("{{ message['content' | }}".into()),
        )],
    );
    let report = validate_metadata(&invalid).unwrap();
    assert!(report
        .issues
        .iter()
        .any(|issue| issue.key == "tokenizer.chat_template"
            && issue.message.contains("Jinja syntax error")));
}

#[test]
fn chat_template_renders_with_a_real_context() {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("chat.gguf");
    build_metadata_model(
        &model,
        &[(
            "tokenizer.chat_template",
            MetadataValue::String(
                "{% for message in messages %}<{{ message.role }}>{{ message.content }}</{{ message.role }}>{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}"
                    .to_string(),
            ),
        )],
    );
    let context = serde_json::json!({
        "messages": [
            { "role": "user", "content": "Ciao" }
        ],
        "add_generation_prompt": true
    });
    let rendered = render_chat_template(&model, &context).unwrap();
    assert_eq!(rendered, "<user>Ciao</user><assistant>");
}
