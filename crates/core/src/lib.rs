//! Native GGUF conversion engine used by RustBlock.
//!
//! It supports lossless reblocking of Prism `PQ2_0` into upstream `Q2_0`,
//! float conversion (`F32`, `F16`, `BF16`), legacy quantization (`Q4_0`,
//! `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`), K-quantization (`Q2_K` through `Q6_K`),
//! ternary (`TQ1_0`, `TQ2_0`), block-float (`MXFP4`, `NVFP4`) and 1-bit
//! (`Q1_0`) targets.
//!
//! `PQ2_0` uses 128-element blocks (34 bytes), while upstream `Q2_0` uses
//! 64-element blocks (18 bytes). Both share the same 2-bit ternary codec, so
//! reblocking splits each block into two blocks with the original scale and
//! never dequantizes or requantizes.

pub mod budget;
pub mod cache;
pub mod convert;
pub mod download;
pub mod fit;
pub mod gguf;
pub mod imatrix;
pub mod install;
pub mod ops;
pub mod quality;
pub mod recipe;
pub mod safetensors;
pub mod safetensors_gguf;
pub mod safetensors_native;
pub mod transcode;
pub mod types;

pub use budget::{plan as plan_budget, BudgetCandidate, BudgetError, BudgetPlan, BudgetRequest};
pub use cache::{cache_index, cache_stats, trim_cache, CacheEntry, CacheStats, CacheTrimReport};
pub use convert::{
    convert, ConvertError, ConvertOptions, Phase, Progress, Report, VerifyMode,
    LLAMA_FTYPE_MOSTLY_Q2_0,
};
pub use download::{
    download_many, download_many_with_events, download_mmproj, download_url,
    download_url_with_progress, BatchDownloadEvent, DownloadError, DownloadOptions, DownloadReport,
};
pub use fit::{
    detect_hardware, estimate as estimate_fit, hardware_fits, human_bytes, suggest_target,
    FitEstimate, FitRequest, Hardware,
};
pub use gguf::{align_up, Gguf, GgufError, TensorInfo};
pub use imatrix::{
    generate_with_llama_imatrix, ImatrixError, ImportanceMatrix, OVERFITTING_WARNING,
};
pub use install::{
    install_ollama, quick_test, write_lmstudio_instructions, InstallError, InstallReport,
    QuickTestReport,
};
pub use ops::{
    diff, merge, metadata_get, metadata_list, metadata_set, prune, render_chat_template,
    sha256_file, split, validate_metadata, verify_hash, DiffReport, MetadataValidation, OpsError,
    ValidationIssue,
};
pub use quality::{
    compare as compare_quality, run_kld, to_csv as quality_csv, KldError, KldReport, QualityReport,
    TensorQuality,
};
pub use recipe::{
    analyze_tensors, effective_tensor_types, exact_name_pattern, parse_tensor_type, select_type,
    tensor_can_be_quantized, use_more_bits, RecipeContext, RecipeError, TensorCategory, TensorMeta,
    TensorOverride,
};
pub use safetensors_gguf::{
    convert_llama_safetensors, ToGgufError as SafetensorsToGgufError,
    ToGgufReport as SafetensorsToGgufReport,
};
pub use safetensors_native::{
    dequantize_file as dequantize_safetensors_file, inspect_file as inspect_safetensors,
    NativeError as SafetensorsNativeError, NativeReport as SafetensorsNativeReport,
};
pub use transcode::{
    conversion_workers, source_supported, transcode, transcode_with_imatrix,
    transcode_with_overrides, Target,
};
pub use types::{type_info, type_name, TypeInfo, TY_PQ2_0, TY_PTQ1_0, TY_Q2_0};
