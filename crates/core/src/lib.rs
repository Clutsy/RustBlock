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

pub mod convert;
pub mod gguf;
pub mod transcode;
pub mod types;

pub use convert::{
    convert, ConvertError, ConvertOptions, Phase, Progress, Report, VerifyMode,
    LLAMA_FTYPE_MOSTLY_Q2_0,
};
pub use gguf::{align_up, Gguf, GgufError, TensorInfo};
pub use transcode::{conversion_workers, source_supported, transcode, Target};
pub use types::{type_info, type_name, TypeInfo, TY_PQ2_0, TY_PTQ1_0, TY_Q2_0};
