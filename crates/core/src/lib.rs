//! Lossless GGUF reblocker for Prism's private `PQ2_0` tensor type.
//!
//! Prism ML publishes Ternary Bonsai GGUFs in two layouts:
//!
//! * `PQ2_0` (type 142): 128-element blocks, 34 bytes per block (fork-only).
//! * `Q2_0`  (type 42):   64-element blocks, 18 bytes per block (upstream).
//!
//! Both use the same 2-bit ternary codec, so `PQ2_0` can be rewritten as
//! `Q2_0` by splitting each 128-element block into two 64-element blocks that
//! share the original scale. No dequantization or requantization is involved.

pub mod convert;
pub mod gguf;
pub mod types;

pub use convert::{
    convert, ConvertError, ConvertOptions, Progress, Report, VerifyMode, LLAMA_FTYPE_MOSTLY_Q2_0,
};
pub use gguf::{align_up, Gguf, GgufError, TensorInfo};
pub use types::{type_info, type_name, TypeInfo, TY_PQ2_0, TY_PTQ1_0, TY_Q2_0};
