//! GGML tensor type table (the subset needed to parse and validate GGUF files).
//!
//! A type is identified by the numeric id stored in GGUF tensor-info records.
//! `block` is the number of elements per block; `bytes` is the on-disk size of
//! one block. Types marked `None` are recognized by name but cannot be sized
//! (Prism PTQ1_0) or are unknown to this tool.

pub const TY_Q1_0: u32 = 41;
pub const TY_Q2_0: u32 = 42;
pub const TY_PQ2_0: u32 = 142;
pub const TY_PTQ1_0: u32 = 143;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TypeInfo {
    pub name: &'static str,
    pub block: u64,
    pub bytes: u64,
}

/// Returns the block layout for a tensor type, or `None` if unknown/unsized.
pub fn type_info(ty: u32) -> Option<TypeInfo> {
    use TypeInfo as T;
    let (name, block, bytes) = match ty {
        0 => ("F32", 1, 4),
        1 => ("F16", 1, 2),
        2 => ("Q4_0", 32, 18),
        3 => ("Q4_1", 32, 20),
        6 => ("Q5_0", 32, 22),
        7 => ("Q5_1", 32, 24),
        8 => ("Q8_0", 32, 34),
        9 => ("Q8_1", 32, 36),
        10 => ("Q2_K", 256, 84),
        11 => ("Q3_K", 256, 110),
        12 => ("Q4_K", 256, 144),
        13 => ("Q5_K", 256, 176),
        14 => ("Q6_K", 256, 210),
        15 => ("Q8_K", 256, 292),
        16 => ("IQ2_XXS", 256, 66),
        17 => ("IQ2_XS", 256, 74),
        18 => ("IQ3_XXS", 256, 98),
        19 => ("IQ1_S", 256, 50),
        20 => ("IQ4_NL", 32, 18),
        21 => ("IQ3_S", 256, 110),
        22 => ("IQ2_S", 256, 82),
        23 => ("IQ4_XS", 256, 136),
        24 => ("I8", 1, 1),
        25 => ("I16", 1, 2),
        26 => ("I32", 1, 4),
        27 => ("I64", 1, 8),
        28 => ("F64", 1, 8),
        29 => ("IQ1_M", 256, 56),
        30 => ("BF16", 1, 2),
        31 => ("Q4_0_4_4", 32, 18),
        32 => ("Q4_0_4_8", 32, 18),
        33 => ("Q4_0_8_8", 32, 18),
        34 => ("TQ1_0", 256, 54),
        35 => ("TQ2_0", 256, 66),
        36 => ("IQ4_NL_4_4", 32, 18),
        37 => ("IQ4_NL_4_8", 32, 18),
        38 => ("IQ4_NL_8_8", 32, 18),
        39 => ("MXFP4", 32, 17),
        41 => ("Q1_0", 128, 18),
        42 => ("Q2_0", 64, 18),
        142 => ("PQ2_0", 128, 34),
        _ => return None,
    };
    Some(T { name, block, bytes })
}

/// Human-readable name for any id, including ones we cannot size.
pub fn type_name(ty: u32) -> &'static str {
    if ty == TY_PTQ1_0 {
        return "PTQ1_0";
    }
    type_info(ty).map(|t| t.name).unwrap_or("UNKNOWN")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pq2_is_34_bytes_per_128() {
        let t = type_info(TY_PQ2_0).unwrap();
        assert_eq!((t.block, t.bytes), (128, 34));
    }

    #[test]
    fn q2_is_18_bytes_per_64() {
        let t = type_info(TY_Q2_0).unwrap();
        assert_eq!((t.block, t.bytes), (64, 18));
    }

    #[test]
    fn ptq1_has_no_layout() {
        assert!(type_info(TY_PTQ1_0).is_none());
        assert_eq!(type_name(TY_PTQ1_0), "PTQ1_0");
    }
}
