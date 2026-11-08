//! Choose the highest-quality quantization target that fits a byte budget and
//! a context length.
//!
//! The weights estimate uses the exact tensor table and the same recipe engine
//! as the converter, so it does not depend on running a trial conversion.

use std::collections::BTreeMap;
use std::path::Path;

use crate::fit::{estimate, hardware_fits, FitRequest, Hardware};
use crate::gguf::Gguf;
use crate::recipe::effective_tensor_types;
use crate::transcode::Target;
use crate::types::type_info;

#[derive(Debug, thiserror::Error)]
pub enum BudgetError {
    #[error("cannot read GGUF: {0}")]
    Gguf(String),
    #[error("cannot estimate KV cache: {0}")]
    Fit(String),
    #[error("no usable target for this model")]
    NoTargets,
}

#[derive(Debug, Clone, Copy)]
pub struct BudgetRequest {
    pub budget_bytes: Option<u64>,
    pub context: u64,
    pub overhead_bytes: u64,
    pub hardware: Option<Hardware>,
}

impl Default for BudgetRequest {
    fn default() -> Self {
        Self {
            budget_bytes: None,
            context: 4096,
            overhead_bytes: 512 * 1024 * 1024,
            hardware: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BudgetCandidate {
    pub target: Target,
    pub weights_bytes: u64,
    pub kv_cache_bytes: u64,
    pub total_bytes: u64,
    pub fits_budget: Option<bool>,
    pub fits_ram: Option<bool>,
    pub fits_vram: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct BudgetPlan {
    pub path: String,
    pub context: u64,
    pub budget_bytes: Option<u64>,
    pub overhead_bytes: u64,
    pub candidates: Vec<BudgetCandidate>,
    pub selected: Target,
    pub selected_total_bytes: u64,
    pub fits_ram: Option<bool>,
    pub fits_vram: Option<bool>,
    pub reason: String,
}

impl BudgetPlan {
    pub fn selected_candidate(&self) -> Option<&BudgetCandidate> {
        self.candidates
            .iter()
            .find(|candidate| candidate.target == self.selected)
    }
}

/// Targets are ordered from highest expected quality to smallest. The ternary
/// and Prism-only targets are intentionally excluded from automatic selection.
const QUALITY_ORDER: &[Target] = &[
    Target::F16,
    Target::Bf16,
    Target::Q8_0,
    Target::Q6K,
    Target::Q5KM,
    Target::Q5K,
    Target::Q4KM,
    Target::Q4K,
    Target::Q3KM,
    Target::Q3K,
    Target::Q2K,
    Target::F32,
];

pub fn plan(path: &Path, request: BudgetRequest) -> Result<BudgetPlan, BudgetError> {
    let gguf = Gguf::open(path).map_err(|error| BudgetError::Gguf(error.to_string()))?;
    let fit = estimate(
        path,
        FitRequest {
            context: Some(request.context.max(1)),
            cache_element_bytes: 2,
            overhead_bytes: 0,
            hardware: request.hardware,
        },
    )
    .map_err(|error| BudgetError::Fit(error.to_string()))?;
    let kv_cache_bytes = fit.kv_cache_bytes;

    let mut candidates = Vec::new();
    for &target in QUALITY_ORDER {
        let weights_bytes = estimate_weights(&gguf, target);
        let total_bytes = weights_bytes
            .saturating_add(kv_cache_bytes)
            .saturating_add(request.overhead_bytes);
        let (fits_ram, fits_vram) = fit_verdicts(request.hardware, total_bytes);
        let fits_budget = request.budget_bytes.map(|budget| total_bytes <= budget);
        candidates.push(BudgetCandidate {
            target,
            weights_bytes,
            kv_cache_bytes,
            total_bytes,
            fits_budget,
            fits_ram,
            fits_vram,
        });
    }
    if candidates.is_empty() {
        return Err(BudgetError::NoTargets);
    }

    let hardware_known = request
        .hardware
        .is_some_and(|hardware| hardware.ram_bytes.is_some() || hardware.vram_bytes.is_some());
    let selected = if request.budget_bytes.is_some() {
        candidates
            .iter()
            .find(|candidate| {
                candidate.fits_budget == Some(true)
                    && candidate.fits_ram != Some(false)
                    && candidate.fits_vram != Some(false)
            })
            .or_else(|| {
                candidates
                    .iter()
                    .find(|candidate| candidate.fits_budget == Some(true))
            })
            .or_else(|| {
                candidates
                    .iter()
                    .min_by_key(|candidate| candidate.total_bytes)
            })
            .expect("non-empty")
    } else if hardware_known {
        candidates
            .iter()
            .find(|candidate| {
                candidate.fits_ram != Some(false)
                    && candidate.fits_vram != Some(false)
                    && (candidate.fits_ram.is_some() || candidate.fits_vram.is_some())
            })
            .or_else(|| {
                candidates
                    .iter()
                    .min_by_key(|candidate| candidate.total_bytes)
            })
            .expect("non-empty")
    } else {
        candidates
            .iter()
            .find(|candidate| candidate.target == Target::Q4KM)
            .or_else(|| {
                candidates
                    .iter()
                    .min_by_key(|candidate| candidate.total_bytes)
            })
            .expect("non-empty")
    };

    let reason = if request.budget_bytes.is_some() {
        if selected.fits_budget == Some(true) {
            "highest-quality target inside the requested budget".to_string()
        } else {
            "no target fits the requested budget; showing the smallest estimate".to_string()
        }
    } else if hardware_known {
        if selected.fits_ram == Some(true) || selected.fits_vram == Some(true) {
            "highest-quality target that fits the detected hardware".to_string()
        } else {
            "no target fits the detected hardware; showing the smallest estimate".to_string()
        }
    } else {
        "no hardware information; using the balanced default".to_string()
    };

    let selected_target = selected.target;
    let selected_total_bytes = selected.total_bytes;
    let fits_ram = selected.fits_ram;
    let fits_vram = selected.fits_vram;
    Ok(BudgetPlan {
        path: path.display().to_string(),
        context: request.context.max(1),
        budget_bytes: request.budget_bytes,
        overhead_bytes: request.overhead_bytes,
        candidates,
        selected: selected_target,
        selected_total_bytes,
        fits_ram,
        fits_vram,
        reason,
    })
}

fn estimate_weights(gguf: &Gguf, target: Target) -> u64 {
    let types = effective_tensor_types(&gguf.tensors, target, &[]);
    let by_name: BTreeMap<&str, u32> = types
        .iter()
        .map(|(name, ty)| (name.as_str(), *ty))
        .collect();
    gguf.tensors
        .iter()
        .map(|tensor| {
            let ty = by_name
                .get(tensor.name.as_str())
                .copied()
                .unwrap_or(tensor.ty);
            let Some(info) = type_info(ty) else {
                return tensor.size.unwrap_or(0);
            };
            let elements = tensor.element_count();
            if elements % info.block == 0 {
                (elements / info.block).saturating_mul(info.bytes)
            } else {
                tensor.size.unwrap_or(0)
            }
        })
        .fold(0u64, u64::saturating_add)
}

fn fit_verdicts(hardware: Option<Hardware>, total: u64) -> (Option<bool>, Option<bool>) {
    match hardware {
        Some(hardware) => (
            hardware.ram_bytes.map(|ram| hardware_fits(total, ram)),
            hardware.vram_bytes.map(|vram| hardware_fits(total, vram)),
        ),
        None => (None, None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcode::Target;

    #[test]
    fn quality_order_is_descending_by_expected_quality() {
        let index = |target: Target| QUALITY_ORDER.iter().position(|item| *item == target);
        assert!(index(Target::F16) < index(Target::Q8_0));
        assert!(index(Target::Q6K) < index(Target::Q4KM));
        assert!(index(Target::Q4KM) < index(Target::Q2K));
    }
}
