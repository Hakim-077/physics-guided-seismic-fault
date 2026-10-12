"""
Fault Continuity Index (FCI) and Diffraction-to-Reflection Ratio (DRR).

FCI: fragmentation index over connected components of the binary fault
mask (Simpson's concentration index, landscape ecology).

    FCI = sum(size_i^2) / (sum(size_i))^2

Range (0,1]. 1 = single connected structure, ->0 = fragmented into many
small pieces.

DRR: decomposes the local seismic gradient at each predicted-fault pixel
into components parallel/perpendicular to the REGIONAL (smoothed) dip
trend. Reflection-aligned picks -> low DRR; diffraction-like (genuine
fault) picks -> high DRR.

    DRR = mean(|grad . D_perp|) / mean(|grad . D_parallel|)

Note: must decompose against the smoothed regional dip, not the raw
local dip -- decomposing a vector against its own direction gives a
perpendicular component of exactly zero.
"""
import numpy as np
from scipy import ndimage


def fault_continuity_index(binary_fault_mask: np.ndarray) -> float:
    mask = binary_fault_mask.astype(bool)
    if mask.sum() == 0:
        return 0.0
    labeled, n_components = ndimage.label(mask, structure=np.ones((3, 3)))
    if n_components == 0:
        return 0.0
    sizes = ndimage.sum(mask, labeled, index=np.arange(1, n_components + 1))
    total = sizes.sum()
    return float(np.sum(sizes ** 2) / (total ** 2))


def diffraction_reflection_ratio(seismic: np.ndarray, binary_fault_mask: np.ndarray,
                                  regional_sigma: float = 8.0, eps: float = 1e-8) -> float:
    mask = binary_fault_mask.astype(bool)
    if mask.sum() == 0:
        return 0.0

    gz, gx = np.gradient(seismic)
    mag = np.sqrt(gx ** 2 + gz ** 2) + eps
    dz_local, dx_local = gz / mag, gx / mag

    dz_reg = ndimage.gaussian_filter(dz_local, sigma=regional_sigma)
    dx_reg = ndimage.gaussian_filter(dx_local, sigma=regional_sigma)
    reg_mag = np.sqrt(dx_reg ** 2 + dz_reg ** 2) + eps
    dz_reg, dx_reg = dz_reg / reg_mag, dx_reg / reg_mag

    pz_reg, px_reg = -dx_reg, dz_reg

    parallel_component = np.abs(gx * dx_reg + gz * dz_reg)
    perp_component = np.abs(gx * px_reg + gz * pz_reg)

    mean_parallel = parallel_component[mask].mean()
    mean_perp = perp_component[mask].mean()

    return float(mean_perp / (mean_parallel + eps))


def summarize_continuity_and_diffraction(seismic: np.ndarray, prob_map: np.ndarray,
                                          threshold: float) -> dict:
    binary = (prob_map > threshold).astype(np.uint8)
    return {
        "fault_continuity_index": fault_continuity_index(binary),
        "diffraction_reflection_ratio": diffraction_reflection_ratio(seismic, binary),
        "n_fault_pixels": int(binary.sum()),
    }
