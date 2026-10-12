"""
Run after main_pipeline.py in the same session (needs baseline_model,
attention_model, physics_model, device, optimal_threshold_baseline,
optimal_threshold_attention, optimal_threshold_physics, seismic_data).

Produces:
  1. Full-section fault probability maps (sliding window, overlap-averaged)
  2. FCI / DRR for all three models
  3. RTM overlay figure (baseline vs physics-guided)
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from continuity_diffraction_metrics import summarize_continuity_and_diffraction

PATCH = 256
STRIDE = 128


@torch.no_grad()
def predict_full_section(model, seismic_2d, patch=PATCH, stride=STRIDE, resize_to=128):
    import torch.nn.functional as F

    model.eval()
    H, W = seismic_2d.shape
    prob_sum = np.zeros((H, W), dtype=np.float64)
    count = np.zeros((H, W), dtype=np.float64)

    for r in range(0, H - patch + 1, stride):
        for c in range(0, W - patch + 1, stride):
            tile = seismic_2d[r:r + patch, c:c + patch]
            t = torch.from_numpy(tile).float().unsqueeze(0).unsqueeze(0).to(device)
            t_resized = F.interpolate(t, size=(resize_to, resize_to), mode='bilinear', align_corners=False)
            pred = model(t_resized)
            pred_full = F.interpolate(pred, size=(patch, patch), mode='bilinear', align_corners=False)
            pred_np = pred_full.squeeze().cpu().numpy()
            prob_sum[r:r + patch, c:c + patch] += pred_np
            count[r:r + patch, c:c + patch] += 1.0

    count[count == 0] = 1.0
    return (prob_sum / count).astype(np.float32)


print("Reconstructing full-section predictions...")
prob_baseline_full = predict_full_section(baseline_model, seismic_data)
prob_attention_full = predict_full_section(attention_model, seismic_data)
prob_physics_full = predict_full_section(physics_model, seismic_data)
print("Done.")

results = {}
for name, prob_map, thr in [
    ("Baseline U-Net", prob_baseline_full, optimal_threshold_baseline),
    ("Attention U-Net", prob_attention_full, optimal_threshold_attention),
    ("Physics-guided U-Net", prob_physics_full, optimal_threshold_physics),
]:
    metrics = summarize_continuity_and_diffraction(seismic_data, prob_map, thr)
    results[name] = metrics
    print(f"\n{name} (threshold={thr:.3f}):")
    print(f"  Fault Continuity Index:        {metrics['fault_continuity_index']:.4f}")
    print(f"  Diffraction-to-Reflection Ratio: {metrics['diffraction_reflection_ratio']:.4f}")
    print(f"  Predicted fault pixels:        {metrics['n_fault_pixels']}")

print("\n" + "=" * 70)
print("LaTeX table row (Method & FCI & DRR):")
for name, m in results.items():
    print(f"{name} & {m['fault_continuity_index']:.3f} & {m['diffraction_reflection_ratio']:.3f} \\\\\\\\\\\\\\\\")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (title, prob_map) in zip(
    axes, [("(a) Baseline U-Net", prob_baseline_full), ("(b) Physics-Guided U-Net", prob_physics_full)]
):
    ax.imshow(seismic_data, cmap='gray', aspect='auto', alpha=0.8)
    masked_prob = np.ma.masked_where(prob_map < 0.3, prob_map)
    ax.imshow(masked_prob, cmap='hot', aspect='auto', alpha=0.6, vmin=0.3, vmax=1.0)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Traces")
    ax.set_ylabel("Samples")
plt.tight_layout()
plt.savefig("Fig10.png", dpi=200, bbox_inches='tight')
print("\nSaved Fig10.png")
