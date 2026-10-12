"""
C3 coherence baseline, evaluated on the same test patches/labels/bootstrap
procedure as the U-Net models, for a like-for-like comparison row.
"""
import numpy as np
import segyio
from c3_coherence import c3_coherence_fast


def load_segy_section(path):
    with segyio.open(path, 'r', ignore_geometry=True) as f:
        data = np.stack([f.trace[i] for i in range(f.tracecount)]).T
    return data


def standardize(section):
    s = (section - section.mean()) / (section.std() + 1e-8)
    return np.clip(s, -4, 4)


def evaluate_c3_on_patches(section_std, faults_data, patch_coords, patch_size=256,
                            time_window=9, trace_window=5, n_bootstraps=1000, seed=42):
    coh = c3_coherence_fast(section_std, time_window=time_window, trace_window=trace_window)
    fault_attr = 1.0 - coh

    preds, targets = [], []
    for (r, c) in patch_coords:
        preds.append(fault_attr[r:r + patch_size, c:c + patch_size])
        targets.append(faults_data[r:r + patch_size, c:c + patch_size])
    preds = np.stack(preds)
    targets = np.stack(targets)

    flat_p = preds.flatten()
    flat_t = targets.flatten()
    thresholds = np.linspace(flat_p.min(), flat_p.max(), 200)
    best_f1, best_thr = -1, 0.5
    for thr in thresholds:
        pred_bin = flat_p > thr
        tp = np.sum(pred_bin & (flat_t > 0.5))
        fp = np.sum(pred_bin & (flat_t <= 0.5))
        fn = np.sum((~pred_bin) & (flat_t > 0.5))
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    print(f"  C3 coherence optimal threshold: {best_thr:.4f} (F1={best_f1:.4f})")

    pred_bin_flat = (flat_p > best_thr).astype(np.float32)
    indices = np.arange(len(flat_t))
    rng = np.random.default_rng(seed)
    out = {"iou": [], "f1": [], "precision": [], "recall": []}
    for _ in range(n_bootstraps):
        idx = rng.choice(indices, size=len(indices), replace=True)
        rt, rp = flat_t[idx], pred_bin_flat[idx]
        inter = np.sum(rp * rt)
        union = np.sum((rp + rt) > 0)
        iou = inter / (union + 1e-8)
        tp = inter
        fp = np.sum(rp) - tp
        fn = np.sum(rt) - tp
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        out["iou"].append(iou); out["f1"].append(f1)
        out["precision"].append(prec); out["recall"].append(rec)

    summary = {}
    for k, v in out.items():
        v = np.array(v)
        summary[k] = {"mean": float(v.mean()), "std": float(v.std()),
                      "ci95_lo": float(np.percentile(v, 2.5)),
                      "ci95_hi": float(np.percentile(v, 97.5))}
    return summary


if __name__ == "__main__":
    import sys
    segy_path = sys.argv[1] if len(sys.argv) > 1 else "SYNTHETIC.segy"

    section = load_segy_section(segy_path)
    section_std = standardize(section)

    gx, gy = np.gradient(section_std)
    grad = np.sqrt(gx**2 + gy**2)
    grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)
    faults_data = (grad > np.percentile(grad, 95)).astype(np.float32)
    print("Fault label fraction:", faults_data.mean())

    ns, nt = section_std.shape
    patch_size, stride = 256, 128
    coords = []
    for r in range(0, ns - patch_size + 1, stride):
        for c in range(0, nt - patch_size + 1, stride):
            coords.append((r, c))
    print(f"Total patches available: {len(coords)}")

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(coords))
    n = len(coords)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    test_idx = idx[n_train + n_val:]
    test_coords = [coords[i] for i in test_idx]
    print(f"Test patches: {len(test_coords)}")

    print("\nEvaluating C3 coherence on test patches...")
    summary = evaluate_c3_on_patches(section_std, faults_data, test_coords,
                                      patch_size=patch_size, n_bootstraps=1000)

    print("\n" + "="*80)
    print("TRADITIONAL (C3 COHERENCE)")
    print("="*80)
    for k, v in summary.items():
        print(f"  {k}: {v['mean']:.3f} [{v['ci95_lo']:.3f}, {v['ci95_hi']:.3f}]")

    def fmt(m):
        return f"{summary[m]['mean']:.3f} [{summary[m]['ci95_lo']:.3f}, {summary[m]['ci95_hi']:.3f}]"
    print("\nLaTeX row:")
    print(f"Traditional (C3 Coherence) & {fmt('iou')} & {fmt('f1')} & {fmt('precision')} & {fmt('recall')} \\\\\\\\\\\\\\\\")
