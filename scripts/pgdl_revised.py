# =============================================================================
# PHYSICS-GUIDED SEISMIC FAULT DETECTION -- REVISED EXPERIMENTAL PIPELINE
# (AJGS-D-26-00099, revision)
#
# What changed relative to the previous notebook, and why (keep this list
# for the response letter and the Methods section):
#   1. Every model is trained FROM SCRATCH (fresh random initialisation).
#      Previously the physics-guided, lambda-sweep and ablation models were
#      initialised from the trained baseline, i.e. they received the
#      baseline's 10 epochs PLUS their own; gains could not be attributed to
#      the physics terms.
#   2. All models get the SAME training budget: up to MAX_EPOCHS epochs with
#      early stopping (patience PATIENCE) on the validation BCE, and the
#      checkpoint with the lowest validation BCE is kept.
#   3. Spatially blocked cross-validation. Patches (256 px, stride 128)
#      overlap their lateral neighbours by 50 %, so a random split puts
#      near-copies of test patches in the training set. The section is now
#      split into N_FOLDS contiguous lateral blocks; in each fold one block
#      is the test set, the next block is the validation set, and the patch
#      columns adjacent to both are removed from training (guard band), so no
#      training patch overlaps a validation or test patch. Every patch is
#      tested exactly once; results are pooled over folds.
#   4. Decision thresholds and lambda are selected on the VALIDATION block
#      of each fold and then applied unchanged to the test block (previously
#      the "optimal threshold" was taken from the test-set PR curve).
#   5. Several random seeds per configuration; results are reported as
#      mean +/- s.d. over seeds.
#   6. Statistics use independent units: 95 % CIs by bootstrap over TEST
#      PATCHES (not pixels); paired Wilcoxon signed-rank tests and Cliff's
#      delta on per-patch IoU; Holm correction for multiple comparisons.
#      (Previously the Wilcoxon test was applied to 1000 bootstrap replicates
#      of one result, which always returns p ~ 3e-165.)
#   7. FCI / DRR are computed on the stitched held-out predictions (each
#      column predicted by the model for which it was test data), with the
#      validation threshold, for every seed.
#   8. The same resize routine is used for training patches and for
#      full-section inference.
# The network architecture and the four physics terms are unchanged.
# =============================================================================
import os
import json
import random
import warnings

import numpy as np

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy import ndimage
from scipy.stats import wilcoxon
from skimage.transform import resize
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, roc_curve

warnings.filterwarnings("ignore")

# =========================
# 0. CONFIGURATION
# =========================
SEGY_PATH = "SYNTHETIC.segy"          # migrated section (.segy or .npy, samples x traces)
VP_PATH = "vp_marmousi-ii.segy"       # only needed if LABEL_SOURCE == "velocity"
LABEL_SOURCE = "gradient_p95"         # "gradient_p95" (as in the manuscript) or "velocity"
PATCH, STRIDE, NET_SIZE = 256, 128, 128
MIN_FAULT_FRACTION = 0.005            # patch kept if its label fraction exceeds this
N_FOLDS = 5
SEEDS = [0, 1, 2, 3, 4]
LAMBDAS = [0.01, 0.02, 0.05, 0.1, 0.2]
MAX_EPOCHS, PATIENCE, LR, BATCH = 60, 10, 1e-3, 8
N_BOOT = 2000
THR_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)
OUT = "results_revised"
RUN_ABLATIONS = True
QUICK = False                          # True: 2 seeds, 3 lambdas, 8 epochs (pipeline check only)

if QUICK:
    SEEDS, LAMBDAS, MAX_EPOCHS, PATIENCE, N_BOOT = [0, 1], [0.02, 0.05, 0.1], 8, 4, 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "log.txt"), "w")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# 1. DATA AND LABELS
# =========================
def load_section(path):
    if path.endswith(".npy"):
        data = np.load(path).astype(np.float32)
    else:
        import segyio
        with segyio.open(path, "r", ignore_geometry=True) as f:
            data = np.stack([f.trace[i] for i in range(f.tracecount)]).T.astype(np.float32)
    return data


def standardize(data):
    data = (data - data.mean()) / data.std()
    return np.clip(data, -4, 4).astype(np.float32)


def labels_gradient_p95(seismic):
    """Manuscript Eq. (pseudo-labels): top 5 % of the gradient magnitude."""
    gz, gx = np.gradient(seismic)
    grad = np.sqrt(gx ** 2 + gz ** 2)
    return (grad > np.percentile(grad, 95)).astype(np.float32)


def labels_from_velocity(vp, shape, sigma_d=1.5, sigma_i=10.0, low=0.35, high=0.8, min_angle_deg=25.0):
    """Optional: faults = discontinuity of log(Vp) ALONG the local bedding
    (structure-tensor tangent), hysteresis-thresholded; bedding-parallel
    segments rejected. Must be reviewed by an interpreter before use."""
    from skimage.filters import apply_hysteresis_threshold
    from skimage.measure import label as cc_label, regionprops
    from skimage.morphology import skeletonize, disk
    if vp.shape != shape:
        log(f"WARNING: Vp grid {vp.shape} != image grid {shape}; resampling Vp (check alignment!)")
        vp = resize(vp, shape, order=1, preserve_range=True, anti_aliasing=True)
    lv = np.log(np.maximum(vp, 1.0))
    gz = ndimage.gaussian_filter(lv, sigma_d, order=(1, 0))
    gx = ndimage.gaussian_filter(lv, sigma_d, order=(0, 1))
    jzz, jxx, jzx = (ndimage.gaussian_filter(a, sigma_i) for a in (gz * gz, gx * gx, gz * gx))
    th = 0.5 * np.arctan2(2 * jzx, jzz - jxx)
    nz, nx = np.cos(th), np.sin(th)
    tang = np.abs(gz * (-nx) + gx * nz)
    valid = ndimage.binary_erosion(vp > 1550, iterations=10, border_value=1)
    tang[~valid] = 0
    lik = tang / (np.percentile(tang[valid], 99.5) + 1e-12)
    cand = apply_hysteresis_threshold(lik, low, high)
    keep = np.zeros_like(cand)
    lab = cc_label(cand, connectivity=2)
    cmin = np.cos(np.deg2rad(min_angle_deg))
    for r in regionprops(lab):
        if r.area < 40:
            continue
        rr, cc = r.coords[:, 0], r.coords[:, 1]
        axis = np.linalg.eigh(np.cov(np.vstack([rr, cc])))[1][:, -1]
        t = np.array([-nx[rr, cc].mean(), nz[rr, cc].mean()])
        if abs(axis @ (t / (np.linalg.norm(t) + 1e-12))) < cmin:
            keep[lab == r.label] = True
    out = ndimage.binary_dilation(skeletonize(keep), structure=disk(1))
    return (out & valid).astype(np.float32)


def to_net(a, is_label=False):
    """Single resize routine used for training AND inference."""
    if is_label:
        r = resize(a, (NET_SIZE, NET_SIZE), order=1, mode="constant", anti_aliasing=False)
        return (r > 0.5).astype(np.float32)
    return resize(a, (NET_SIZE, NET_SIZE), order=1, mode="constant", anti_aliasing=True).astype(np.float32)


def extract_patches(seismic, faults):
    H, W = seismic.shape
    xs, ys, rows, cols = [], [], [], []
    for i in range(0, H - PATCH + 1, STRIDE):
        for j in range(0, W - PATCH + 1, STRIDE):
            f = faults[i:i + PATCH, j:j + PATCH]
            if f.mean() > MIN_FAULT_FRACTION:
                xs.append(to_net(seismic[i:i + PATCH, j:j + PATCH]))
                ys.append(to_net(f, True))
                rows.append(i)
                cols.append(j // STRIDE)
    X = np.stack(xs)[:, None]
    Y = np.stack(ys)[:, None]
    return X, Y, np.array(rows), np.array(cols)


def spatial_folds(cols, n_folds):
    """Contiguous lateral blocks of patch columns; test = block k, val = block k+1,
    guard = columns adjacent to test/val removed from training."""
    uc = np.unique(cols)
    blocks = np.array_split(uc, n_folds)
    folds = []
    for k in range(n_folds):
        tcols, vcols = set(blocks[k]), set(blocks[(k + 1) % n_folds])
        banned = set()
        for c in tcols | vcols:
            banned |= {c - 1, c, c + 1}
        test = np.where(np.isin(cols, list(tcols)))[0]
        val = np.where(np.isin(cols, list(vcols)))[0]
        train = np.where(~np.isin(cols, list(banned)))[0]
        folds.append(dict(train=train, val=val, test=test, test_cols=sorted(tcols)))
    return folds


# =========================
# 2. MODELS (unchanged from the manuscript code)
# =========================
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                                  nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.enc2 = nn.Sequential(nn.MaxPool2d(2),
                                  nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(64, 32, 2, stride=2),
                                  nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.final = nn.Sequential(nn.Conv2d(32, 1, 1), nn.Sigmoid())

    def forward(self, x):
        x1 = self.enc1(x)
        return self.final(x1 + self.dec2(self.enc2(x1)))


class AttentionGate(nn.Module):
    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.w_gate = nn.Sequential(nn.Conv2d(gate_ch, inter_ch, 1), nn.BatchNorm2d(inter_ch))
        self.w_skip = nn.Sequential(nn.Conv2d(skip_ch, inter_ch, 1), nn.BatchNorm2d(inter_ch))
        self.psi = nn.Sequential(nn.Conv2d(inter_ch, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, gate, skip):
        return skip * self.psi(F.relu(self.w_gate(gate) + self.w_skip(skip)))


class AttentionUNet(UNet):
    def __init__(self):
        super().__init__()
        self.att = AttentionGate(32, 32, 16)

    def forward(self, x):
        x1 = self.enc1(x)
        up = self.dec2(self.enc2(x1))
        return self.final(self.att(up, x1) + up)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# =========================
# 3. PHYSICS TERMS (unchanged from the manuscript code)
# =========================
PHYSICS_WEIGHTS = dict(dip=0.4, sparsity=0.3, continuity=0.2, contrast=0.1)


def compute_dip_field(s):
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=s.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=s.device).view(1, 1, 3, 3)
    d = torch.cat([F.conv2d(s, kx, padding=1), F.conv2d(s, ky, padding=1)], 1)
    return d / (torch.norm(d, dim=1, keepdim=True) + 1e-8)


def dip_alignment_loss(p, s):
    gy, gx = torch.gradient(p, dim=[2, 3])
    g = torch.cat([gx, gy], 1)
    g = g / (torch.norm(g, dim=1, keepdim=True) + 1e-8)
    return torch.abs((g * compute_dip_field(s)).sum(1)).mean()


def sparsity_loss(p):
    return 0.8 * p.abs().mean() + 0.2 * ((p - p.mean(dim=[2, 3], keepdim=True)) ** 2).mean()


def continuity_loss(p):
    return 0.5 * ((p[:, :, 1:] - p[:, :, :-1]).abs().mean() + (p[:, :, :, 1:] - p[:, :, :, :-1]).abs().mean())


def contrast_loss(p, s):
    gy, gx = torch.gradient(s, dim=[2, 3])
    g = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
    g = (g - g.min()) / (g.max() - g.min() + 1e-8)
    return 1.0 - (p * g).sum() / (p.sum() + 1e-8)


TERMS = dict(dip=lambda p, s: dip_alignment_loss(p, s), sparsity=lambda p, s: sparsity_loss(p),
             continuity=lambda p, s: continuity_loss(p), contrast=lambda p, s: contrast_loss(p, s))


def physics_loss(p, s, lam, weights):
    if lam == 0:
        return p.new_zeros(())
    return lam * sum(w * TERMS[k](p, s) for k, w in weights.items() if w > 0)


# =========================
# 4. TRAINING AND PREDICTION
# =========================
def make_loader(X, Y, idx, shuffle, seed):
    ds = TensorDataset(torch.from_numpy(X[idx]), torch.from_numpy(Y[idx]))
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, generator=g)


def train_fresh(arch, X, Y, fold, seed, lam, weights):
    """ALWAYS a new randomly initialised model; identical budget for every run."""
    set_seed(seed)
    model = (AttentionUNet if arch == "attention" else UNet)().to(device)
    opt = torch.optim.Adam(model.parameters(), LR)
    bce = nn.BCELoss()
    tl = make_loader(X, Y, fold["train"], True, seed)
    vl = make_loader(X, Y, fold["val"], False, seed)
    best, best_state, wait, best_ep = np.inf, None, 0, 0
    for ep in range(MAX_EPOCHS):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            p = model(x)
            loss = bce(p, y) + physics_loss(p, x, lam, weights)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = np.mean([bce(model(x.to(device)), y.to(device)).item() for x, y in vl])
        if v < best - 1e-5:
            best, wait, best_ep = v, 0, ep + 1
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best_ep


@torch.no_grad()
def predict(model, X):
    model.eval()
    out = [model(torch.from_numpy(X[i:i + 32]).to(device)).cpu().numpy() for i in range(0, len(X), 32)]
    return np.concatenate(out)[:, 0]


def counts(prob, y, thr):
    pb = prob > thr
    yb = y > 0.5
    return np.array([(pb & yb).sum(), (pb & ~yb).sum(), (~pb & yb).sum()], dtype=np.float64)


def scores(c):
    tp, fp, fn = c
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    return dict(iou=tp / (tp + fp + fn + 1e-8), precision=p, recall=r, f1=2 * p * r / (p + r + 1e-8))


def val_threshold(prob, y):
    f1 = [scores(counts(prob, y, t))["f1"] for t in THR_GRID]
    i = int(np.argmax(f1))
    return float(THR_GRID[i]), float(f1[i])


@torch.no_grad()
def predict_section(model, sec):
    """Sliding-window inference on a (sub)section with the training resize routine."""
    H, W = sec.shape
    rs = list(range(0, H - PATCH + 1, STRIDE // 2))
    cs = list(range(0, W - PATCH + 1, STRIDE // 2))
    if rs[-1] != H - PATCH:
        rs.append(H - PATCH)
    if cs[-1] != W - PATCH:
        cs.append(W - PATCH)
    acc = np.zeros((H, W))
    cnt = np.zeros((H, W))
    model.eval()
    for r in rs:
        tiles = np.stack([to_net(sec[r:r + PATCH, c:c + PATCH]) for c in cs])[:, None]
        pr = predict(model, tiles)
        for c, p in zip(cs, pr):
            acc[r:r + PATCH, c:c + PATCH] += resize(p, (PATCH, PATCH), order=1)
            cnt[r:r + PATCH, c:c + PATCH] += 1
    return (acc / np.maximum(cnt, 1)).astype(np.float32)


# =========================
# 5. STRUCTURAL METRICS (definitions as in the manuscript, Sect. 5.6)
# =========================
def fault_continuity_index(mask):
    lab, n = ndimage.label(mask.astype(bool), structure=np.ones((3, 3)))
    if n == 0:
        return 0.0
    s = ndimage.sum(mask.astype(bool), lab, index=np.arange(1, n + 1))
    return float((s ** 2).sum() / s.sum() ** 2)


def diffraction_reflection_ratio(seismic, mask, regional_sigma=8.0, eps=1e-8):
    mask = mask.astype(bool)
    if not mask.any():
        return float("nan")
    gz, gx = np.gradient(seismic)
    mag = np.sqrt(gx ** 2 + gz ** 2) + eps
    dz = ndimage.gaussian_filter(gz / mag, regional_sigma)
    dx = ndimage.gaussian_filter(gx / mag, regional_sigma)
    m = np.sqrt(dx ** 2 + dz ** 2) + eps
    dz, dx = dz / m, dx / m
    par = np.abs(gx * dx + gz * dz)
    perp = np.abs(gx * dz - gz * dx)
    return float(perp[mask].mean() / (par[mask].mean() + eps))


# =========================
# 6. STATISTICS (independent unit = test patch)
# =========================
def patch_bootstrap_ci(per_patch_counts, key, n_boot=N_BOOT, seed=0):
    c = np.asarray(per_patch_counts)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(c), (n_boot, len(c)))
    b = np.array([scores(c[i].sum(0))[key] for i in idx])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def cliffs_delta(a, b):
    a = np.asarray(a)[:, None]
    b = np.asarray(b)[None, :]
    return float(((a > b).sum() - (a < b).sum()) / (a.size * b.size))


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for k, i in enumerate(o):
        run = max(run, (len(p) - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def per_patch_iou(c):
    return np.array([scores(x)["iou"] for x in c])


# =========================
# 7. EXPERIMENT
# =========================
def run_config(name, arch, lam_per_fold, weights, X, Y, folds, seeds):
    """Train/evaluate one configuration over all folds and seeds.
    Returns per-seed arrays of per-patch counts (ordered by global patch index),
    per-seed test probabilities, thresholds and early-stopping epochs."""
    res = dict(counts={}, probs={}, thr={}, epochs={}, fold_scores={}, models={})
    for s in seeds:
        C = np.zeros((len(X), 3))
        P = np.zeros((len(X), NET_SIZE, NET_SIZE), np.float32)
        thr_s, ep_s, fs = [], [], []
        for k, fold in enumerate(folds):
            lam = lam_per_fold[k]
            model, ep = train_fresh(arch, X, Y, fold, s, lam, weights)
            t, _ = val_threshold(predict(model, X[fold["val"]]), Y[fold["val"], 0])
            pt = predict(model, X[fold["test"]])
            for j, gi in enumerate(fold["test"]):
                C[gi] = counts(pt[j], Y[gi, 0], t)
                P[gi] = pt[j]
            fs.append(scores(C[fold["test"]].sum(0)))
            thr_s.append(t)
            ep_s.append(ep)
            res["models"][(s, k)] = model.cpu()
        res["counts"][s], res["probs"][s] = C, P
        res["thr"][s], res["epochs"][s], res["fold_scores"][s] = thr_s, ep_s, fs
        pooled = scores(C.sum(0))
        log(f"  {name:28s} seed {s}: IoU {pooled['iou']:.3f} F1 {pooled['f1']:.3f} "
            f"P {pooled['precision']:.3f} R {pooled['recall']:.3f} | thr {np.round(thr_s, 2).tolist()} "
            f"| epochs {ep_s}")
    return res


def summarize(res, Y):
    seeds = sorted(res["counts"])
    per_seed = [scores(res["counts"][s].sum(0)) for s in seeds]
    out = {k: (float(np.mean([p[k] for p in per_seed])), float(np.std([p[k] for p in per_seed], ddof=1)
                                                               if len(seeds) > 1 else 0.0))
           for k in ("iou", "f1", "precision", "recall")}
    Csum = np.sum([res["counts"][s] for s in seeds], 0)
    out["ci"] = {k: patch_bootstrap_ci(Csum, k) for k in ("iou", "f1")}
    ap, auc_ = [], []
    for s in seeds:
        y = (Y[:, 0] > 0.5).ravel()
        ap.append(average_precision_score(y, res["probs"][s].ravel()))
        auc_.append(roc_auc_score(y, res["probs"][s].ravel()))
    out["ap"] = (float(np.mean(ap)), float(np.std(ap, ddof=1)) if len(ap) > 1 else 0.0)
    out["roc_auc"] = (float(np.mean(auc_)), float(np.std(auc_, ddof=1)) if len(auc_) > 1 else 0.0)
    out["per_patch_iou"] = np.mean([per_patch_iou(res["counts"][s]) for s in seeds], 0)
    out["epochs"] = float(np.mean([np.mean(res["epochs"][s]) for s in seeds]))
    return out


def fmt(ms, d=3):
    return f"{ms[0]:.{d}f} $\\pm$ {ms[1]:.{d}f}"


def main():
    log(f"device={device}  QUICK={QUICK}  seeds={SEEDS}")
    seismic = standardize(load_section(SEGY_PATH))
    if LABEL_SOURCE == "velocity":
        faults = labels_from_velocity(load_section(VP_PATH), seismic.shape)
    else:
        faults = labels_gradient_p95(seismic)
    log(f"section {seismic.shape}; label source {LABEL_SOURCE}; fault fraction {faults.mean():.4f}")

    X, Y, rows, cols = extract_patches(seismic, faults)
    folds = spatial_folds(cols, N_FOLDS)
    log(f"{len(X)} patches, {len(np.unique(cols))} patch columns")
    for k, f in enumerate(folds):
        log(f"fold {k + 1}: train {len(f['train'])}, val {len(f['val'])}, test {len(f['test'])}, "
            f"test columns {f['test_cols'][0]}-{f['test_cols'][-1]}")
    log(f"parameters: U-Net {n_params(UNet())}, Attention U-Net {n_params(AttentionUNet())}")

    # ---- lambda selection on the validation block of each fold (seed 0, fresh models)
    lam_per_fold, lam_curve = [], {lam: [] for lam in LAMBDAS}
    for k, fold in enumerate(folds):
        best = (None, -1)
        for lam in LAMBDAS:
            m, _ = train_fresh("unet", X, Y, fold, SEEDS[0], lam, PHYSICS_WEIGHTS)
            _, f1 = val_threshold(predict(m, X[fold["val"]]), Y[fold["val"], 0])
            lam_curve[lam].append(f1)
            if f1 > best[1]:
                best = (lam, f1)
        lam_per_fold.append(best[0])
        log(f"fold {k + 1}: selected lambda = {best[0]} (validation F1 {best[1]:.3f})")

    zero = [0.0] * N_FOLDS
    configs = {
        "Baseline U-Net": ("unet", zero, PHYSICS_WEIGHTS),
        "Attention U-Net": ("attention", zero, PHYSICS_WEIGHTS),
        "Physics-guided U-Net": ("unet", lam_per_fold, PHYSICS_WEIGHTS),
    }
    ablations = {f"w/o {t} constraint": ("unet", lam_per_fold, {**PHYSICS_WEIGHTS, t: 0.0})
                 for t in PHYSICS_WEIGHTS}
    ablations["dip constraint only"] = ("unet", lam_per_fold, dict(dip=1.0))
    if not RUN_ABLATIONS:
        ablations = {}

    results = {}
    for name, (arch, lams, w) in {**configs, **ablations}.items():
        log(f"\n== {name}")
        results[name] = run_config(name, arch, lams, w, X, Y, folds, SEEDS)

    summ = {n: summarize(r, Y) for n, r in results.items()}

    # ---- paired tests on per-patch IoU (seed-averaged); Holm-corrected
    ref = "Physics-guided U-Net"
    comps = [n for n in summ if n != ref]
    tests = []
    for n in comps:
        a, b = summ[ref]["per_patch_iou"], summ[n]["per_patch_iou"]
        p = wilcoxon(a, b).pvalue if not np.allclose(a, b) else 1.0
        d_seed = [scores(results[ref]["counts"][s].sum(0))["iou"] - scores(results[n]["counts"][s].sum(0))["iou"]
                  for s in SEEDS]
        tests.append(dict(vs=n, median_diff=float(np.median(a - b)), p=float(p), cliffs_delta=cliffs_delta(a, b),
                          seeds_better=int(np.sum(np.array(d_seed) > 0)), n_seeds=len(SEEDS), n_patches=len(a)))
    for t, pa in zip(tests, holm([t["p"] for t in tests])):
        t["p_holm"] = float(pa)

    # ---- FCI / DRR on stitched held-out predictions
    struct = {}
    for name in configs:
        vals = []
        for s in SEEDS:
            stitched = np.zeros_like(seismic)
            binary = np.zeros(seismic.shape, bool)
            for k, fold in enumerate(folds):
                c0 = fold["test_cols"][0] * STRIDE
                c1 = seismic.shape[1] if k == N_FOLDS - 1 else fold["test_cols"][-1] * STRIDE + PATCH
                sub = predict_section(results[name]["models"][(s, k)].to(device), seismic[:, c0:c1])
                own1 = c1 if k == N_FOLDS - 1 else (fold["test_cols"][-1] + 1) * STRIDE
                own0 = 0 if k == 0 else c0
                stitched[:, own0:own1] = sub[:, own0 - c0:own1 - c0]
                binary[:, own0:own1] = sub[:, own0 - c0:own1 - c0] > results[name]["thr"][s][k]
                results[name]["models"][(s, k)].cpu()
            vals.append((fault_continuity_index(binary), diffraction_reflection_ratio(seismic, binary),
                         int(binary.sum())))
            if s == SEEDS[0]:
                np.save(os.path.join(OUT, f"heldout_prob_{name.split()[0].lower()}.npy"), stitched)
                np.save(os.path.join(OUT, f"heldout_binary_{name.split()[0].lower()}.npy"), binary)
        v = np.array(vals, float)
        struct[name] = dict(fci=(v[:, 0].mean(), v[:, 0].std(ddof=1) if len(v) > 1 else 0),
                            drr=(v[:, 1].mean(), v[:, 1].std(ddof=1) if len(v) > 1 else 0),
                            pixels=(v[:, 2].mean(), v[:, 2].std(ddof=1) if len(v) > 1 else 0))
    label_struct = dict(fci=fault_continuity_index(faults), drr=diffraction_reflection_ratio(seismic, faults))

    # ---- tables
    T = []
    T.append("% TABLE 1: pooled held-out results over 5 spatial folds (mean +/- s.d. over seeds; [95% patch-bootstrap CI])")
    T.append("% Method & Params & IoU & F1 & Precision & Recall & AP & ROC-AUC \\\\")
    for n in configs:
        s = summ[n]
        npar = n_params(AttentionUNet() if n.startswith("Attention") else UNet())
        T.append(f"{n} & {npar:,} & {fmt(s['iou'])} [{s['ci']['iou'][0]:.3f}, {s['ci']['iou'][1]:.3f}] & "
                 f"{fmt(s['f1'])} [{s['ci']['f1'][0]:.3f}, {s['ci']['f1'][1]:.3f}] & {fmt(s['precision'])} & "
                 f"{fmt(s['recall'])} & {fmt(s['ap'])} & {fmt(s['roc_auc'])} \\\\")
    T.append("\n% TABLE 2: per-fold IoU (mean +/- s.d. over seeds)")
    T.append("% Method & " + " & ".join(f"Fold {k + 1}" for k in range(N_FOLDS)) + " \\\\")
    for n in configs:
        per = np.array([[fs["iou"] for fs in results[n]["fold_scores"][s]] for s in SEEDS])
        T.append(n + " & " + " & ".join(f"{per[:, k].mean():.3f} $\\pm$ {per[:, k].std(ddof=1) if len(SEEDS) > 1 else 0:.3f}"
                                        for k in range(N_FOLDS)) + " \\\\")
    T.append("% selected lambda per fold: " + ", ".join(str(x) for x in lam_per_fold))
    T.append("\n% TABLE 3: ablation (pooled, mean +/- s.d. over seeds)")
    T.append("% Variant & IoU & F1 & Precision & Recall \\\\")
    for n in ["Baseline U-Net", "Physics-guided U-Net", *ablations]:  # noqa
        s = summ[n]
        T.append(f"{n} & {fmt(s['iou'])} & {fmt(s['f1'])} & {fmt(s['precision'])} & {fmt(s['recall'])} \\\\")
    T.append("\n% TABLE 4: structural metrics on stitched held-out predictions (mean +/- s.d. over seeds)")
    T.append("% Method & FCI & DRR & predicted fault pixels \\\\")
    T.append(f"Reference labels & {label_struct['fci']:.4f} & {label_struct['drr']:.3f} & {int(faults.sum())} \\\\")
    for n, v in struct.items():
        T.append(f"{n} & {v['fci'][0]:.4f} $\\pm$ {v['fci'][1]:.4f} & {v['drr'][0]:.3f} $\\pm$ {v['drr'][1]:.3f} & "
                 f"{v['pixels'][0]:.0f} \\\\")
    T.append("\n% TABLE 5: paired comparisons, Physics-guided U-Net vs X (per-patch IoU, seed-averaged)")
    T.append("% Comparison & median diff & Cliff delta & p (Wilcoxon) & p (Holm) & seeds better \\\\")
    for t in tests:
        T.append(f"vs {t['vs']} & {t['median_diff']:+.3f} & {t['cliffs_delta']:+.2f} & {t['p']:.2g} & "
                 f"{t['p_holm']:.2g} & {t['seeds_better']}/{t['n_seeds']} \\\\")
    T.append(f"% n test patches = {tests[0]['n_patches']}; mean early-stopping epoch: " +
             ", ".join(f"{n}: {summ[n]['epochs']:.1f}" for n in configs))
    open(os.path.join(OUT, "tables.tex"), "w").write("\n".join(T) + "\n")
    log("\n" + "\n".join(T))

    json.dump(dict(lambda_per_fold=lam_per_fold,
                   lambda_curve={str(k): v for k, v in lam_curve.items()},
                   summary={n: {k: v for k, v in s.items() if k != "per_patch_iou"} for n, s in summ.items()},
                   tests=tests, structural={n: {k: list(map(float, v)) for k, v in d.items()} for n, d in struct.items()},
                   label_structural=label_struct),
              open(os.path.join(OUT, "summary.json"), "w"), indent=2, default=float)

    # ---- figures
    y = (Y[:, 0] > 0.5).ravel()
    s0 = SEEDS[0]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for n in configs:
        p = results[n]["probs"][s0].ravel()
        pr, rc, _ = precision_recall_curve(y, p)
        fp, tp, _ = roc_curve(y, p)
        ax[0].plot(rc, pr, label=f"{n} (AP={summ[n]['ap'][0]:.3f})")
        ax[1].plot(fp, tp, label=f"{n} (AUC={summ[n]['roc_auc'][0]:.3f})")
    ax[1].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0].set(xlabel="Recall", ylabel="Precision", title="(a) Precision-recall, held-out patches")
    ax[1].set(xlabel="False positive rate", ylabel="True positive rate", title="(b) ROC, held-out patches")
    for a in ax:
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "Fig_PR_ROC.png"), dpi=300)
    plt.close(fig)

    fig, a = plt.subplots(figsize=(6, 4))
    lv = np.array([lam_curve[l] for l in LAMBDAS])
    a.errorbar(LAMBDAS, lv.mean(1), yerr=lv.std(1), marker="o", capsize=3)
    a.set(xscale="log", xlabel=r"$\lambda$", ylabel=r"Validation F1 (mean $\pm$ s.d. over folds)",
          title="Sensitivity to the physics weight")
    a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "Fig_lambda.png"), dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(len(configs) + 1, 1, figsize=(12, 3 * (len(configs) + 1)), sharex=True)
    ax[0].imshow(seismic, cmap="gray", aspect="auto", vmin=-3, vmax=3)
    ax[0].imshow(np.ma.masked_where(faults < 0.5, faults), cmap="autumn", aspect="auto")
    ax[0].set_title("Seismic section with reference labels")
    for a, n in zip(ax[1:], configs):
        bmap = np.load(os.path.join(OUT, f"heldout_binary_{n.split()[0].lower()}.npy"))
        a.imshow(seismic, cmap="gray", aspect="auto", vmin=-3, vmax=3)
        a.imshow(np.ma.masked_where(~bmap, bmap.astype(float)), cmap="autumn", aspect="auto", alpha=0.9)
        a.set_title(f"{n}: held-out fault picks at the validation threshold (seed {SEEDS[0]})")
    for k, f in enumerate(folds[1:], 1):
        for a in ax:
            a.axvline(f["test_cols"][0] * STRIDE, color="c", lw=0.6, ls="--")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "Fig_heldout_overlay.png"), dpi=200)
    plt.close(fig)
    log(f"\nDone. Tables: {OUT}/tables.tex  Figures: {OUT}/Fig_*.png")


if __name__ == "__main__":
    main()
