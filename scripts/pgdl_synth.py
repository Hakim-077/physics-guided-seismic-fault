# =============================================================================
# PHYSICS-GUIDED SEISMIC FAULT DETECTION -- SYNTHETIC-TRAINING / MARMOUSI-TEST
# (AJGS-D-26-00099, revision; design of Wu et al. 2019 and Pochet et al. 2019)
#
#   TRAIN + VALIDATE : randomly generated 2D faulted, folded layered sections
#                      with EXACT fault labels (never Marmousi).
#   TEST             : the full migrated Marmousi-II section, with fault labels
#                      derived from the Marmousi-II velocity model.
#
# Because training and test data are different, there is no train/test leakage.
# Thresholds and lambda are chosen on the synthetic validation set only, so
# nothing is tuned on Marmousi. The three models (U-Net, Attention U-Net,
# physics-guided U-Net) and the four physics terms are unchanged from the
# manuscript code.
#
# Key metric for the paper's hypothesis ("the dip term suppresses reflector
# leakage"): the Reflector Leakage Rate (RLR) = fraction of predicted fault
# pixels that lie on strong reflectors (top-5 % gradient pixels) farther than
# RLR_BUFFER samples from any labelled fault. Lower is better.
#
# Run in Colab (GPU):  !python pgdl_synth.py
# Quick pipeline check: set QUICK = True (minutes, numbers meaningless).
# =============================================================================
import os
import json
import random
import sys
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy import ndimage
from scipy.signal import fftconvolve
from scipy.stats import wilcoxon
from skimage.transform import resize
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

warnings.filterwarnings("ignore")

# =========================
# 0. CONFIGURATION
# =========================
SEGY_PATH = "SYNTHETIC.segy"            # migrated Marmousi-II section (samples x traces)
VP_PATH = "vp_marmousi-ii.segy"         # Marmousi-II P-wave velocity (m/s or km/s)
# Boxes (z0, z1, x0, x1) on the image grid where automatic labels are NOT faults
# (unconformity truncations seen in the label QC image). Edit after your own QC.
EXCLUDE_BOXES = [(610, 680, 420, 650), (550, 605, 1175, 1250)]
PATCH, NET_SIZE, INFER_STEP = 256, 128, 64
N_TRAIN, N_VAL = 3000, 400              # synthetic sections (256 x 256)
SEEDS = [0, 1, 2, 3, 4]
LAMBDAS = [0.01, 0.02, 0.05, 0.1, 0.2]
MAX_EPOCHS, PATIENCE, LR, BATCH = 40, 8, 1e-3, 16
TOL = 3                                  # tolerance (samples) for tolerance-based scores
RLR_BUFFER = 5                           # reflector pixels closer than this to a fault are ignored
EVAL_TILE = 128                          # statistical unit on the Marmousi section
N_BOOT = 2000
THR_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)
OUT = "results_synth"
RUN_ABLATIONS = False                    # True adds 5 ablation variants x 5 seeds (~2x runtime)
QUICK = False

if QUICK:
    N_TRAIN, N_VAL, SEEDS, LAMBDAS, MAX_EPOCHS, PATIENCE, N_BOOT = 96, 32, [0, 1], [0.05], 2, 2, 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "log.txt"), "w")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
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
# 1. SYNTHETIC TRAINING DATA
# =========================
def ricker(f):
    """Ricker wavelet, peak frequency f in cycles per sample."""
    n = int(2.5 / f) | 1
    t = np.arange(n) - n // 2
    a = (np.pi * f * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


def synthetic_section(rng, H=PATCH, W=PATCH):
    """One folded, faulted layered section with exact fault labels.
    Layers are defined by a relative-geologic-time field t(z, x); folding bends
    t, and each fault shifts t on its hanging-wall side by a throw that tapers
    to zero at the fault tips. Reflectivity = vertical derivative of log
    impedance, convolved with a Ricker wavelet; band-limited noise added."""
    zz, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    t = zz.copy()
    for _ in range(rng.integers(1, 4)):                         # folding
        amp, wl, ph = rng.uniform(0, 18), rng.uniform(0.6 * W, 3 * W), rng.uniform(0, 2 * np.pi)
        t -= amp * np.sin(2 * np.pi * xx / wl + ph) * (1 - rng.uniform(0, 0.8) * zz / H)
    t -= np.tan(np.deg2rad(rng.uniform(-25, 25))) * (xx - W / 2)  # regional dip

    lab = np.zeros((H, W), bool)
    for _ in range(rng.choice([0, 1, 1, 2, 2, 3, 3, 4])):     # 0-4 faults (some sections have none)
        x0, zc = rng.uniform(0.15 * W, 0.85 * W), rng.uniform(0.2 * H, 0.8 * H)
        dip = np.deg2rad(rng.uniform(45, 85))
        a = rng.choice([-1, 1]) / np.tan(dip)                    # dx/dz of the fault plane
        b = rng.uniform(0, 1.5e-3) * np.sign(a) * rng.choice([0, 1])  # optional listric curvature
        xf = x0 + a * (zz - zc) + b * (zz - zc) ** 2
        ztop = rng.uniform(-0.3 * H, 0.4 * H)
        zbot = ztop + rng.uniform(0.6 * H, 1.6 * H)
        taper = np.clip(np.minimum(zz - ztop, zbot - zz) / (0.15 * H), 0, 1)
        throw = rng.uniform(4, 25) * rng.choice([1, -1])        # normal or reverse
        t = t - np.where(xx > xf, throw * taper, 0.0)
        slope = a + 2 * b * (zz - zc)
        dist = np.abs(xx - xf) / np.sqrt(1 + slope ** 2)
        lab |= (dist < 1.5) & (np.abs(throw) * taper >= 2.0)   # label only where the offset is >= 2 samples

    # layer stack as a function of t, with occasional strong contrasts (Marmousi-like horizons)
    tmin, tmax = t.min() - 5, t.max() + 5
    edges = [tmin]
    while edges[-1] < tmax:
        edges.append(edges[-1] + rng.uniform(2, 14))
    edges = np.array(edges)
    li = np.cumsum(rng.normal(0, 0.03, len(edges)))
    strong = rng.random(len(edges)) < 0.08
    li += np.cumsum(np.where(strong, rng.choice([-1, 1], len(edges)) * rng.uniform(0.15, 0.4, len(edges)), 0))
    logimp = li[np.clip(np.searchsorted(edges, t) - 1, 0, len(li) - 1)]
    refl = np.zeros_like(logimp)
    refl[1:] = np.diff(logimp, axis=0)
    img = fftconvolve(refl, ricker(rng.uniform(0.06, 0.25))[:, None], mode="same")

    if rng.random() < 0.3:                                     # water layer at the top, as in Marmousi
        wz = int(rng.uniform(0.05, 0.35) * H)
        img[:wz] = 0.0
        lab[:wz + 3] = False
    img /= img.std() + 1e-8
    noise = ndimage.gaussian_filter(rng.standard_normal((H, W)), (rng.uniform(0.5, 2), rng.uniform(0.5, 2)))
    img += noise / (noise.std() + 1e-8) * rng.uniform(0, 0.5)
    img = img / (img.std() + 1e-8) * np.exp(rng.uniform(-0.7, 0.5))
    return np.clip(img, -4, 4).astype(np.float32), lab.astype(np.float32)


def to_net(a, is_label=False):
    """Same resize routine for synthetic patches and Marmousi inference."""
    if is_label:
        return (resize(a, (NET_SIZE, NET_SIZE), order=1, mode="constant", anti_aliasing=False) > 0.5).astype(np.float32)
    return resize(a, (NET_SIZE, NET_SIZE), order=1, mode="constant", anti_aliasing=True).astype(np.float32)


def make_synthetic_set(n, seed):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 1, NET_SIZE, NET_SIZE), np.float32)
    Y = np.zeros_like(X)
    for i in range(n):
        s, l = synthetic_section(rng)
        X[i, 0], Y[i, 0] = to_net(s), to_net(l, True)
    return X, Y


# =========================
# 2. MARMOUSI-II TEST DATA
# =========================
def load_section(path):
    if path.endswith(".npy"):
        return np.load(path).astype(np.float32)
    import segyio
    with segyio.open(path, "r", ignore_geometry=True) as f:
        return np.stack([f.trace[i] for i in range(f.tracecount)]).T.astype(np.float32)


def standardize(d):
    return np.clip((d - d.mean()) / d.std(), -4, 4).astype(np.float32)


def vp_on_grid(vp, shape):
    vp = np.asarray(vp, np.float64)
    if vp.max() < 100:
        log(f"Vp range {vp.min():.3f}-{vp.max():.3f}: km/s -> m/s")
        vp = vp * 1000.0
    if vp.shape != shape:
        log(f"Resampling Vp {vp.shape} -> {shape}")
        vp = resize(vp, shape, order=1, preserve_range=True, anti_aliasing=True)
    return vp


def labels_from_velocity(vp, sigma_d=1.5, sigma_i=10.0, low=0.35, high=0.8, min_angle_deg=25.0):
    """Discontinuity of log(Vp) along the local bedding (structure-tensor tangent)."""
    from skimage.filters import apply_hysteresis_threshold
    from skimage.measure import label as cc_label, regionprops
    from skimage.morphology import skeletonize, disk
    lv = np.log(np.maximum(vp, 1.0))
    gz = ndimage.gaussian_filter(lv, sigma_d, order=(1, 0))
    gx = ndimage.gaussian_filter(lv, sigma_d, order=(0, 1))
    jzz, jxx, jzx = (ndimage.gaussian_filter(a, sigma_i) for a in (gz * gz, gx * gx, gz * gx))
    th = 0.5 * np.arctan2(2 * jzx, jzz - jxx)
    nz, nx = np.cos(th), np.sin(th)
    tang = np.abs(gz * (-nx) + gx * nz)
    valid = ndimage.binary_erosion(vp > 1550, iterations=10, border_value=1)
    if not valid.any():
        raise ValueError("No sub-seafloor samples (Vp > 1550 m/s); check the Vp file.")
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
        tv = np.array([-nx[rr, cc].mean(), nz[rr, cc].mean()])
        if abs(axis @ (tv / (np.linalg.norm(tv) + 1e-12))) < cmin:
            keep[lab == r.label] = True
    out = ndimage.binary_dilation(skeletonize(keep), structure=disk(1)) & valid
    for z0, z1, x0, x1 in EXCLUDE_BOXES:
        out[z0:z1, x0:x1] = False
    return out, valid


def reflector_mask(seismic, faults):
    """Strong reflectors = top 5 % of the image gradient, excluding a buffer around faults."""
    gz, gx = np.gradient(seismic)
    g = np.sqrt(gx ** 2 + gz ** 2)
    refl = g > np.percentile(g, 95)
    near = ndimage.distance_transform_edt(~faults) <= RLR_BUFFER
    return refl & ~near


# =========================
# 3. MODELS AND PHYSICS TERMS (unchanged from the manuscript code)
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
# 4. TRAINING, PREDICTION, METRICS
# =========================
def train_fresh(arch, Xtr, Ytr, Xva, Yva, seed, lam, weights):
    set_seed(seed)
    model = (AttentionUNet if arch == "attention" else UNet)().to(device)
    opt = torch.optim.Adam(model.parameters(), LR)
    bce = nn.BCELoss()
    g = torch.Generator()
    g.manual_seed(seed)
    tl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr)), BATCH, shuffle=True, generator=g)
    vl = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(Yva)), 64, shuffle=False)
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
    out = [model(torch.from_numpy(X[i:i + 64]).to(device)).cpu().numpy() for i in range(0, len(X), 64)]
    return np.concatenate(out)[:, 0]


def counts(pred, y):
    pb, yb = pred.astype(bool), y > 0.5
    return np.array([(pb & yb).sum(), (pb & ~yb).sum(), (~pb & yb).sum()], dtype=np.float64)


def scores(c):
    tp, fp, fn = c
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    return dict(iou=tp / (tp + fp + fn + 1e-8), precision=p, recall=r, f1=2 * p * r / (p + r + 1e-8))


def val_threshold(prob, y):
    f1 = [scores(counts(prob > t, y))["f1"] for t in THR_GRID]
    i = int(np.argmax(f1))
    return float(THR_GRID[i]), float(f1[i])


def tolerant_counts(pred, lab, valid):
    """(tp_pred, n_pred, tp_lab, n_lab) with distance tolerance TOL."""
    pred = pred & valid
    lab = lab & valid
    d_lab = ndimage.distance_transform_edt(~lab) if lab.any() else np.full(lab.shape, np.inf)
    d_pr = ndimage.distance_transform_edt(~pred) if pred.any() else np.full(pred.shape, np.inf)
    return np.array([(pred & (d_lab <= TOL)).sum(), pred.sum(), (lab & (d_pr <= TOL)).sum(), lab.sum()], float)


def tol_scores(c):
    tpp, npred, tpl, nlab = c
    p = tpp / npred if npred else 0.0
    r = tpl / nlab if nlab else 0.0
    return dict(tol_precision=p, tol_recall=r, tol_f1=2 * p * r / (p + r) if p + r else 0.0)


@torch.no_grad()
def predict_section(model, sec):
    H, W = sec.shape
    rs = list(range(0, H - PATCH + 1, INFER_STEP))
    cs = list(range(0, W - PATCH + 1, INFER_STEP))
    if rs[-1] != H - PATCH:
        rs.append(H - PATCH)
    if cs[-1] != W - PATCH:
        cs.append(W - PATCH)
    acc, cnt = np.zeros((H, W)), np.zeros((H, W))
    w = np.outer(np.hanning(PATCH + 2)[1:-1], np.hanning(PATCH + 2)[1:-1]) + 1e-3
    for r in rs:
        tiles = np.stack([to_net(sec[r:r + PATCH, c:c + PATCH]) for c in cs])[:, None]
        for c, p in zip(cs, predict(model, tiles)):
            acc[r:r + PATCH, c:c + PATCH] += resize(p, (PATCH, PATCH), order=1) * w
            cnt[r:r + PATCH, c:c + PATCH] += w
    return (acc / np.maximum(cnt, 1e-6)).astype(np.float32)


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


def cliffs_delta(a, b):
    a, b = np.asarray(a)[:, None], np.asarray(b)[None, :]
    return float(((a > b).sum() - (a < b).sum()) / (a.size * b.size))


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj, run = np.empty_like(p), 0.0
    for k, i in enumerate(o):
        run = max(run, (len(p) - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def ms(v):
    v = np.asarray(v, float)
    return (float(np.nanmean(v)), float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0)


def fmt(m, d=3):
    return f"{m[0]:.{d}f} $\\pm$ {m[1]:.{d}f}"


# =========================
# 5. MAIN
# =========================
def main():
    t0 = time.time()
    log(f"device={device} QUICK={QUICK} seeds={SEEDS} N_TRAIN={N_TRAIN} N_VAL={N_VAL}")

    # ---- Marmousi test data
    seismic = standardize(load_section(SEGY_PATH))
    vp = vp_on_grid(load_section(VP_PATH), seismic.shape)
    faults, valid = labels_from_velocity(vp)
    refl = reflector_mask(seismic, faults) & valid
    log(f"Marmousi section {seismic.shape}; fault fraction {faults.mean():.4f}; "
        f"reflector-mask fraction {refl.mean():.4f}")
    H, W = seismic.shape
    tiles = [(z, x) for z in range(0, H - EVAL_TILE + 1, EVAL_TILE) for x in range(0, W - EVAL_TILE + 1, EVAL_TILE)]
    fault_tiles = [i for i, (z, x) in enumerate(tiles) if faults[z:z + EVAL_TILE, x:x + EVAL_TILE].sum() >= 20]
    log(f"{len(tiles)} evaluation tiles ({EVAL_TILE} px), {len(fault_tiles)} contain labelled faults")

    # ---- synthetic data
    Xtr, Ytr = make_synthetic_set(N_TRAIN, 12345)
    Xva, Yva = make_synthetic_set(N_VAL, 67890)
    log(f"synthetic train {Xtr.shape}, val {Xva.shape}; fault fraction {Ytr.mean():.4f} / {Yva.mean():.4f} "
        f"({time.time() - t0:.0f}s)")
    fig, ax = plt.subplots(2, 6, figsize=(15, 5.4))
    for i in range(6):
        ax[0, i].imshow(Xtr[i, 0], cmap="gray", vmin=-3, vmax=3)
        ax[1, i].imshow(Xtr[i, 0], cmap="gray", vmin=-3, vmax=3)
        ax[1, i].imshow(np.ma.masked_where(Ytr[i, 0] < 0.5, Ytr[i, 0]), cmap="autumn")
        for a in ax[:, i]:
            a.axis("off")
    fig.suptitle("Synthetic training examples (top) with exact fault labels (bottom)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "Fig_synthetic_examples.png"), dpi=200)
    plt.close(fig)

    # ---- lambda selection on synthetic validation (seed 0)
    lam_curve = {}
    for lam in LAMBDAS:
        m, _ = train_fresh("unet", Xtr, Ytr, Xva, Yva, SEEDS[0], lam, PHYSICS_WEIGHTS)
        lam_curve[lam] = val_threshold(predict(m, Xva), Yva[:, 0])[1]
        log(f"lambda {lam}: synthetic validation F1 {lam_curve[lam]:.3f}")
    lam_star = max(lam_curve, key=lam_curve.get)
    log(f"selected lambda = {lam_star}")

    configs = {"Baseline U-Net": ("unet", 0.0, PHYSICS_WEIGHTS),
               "Attention U-Net": ("attention", 0.0, PHYSICS_WEIGHTS),
               "Physics-guided U-Net": ("unet", lam_star, PHYSICS_WEIGHTS)}
    ablations = {}
    if RUN_ABLATIONS:
        ablations = {f"w/o {t} constraint": ("unet", lam_star, {**PHYSICS_WEIGHTS, t: 0.0}) for t in PHYSICS_WEIGHTS}
        ablations["dip constraint only"] = ("unet", lam_star, dict(dip=1.0))

    R = {}
    for name, (arch, lam, w) in {**configs, **ablations}.items():
        R[name] = dict(val_f1=[], thr=[], ep=[], m={}, tile_tol=[], tile_f1=[], prob=[])
        for s in SEEDS:
            model, ep = train_fresh(arch, Xtr, Ytr, Xva, Yva, s, lam, w)
            thr, vf1 = val_threshold(predict(model, Xva), Yva[:, 0])
            prob = predict_section(model, seismic)
            pred = (prob > thr) & valid
            c = counts(pred, faults & valid)
            tc = tolerant_counts(pred, faults, valid)
            vm = valid.ravel()
            m = {**scores(c), **tol_scores(tc),
                 "ap": average_precision_score(faults.ravel()[vm], prob.ravel()[vm]),
                 "roc_auc": roc_auc_score(faults.ravel()[vm], prob.ravel()[vm]),
                 "rlr": float((pred & refl).sum() / max(pred.sum(), 1)),
                 "fci": fault_continuity_index(pred), "drr": diffraction_reflection_ratio(seismic, pred),
                 "pixels": int(pred.sum())}
            for k, v in m.items():
                R[name]["m"].setdefault(k, []).append(v)
            tt = np.array([tolerant_counts(pred[z:z + EVAL_TILE, x:x + EVAL_TILE], faults[z:z + EVAL_TILE, x:x + EVAL_TILE],
                                           valid[z:z + EVAL_TILE, x:x + EVAL_TILE]) for z, x in tiles])
            R[name]["tile_tol"].append(tt)
            R[name]["val_f1"].append(vf1)
            R[name]["thr"].append(thr)
            R[name]["ep"].append(ep)
            if s == SEEDS[0]:
                R[name]["prob"] = prob
                R[name]["pred"] = pred
            log(f"  {name:28s} seed {s}: synth-val F1 {vf1:.3f} thr {thr:.2f} ep {ep} | Marmousi tol-F1 "
                f"{m['tol_f1']:.3f} (P {m['tol_precision']:.3f} R {m['tol_recall']:.3f}) IoU {m['iou']:.3f} "
                f"AP {m['ap']:.3f} RLR {m['rlr']:.3f} ({time.time() - t0:.0f}s)")

    # ---- per-tile statistics (tiles containing labelled faults), seed-averaged
    def tile_metric(name, key):
        vals = []
        for tt in R[name]["tile_tol"]:
            vals.append([tol_scores(tt[i])[key] for i in fault_tiles])
        return np.mean(vals, 0)

    ref = "Physics-guided U-Net"
    tests = []
    for n in [k for k in R if k != ref]:
        a, b = tile_metric(ref, "tol_f1"), tile_metric(n, "tol_f1")
        p = wilcoxon(a, b).pvalue if len(a) > 5 and not np.allclose(a, b) else float("nan")
        seeds_better = int(sum(x > y for x, y in zip(R[ref]["m"]["tol_f1"], R[n]["m"]["tol_f1"])))
        rlr_better = int(sum(x < y for x, y in zip(R[ref]["m"]["rlr"], R[n]["m"]["rlr"])))
        tests.append(dict(vs=n, median_diff=float(np.median(a - b)), cliffs_delta=cliffs_delta(a, b), p=float(p),
                          seeds_better_f1=seeds_better, seeds_lower_rlr=rlr_better, n_tiles=len(a)))
    ps = [t["p"] for t in tests]
    adj = holm([x if np.isfinite(x) else 1.0 for x in ps])
    for t, pa in zip(tests, adj):
        t["p_holm"] = float(pa)

    # ---- tables
    T = ["% TABLE A: Marmousi-II test (trained on synthetics only; threshold from synthetic validation)",
         "% mean +/- s.d. over seeds. tol = tolerance-based (" + str(TOL) + " samples)",
         "% Method & Params & tol-P & tol-R & tol-F1 & IoU (pixel) & AP & ROC-AUC \\\\"]
    for n in configs:
        m = R[n]["m"]
        npar = n_params(AttentionUNet() if n.startswith("Attention") else UNet())
        T.append(f"{n} & {npar:,} & {fmt(ms(m['tol_precision']))} & {fmt(ms(m['tol_recall']))} & "
                 f"{fmt(ms(m['tol_f1']))} & {fmt(ms(m['iou']))} & {fmt(ms(m['ap']))} & {fmt(ms(m['roc_auc']))} \\\\")
    T += ["", "% TABLE B: reflector leakage and structural metrics on Marmousi-II (mean +/- s.d. over seeds)",
          "% Method & RLR (lower=better) & FCI & DRR & predicted pixels \\\\",
          f"Reference labels & -- & {fault_continuity_index(faults):.4f} & "
          f"{diffraction_reflection_ratio(seismic, faults):.3f} & {int(faults.sum())} \\\\"]
    for n in R:
        m = R[n]["m"]
        T.append(f"{n} & {fmt(ms(m['rlr']))} & {fmt(ms(m['fci']), 4)} & {fmt(ms(m['drr']))} & {ms(m['pixels'])[0]:.0f} \\\\")
    T += ["", "% TABLE C: synthetic validation F1 (in-domain), threshold, epochs",
          "% Method & synthetic val F1 & threshold & early-stopping epoch \\\\"]
    for n in R:
        T.append(f"{n} & {fmt(ms(R[n]['val_f1']))} & {ms(R[n]['thr'])[0]:.2f} & {ms(R[n]['ep'])[0]:.1f} \\\\")
    if ablations:
        T += ["", "% TABLE D: ablation on Marmousi-II", "% Variant & tol-F1 & tol-P & tol-R & RLR \\\\"]
        for n in ["Baseline U-Net", "Physics-guided U-Net", *ablations]:
            m = R[n]["m"]
            T.append(f"{n} & {fmt(ms(m['tol_f1']))} & {fmt(ms(m['tol_precision']))} & {fmt(ms(m['tol_recall']))} & "
                     f"{fmt(ms(m['rlr']))} \\\\")
    T += ["", f"% TABLE E: paired comparisons, Physics-guided U-Net vs X (per-tile tolerance F1 on {len(fault_tiles)} "
              "Marmousi tiles with faults, seed-averaged)",
          "% Comparison & median diff & Cliff delta & p (Wilcoxon) & p (Holm) & seeds better F1 & seeds lower RLR \\\\"]
    for t in tests:
        T.append(f"vs {t['vs']} & {t['median_diff']:+.3f} & {t['cliffs_delta']:+.2f} & {t['p']:.2g} & "
                 f"{t['p_holm']:.2g} & {t['seeds_better_f1']}/{len(SEEDS)} & {t['seeds_lower_rlr']}/{len(SEEDS)} \\\\")
    T.append(f"% selected lambda = {lam_star}; lambda curve (synthetic val F1): " +
             ", ".join(f"{k}: {v:.3f}" for k, v in lam_curve.items()))
    open(os.path.join(OUT, "tables.tex"), "w").write("\n".join(T) + "\n")
    log("\n" + "\n".join(T))
    json.dump(dict(lambda_curve={str(k): v for k, v in lam_curve.items()}, lambda_selected=lam_star,
                   metrics={n: R[n]["m"] for n in R}, tests=tests),
              open(os.path.join(OUT, "summary.json"), "w"), indent=2, default=float)

    # ---- figures
    fig, ax = plt.subplots(len(configs) + 1, 1, figsize=(14, 3.2 * (len(configs) + 1)), sharex=True)
    ax[0].imshow(seismic, cmap="gray", aspect="auto", vmin=-3, vmax=3)
    ax[0].imshow(np.ma.masked_where(~faults, faults.astype(float)), cmap="autumn", aspect="auto")
    ax[0].set_title("Marmousi-II migrated section with velocity-derived fault labels")
    for a, n in zip(ax[1:], configs):
        a.imshow(seismic, cmap="gray", aspect="auto", vmin=-3, vmax=3)
        pr = R[n]["pred"]
        a.imshow(np.ma.masked_where(~(pr & refl), (pr & refl).astype(float)), cmap="cool", aspect="auto")
        a.imshow(np.ma.masked_where(~(pr & ~refl), (pr & ~refl).astype(float)), cmap="autumn", aspect="auto")
        a.set_title(f"{n} (seed {SEEDS[0]}): fault picks; magenta = picks on strong reflectors (leakage)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "Fig_marmousi_predictions.png"), dpi=200)
    plt.close(fig)

    fig, a = plt.subplots(figsize=(6.5, 5))
    vm = valid.ravel()
    for n in configs:
        pr_, rc_, _ = precision_recall_curve(faults.ravel()[vm], R[n]["prob"].ravel()[vm])
        a.plot(rc_, pr_, label=f"{n} (AP={ms(R[n]['m']['ap'])[0]:.3f})")
    a.set(xlabel="Recall", ylabel="Precision", title="Marmousi-II (pixel level, seed 0)")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "Fig_marmousi_PR.png"), dpi=300)
    plt.close(fig)

    if len(LAMBDAS) > 1:
        fig, a = plt.subplots(figsize=(6, 4))
        a.plot(LAMBDAS, [lam_curve[l] for l in LAMBDAS], marker="o")
        a.set(xscale="log", xlabel="lambda", ylabel="Synthetic validation F1", title="Sensitivity to the physics weight")
        a.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "Fig_lambda.png"), dpi=300)
        plt.close(fig)
    log(f"\nDone in {time.time() - t0:.0f}s. Tables: {OUT}/tables.tex  Figures: {OUT}/Fig_*.png")


def preview():
    """python pgdl_synth.py --preview : saves synthetic examples and the Marmousi label/reflector masks only."""
    Xs, Ys = make_synthetic_set(12, 1)
    fig, ax = plt.subplots(2, 6, figsize=(15, 5.4))
    for i in range(6):
        ax[0, i].imshow(Xs[i, 0], cmap="gray", vmin=-3, vmax=3)
        ax[1, i].imshow(Xs[i, 0], cmap="gray", vmin=-3, vmax=3)
        ax[1, i].imshow(np.ma.masked_where(Ys[i, 0] < 0.5, Ys[i, 0]), cmap="autumn")
        for a in ax[:, i]:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "Fig_synthetic_examples.png"), dpi=150)
    plt.close(fig)
    seismic = standardize(load_section(SEGY_PATH))
    vp = vp_on_grid(load_section(VP_PATH), seismic.shape)
    faults, valid = labels_from_velocity(vp)
    refl = reflector_mask(seismic, faults) & valid
    fig, a = plt.subplots(figsize=(16, 5))
    a.imshow(seismic, cmap="gray", aspect="auto", vmin=-3, vmax=3)
    a.imshow(np.ma.masked_where(~refl, refl.astype(float)), cmap="cool", aspect="auto", alpha=0.6)
    a.imshow(np.ma.masked_where(~faults, faults.astype(float)), cmap="autumn", aspect="auto")
    a.set_title("Test labels (red: faults) and reflector mask used for the leakage rate (magenta)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "Fig_marmousi_masks.png"), dpi=150)
    plt.close(fig)
    log(f"fault fraction {faults.mean():.4f}; reflector-mask fraction {refl.mean():.4f}; saved previews in {OUT}/")


if __name__ == "__main__":
    if "--preview" in sys.argv:
        preview()
    else:
        main()
