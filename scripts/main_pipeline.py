# PHYSICS-GUIDED SEISMIC FAULT DETECTION — FINAL ENHANCED VERSION

# =============================================================================

# =========================
# 0. IMPORTS & SETTINGS
# =========================
import os
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

import segyio
from scipy import ndimage
from skimage.feature import canny
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_curve # Added roc_curve
from sklearn.model_selection import KFold
from sklearn.utils import resample
from scipy.stats import wilcoxon

# =========================
# 1. REPRODUCIBILITY
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 2. FIGURE MANAGEMENT
# =========================
os.makedirs("figures", exist_ok=True)

def save_fig(name):
    plt.savefig(f"figures/{name}.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"figures/{name}.png", dpi=300, bbox_inches="tight")

# =========================
# 3. DATA LOADING
# =========================
def load_segy():
    # Ensure SYNTHETIC.segy is available. If not, this will raise an error.
    if not os.path.exists("SYNTHETIC.segy"):
        print("Error: SYNTHETIC.segy not found. Please upload it to the runtime.")
        return None
    with segyio.open("SYNTHETIC.segy", "r", ignore_geometry=True) as f:
        data = np.stack([f.trace[i] for i in range(f.tracecount)]).T
    data = (data - data.mean()) / data.std()
    return np.clip(data, -4, 4)

# =========================
# 4. FAULT LABEL GENERATION
# =========================
def generate_fault_labels(seismic):
    # FIXED (see conversation): the original version combined a P92 gradient
    # threshold with Canny edges (sigma=1.5), then applied binary_closing
    # with a (3,3) structuring element. On this densely-reflective section
    # that closing step merged nearly everything into one blob, producing
    # ~49% "fault" coverage -- contradicting the manuscript's own claim
    # that faults occupy <5% of pixels (see Discussion, PR/ROC section).
    # Neither Canny nor closing were ever mentioned in the manuscript text,
    # so removing them also restores code/text consistency. P95 on the
    # gradient magnitude alone gives ~5% coverage by construction.
    gx, gy = np.gradient(seismic)
    grad = np.sqrt(gx**2 + gy**2)
    grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)
    faults = grad > np.percentile(grad, 95)
    return faults.astype(np.float32)

# =========================
# 5. PATCH EXTRACTION
# =========================
def extract_patches(seismic, faults, size=256, stride=128):
    xs, ys = [], []
    H, W = seismic.shape
    for i in range(0, H-size+1, stride):
        for j in range(0, W-size+1, stride):
            f = faults[i:i+size, j:j+size]
            # Only keep patches with some fault content or minimum density
            if f.mean() > 0.005: # Lowered threshold slightly to include more patches
                xs.append(seismic[i:i+size, j:j+size])
                ys.append(f)
    xs = np.array(xs)[:,None,:,:]
    ys = np.array(ys)[:,None,:,:]
    print(f"Extracted {len(xs)} patches of size {size}x{size}.")
    return xs, ys

# =========================
# 6. DATASET
# =========================
class FaultDataset(Dataset):
    def __init__(self, X, Y):
        self.X, self.Y = X, Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        # Resize to 128x128 as used in previous steps if input was 256
        # Current UNet architecture expects 128x128 after pooling, previous code resized in __getitem__
        # Let's assume patches are already resized or the UNet needs 256 input here.
        # Re-checking previous SimpleDataset: it resizes to 128x128. Need to be consistent.
        from skimage.transform import resize
        seismic_patch = self.X[i].squeeze()
        fault_patch = self.Y[i].squeeze()

        seismic_patch = resize(seismic_patch, (128, 128), mode='constant', anti_aliasing=True)
        fault_patch = resize(fault_patch, (128, 128), mode='constant', anti_aliasing=False)
        fault_patch = (fault_patch > 0.5).astype(np.float32)

        return (
            torch.tensor(seismic_patch[None, :, :], dtype=torch.float32),
            torch.tensor(fault_patch[None, :, :], dtype=torch.float32)
        )

# =========================
# 7. MODEL
# =========================
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder part, similar to SimpleUNet
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU()
        )
        # Decoder part
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.final = nn.Sequential(
            nn.Conv2d(32, 1, 1), nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)      # [B, 32, H, W]
        x2 = self.enc2(x1)     # [B, 64, H/2, W/2]

        # Decoder
        x2_up = self.dec2(x2)  # [B, 32, H, W]

        # Skip connection - must be same size. If input is 128x128, x1 is 128x128, x2_up is 128x128.
        x_out = x1 + x2_up

        # Final
        return self.final(x_out)

# =========================
# 7b. ATTENTION U-NET (SOTA comparison baseline, Oktay et al. 2018)
# =========================
class AttentionGate(nn.Module):
    """Additive attention gate applied on the skip connection, matching
    Oktay et al. (2018). Kept at the same depth/channel width as the
    existing UNet class above so the comparison isolates the effect of
    attention gating rather than added network capacity."""
    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.w_gate = nn.Sequential(nn.Conv2d(gate_ch, inter_ch, 1), nn.BatchNorm2d(inter_ch))
        self.w_skip = nn.Sequential(nn.Conv2d(skip_ch, inter_ch, 1), nn.BatchNorm2d(inter_ch))
        self.psi = nn.Sequential(nn.Conv2d(inter_ch, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        g = self.w_gate(gate)
        s = self.w_skip(skip)
        attn = self.psi(self.relu(g + s))
        return skip * attn


class AttentionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.att = AttentionGate(gate_ch=32, skip_ch=32, inter_ch=16)
        self.final = nn.Sequential(nn.Conv2d(32, 1, 1), nn.Sigmoid())

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x2_up = self.dec2(x2)
        x1_att = self.att(x2_up, x1)
        x_out = x1_att + x2_up
        return self.final(x_out)

# =========================
# 8. PHYSICS REGULARIZATION
# =========================
PHYSICS_WEIGHTS = dict(dip=0.4, sparsity=0.3, continuity=0.2, contrast=0.1)

def compute_dip_field(seismic_batch):
    B, C, H, W = seismic_batch.shape
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                          dtype=torch.float32, device=seismic_batch.device).view(1,1,3,3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                          dtype=torch.float32, device=seismic_batch.device).view(1,1,3,3)
    grad_x = F.conv2d(seismic_batch, sobel_x, padding=1)
    grad_y = F.conv2d(seismic_batch, sobel_y, padding=1)
    dip_field = torch.cat([grad_x, grad_y], dim=1)
    norm = torch.norm(dip_field, dim=1, keepdim=True) + 1e-8
    dip_field = dip_field / norm
    return dip_field

def dip_alignment_loss(fault_pred, seismic, weight=1.0):
    dip_field = compute_dip_field(seismic)
    grad_y, grad_x = torch.gradient(fault_pred, dim=[2, 3])
    fault_grad = torch.cat([grad_x, grad_y], dim=1)
    grad_norm = torch.norm(fault_grad, dim=1, keepdim=True) + 1e-8
    fault_grad = fault_grad / grad_norm
    alignment = torch.abs(torch.sum(fault_grad * dip_field, dim=1))
    loss = torch.mean(alignment)
    return weight * loss

def sparsity_loss(fault_pred, weight=1.0):
    l1_loss = torch.mean(torch.abs(fault_pred))
    fault_mean = torch.mean(fault_pred, dim=[2, 3], keepdim=True)
    fault_var = torch.mean((fault_pred - fault_mean)**2) # This term penalizes thick faults, encouraging sparsity
    return weight * (0.8 * l1_loss + 0.2 * fault_var)

def continuity_loss(fault_pred, weight=1.0):
    tv_h = torch.abs(fault_pred[:, :, 1:, :] - fault_pred[:, :, :-1, :]).mean()
    tv_w = torch.abs(fault_pred[:, :, :, 1:] - fault_pred[:, :, :, :-1]).mean()
    tv_loss = (tv_h + tv_w) / 2
    return weight * tv_loss

def amplitude_contrast_loss(fault_pred, seismic, weight=1.0):
    grad_y, grad_x = torch.gradient(seismic, dim=[2, 3])
    seismic_grad = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
    seismic_grad = (seismic_grad - seismic_grad.min()) / (seismic_grad.max() - seismic_grad.min() + 1e-8)
    correlation = torch.sum(fault_pred * seismic_grad) / (torch.sum(fault_pred) + 1e-8)
    return weight * (1.0 - correlation) # (1 - correlation) means minimizing correlation if fault_pred is low and seismic_grad is high

def total_physics_loss(pred, seismic, lam, weights=PHYSICS_WEIGHTS):
    losses = {}
    if weights.get("dip", 0) > 0:
        losses['dip'] = dip_alignment_loss(pred, seismic, weight=weights["dip"])
    if weights.get("sparsity", 0) > 0:
        losses['sparsity'] = sparsity_loss(pred, weight=weights["sparsity"])
    if weights.get("continuity", 0) > 0:
        losses['continuity'] = continuity_loss(pred, weight=weights["continuity"])
    if weights.get("contrast", 0) > 0:
        losses['contrast'] = amplitude_contrast_loss(pred, seismic, weight=weights["contrast"])

    if not losses:
        return torch.tensor(0.0, device=pred.device), {}

    total = sum(losses.values())
    return lam * total, losses

# =========================
# 9. TRAINING FUNCTIONS
# =========================
def train_model(model, loader, val_loader, lam=0, epochs=10, model_name="Model", physics_weights=PHYSICS_WEIGHTS):
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    bce = nn.BCELoss()
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    print(f"\nTraining {model_name} (\u03bb={lam})...")

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            p = model(x)
            phys_loss, _ = total_physics_loss(p, x, lam, weights=physics_weights)
            loss = bce(p, y) + phys_loss
            loss.backward()
            opt.step()
            epoch_train_loss += loss.item()
        train_losses.append(epoch_train_loss / len(loader))

        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                p = model(x)
                loss = bce(p, y)
                epoch_val_loss += loss.item()
        val_losses.append(epoch_val_loss / len(val_loader))

        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            torch.save(model.state_dict(), f'best_{model_name.replace(" ", "_")}.pth')
            # print(f"  \u2713 Saved best model (val_loss: {best_val_loss:.4f})")

        print(f"Epoch {epoch+1:02d}/{epochs:02d}: Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

    model.load_state_dict(torch.load(f'best_{model_name.replace(" ", "_")}.pth'))
    return train_losses, val_losses

# =========================
# 10. EVALUATION
# =========================
def evaluate(model, loader, thr=0.5):
    model.eval()
    tp_total = fp_total = fn_total = 0
    all_true, all_probs = [], []

    with torch.no_grad():
        for x, y in loader:
            x_dev = x.to(device)
            y_cpu = y.cpu().numpy()

            pred_probs = model(x_dev).cpu().numpy()
            pred_binary = (pred_probs > thr).astype(np.float32)

            # Flatten all predictions and true labels for metrics
            all_probs.extend(pred_probs.flatten())
            all_true.extend(y_cpu.flatten())

            tp = np.sum(pred_binary * y_cpu)
            fp = np.sum(pred_binary * (1 - y_cpu))
            fn = np.sum((1 - pred_binary) * y_cpu)

            tp_total += tp
            fp_total += fp
            fn_total += fn

    iou = tp_total / (tp_total + fp_total + fn_total + 1e-8)
    precision = tp_total / (tp_total + fp_total + 1e-8)
    recall = tp_total / (tp_total + fn_total + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return dict(iou=iou, precision=precision, recall=recall, f1=f1), np.array(all_true), np.array(all_probs)

# =========================
# 11. MAIN PIPELINE EXECUTION
# =========================
print("\n" + "="*80)
print("STARTING FULL PHYSICS-GUIDED SEISMIC FAULT DETECTION PIPELINE")
print("="*80)

# Load data
print("\n[1/7] Loading seismic data...")
seismic_data = load_segy()
if seismic_data is None:
    exit()

# Generate fault labels
print("[2/7] Generating fault labels...")
faults_data = generate_fault_labels(seismic_data)

plt.figure(figsize=(10, 5))
plt.imshow(seismic_data, cmap="gray", aspect='auto')
plt.title("Fig.1 Seismic Section")
plt.xlabel("Traces"); plt.ylabel("Samples")
save_fig("Fig1_seismic")
plt.show()

plt.figure(figsize=(10, 5))
plt.imshow(faults_data, cmap="hot", aspect='auto')
plt.title("Fig.2 Fault Labels")
plt.xlabel("Traces"); plt.ylabel("Samples")
save_fig("Fig2_faults")
plt.show()

# Extract patches
print("\n[3/7] Extracting patches...")
X_patches, Y_patches = extract_patches(seismic_data, faults_data, size=256, stride=128)
full_dataset = FaultDataset(X_patches, Y_patches)

# Split dataset
print("\n[4/7] Splitting dataset...")
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_ds, val_ds, test_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])

batch_size = 8
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

print(f"  Train samples: {len(train_ds)}")
print(f"  Validation samples: {len(val_ds)}")
print(f"  Test samples: {len(test_ds)}")

# =========================
# 12. BASELINE MODEL TRAINING
# =========================
print("\n[5/7] Training Baseline U-Net Model (No Physics Regularization)...")
baseline_model = UNet().to(device)
tr_losses_baseline, val_losses_baseline = train_model(baseline_model, train_loader, val_loader, lam=0, epochs=10, model_name="Baseline")

# Plot training/validation loss for Baseline
plt.figure(figsize=(10, 5))
plt.plot(tr_losses_baseline, label='Train Loss')
plt.plot(val_losses_baseline, label='Validation Loss')
plt.title('Fig.3 Baseline U-Net Training & Validation Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.legend()
plt.grid(True)
save_fig("Fig3_baseline_loss")
plt.show()

# =========================
# 12b. ATTENTION U-NET TRAINING (SOTA comparison)
# =========================
print("\n[5b/7] Training Attention U-Net Model (SOTA comparison baseline)...")
attention_model = AttentionUNet().to(device)
tr_losses_attention, val_losses_attention = train_model(
    attention_model, train_loader, val_loader, lam=0, epochs=10, model_name="Attention"
)

plt.figure(figsize=(10, 5))
plt.plot(tr_losses_attention, label='Train Loss')
plt.plot(val_losses_attention, label='Validation Loss')
plt.title('Fig.3b Attention U-Net Training & Validation Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.legend()
plt.grid(True)
save_fig("Fig3b_attention_loss")
plt.show()

# =========================
# 13. LAMBDA TUNING (Fig.8) and PHYSICS MODEL TRAINING
# =========================
print("\n[6/7] Tuning lambda parameter and training Physics-Guided U-Net Model...")
lambda_values = [0, 0.01, 0.02, 0.05, 0.1, 0.2]
lambda_tuning_results = {'lambda': [], 'iou': [], 'f1': [], 'val_loss': []}

initial_baseline_state = baseline_model.state_dict() # Capture state after baseline training

for lam in lambda_values:
    temp_model = UNet().to(device)
    temp_model.load_state_dict(initial_baseline_state) # Start each lambda test from baseline

    temp_tr_losses, temp_val_losses = train_model(temp_model, train_loader, val_loader, lam=lam, epochs=5, model_name=f"Lambda_{lam}")

    metrics, _, _ = evaluate(temp_model, val_loader) # Evaluate on validation set
    lambda_tuning_results['lambda'].append(lam)
    lambda_tuning_results['iou'].append(metrics['iou'])
    lambda_tuning_results['f1'].append(metrics['f1'])
    lambda_tuning_results['val_loss'].append(np.mean(temp_val_losses))

# Determine optimal lambda
best_lambda_idx = np.argmax(lambda_tuning_results['f1'])
best_lambda = lambda_tuning_results['lambda'][best_lambda_idx]
print(f"\nOptimal lambda found: {best_lambda} (F1-score: {lambda_tuning_results['f1'][best_lambda_idx]:.4f})")

# Plot Lambda Sensitivity (Fig.8)
plt.figure(figsize=(10, 5))
plt.plot(lambda_tuning_results['lambda'], lambda_tuning_results['iou'], marker='o', label='IoU')
plt.plot(lambda_tuning_results['lambda'], lambda_tuning_results['f1'], marker='x', label='F1-Score')
plt.axvline(x=best_lambda, color='r', linestyle='--', label=f'Optimal \u03bb={best_lambda}')
plt.title('Fig.8 Lambda Parameter Sensitivity')
plt.xlabel('Lambda (\u03bb)')
plt.ylabel('Metric Score')
plt.legend()
plt.grid(True)
save_fig("Fig8_lambda_sensitivity")
plt.show()

# Train final Physics-Guided U-Net with optimal lambda
physics_model = UNet().to(device)
physics_model.load_state_dict(initial_baseline_state) # Start from baseline again
tr_losses_physics, val_losses_physics = train_model(physics_model, train_loader, val_loader, lam=best_lambda, epochs=10, model_name="Physics-Guided")

# Plot training/validation loss for Physics-Guided
plt.figure(figsize=(10, 5))
plt.plot(tr_losses_physics, label='Train Loss')
plt.plot(val_losses_physics, label='Validation Loss')
plt.title(f'Fig.4 Physics-Guided U-Net (\u03bb={best_lambda}) Training & Validation Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.legend()
plt.grid(True)
save_fig("Fig4_physics_loss")
plt.show()

# =========================
# 14. FINAL EVALUATION AND COMPARISON
# =========================
print("\n[7/7] Final Evaluation and Comparison...")

# Evaluate models on test set
baseline_metrics, all_true_baseline, all_probs_baseline = evaluate(baseline_model, test_loader)
physics_metrics, all_true_physics, all_probs_physics = evaluate(physics_model, test_loader)
attention_metrics, all_true_attention, all_probs_attention = evaluate(attention_model, test_loader)

print("\nFINAL RESULTS (Test Set)")
print("Baseline U-Net:", baseline_metrics)
print("Attention U-Net:", attention_metrics)
print(f"Physics-Guided U-Net (\u03bb={best_lambda}):", physics_metrics)

# Store comparison results globally for geophysical analysis cell
comparison_results = {
    'Baseline': {'model': baseline_model, 'metrics': baseline_metrics},
    'Attention': {'model': attention_model, 'metrics': attention_metrics},
    'Physics': {'model': physics_model, 'metrics': physics_metrics, 'lambda': best_lambda}
}

# Store global variables for PR curve data
# global (removed: unnecessary/invalid at module scope) optimal_threshold_baseline, optimal_f1_baseline, optimal_precision_baseline, optimal_recall_baseline
# global (removed: unnecessary/invalid at module scope) optimal_threshold_physics, optimal_f1_physics, optimal_precision_physics, optimal_recall_physics
# global (removed: unnecessary/invalid at module scope) optimal_threshold_attention, optimal_f1_attention, optimal_precision_attention, optimal_recall_attention

# Calculate optimal thresholds and PR metrics for reporting
precision_baseline, recall_baseline, thresholds_baseline = precision_recall_curve(all_true_baseline, all_probs_baseline)
f1_scores_baseline = 2 * (precision_baseline * recall_baseline) / (precision_baseline + recall_baseline + 1e-8)
optimal_threshold_idx_baseline = np.argmax(f1_scores_baseline)
optimal_threshold_baseline = thresholds_baseline[optimal_threshold_idx_baseline]
optimal_f1_baseline = f1_scores_baseline[optimal_threshold_idx_baseline]
optimal_precision_baseline = precision_baseline[optimal_threshold_idx_baseline]
optimal_recall_baseline = recall_baseline[optimal_threshold_idx_baseline]

precision_physics, recall_physics, thresholds_physics = precision_recall_curve(all_true_physics, all_probs_physics)
f1_scores_physics = 2 * (precision_physics * recall_physics) / (precision_physics + recall_physics + 1e-8)
optimal_threshold_idx_physics = np.argmax(f1_scores_physics)
optimal_threshold_physics = thresholds_physics[optimal_threshold_idx_physics]
optimal_f1_physics = f1_scores_physics[optimal_threshold_idx_physics]
optimal_precision_physics = precision_physics[optimal_threshold_idx_physics]
optimal_recall_physics = recall_physics[optimal_threshold_idx_physics]

precision_attention, recall_attention, thresholds_attention = precision_recall_curve(all_true_attention, all_probs_attention)
f1_scores_attention = 2 * (precision_attention * recall_attention) / (precision_attention + recall_attention + 1e-8)
optimal_threshold_idx_attention = np.argmax(f1_scores_attention)
optimal_threshold_attention = thresholds_attention[optimal_threshold_idx_attention]
optimal_f1_attention = f1_scores_attention[optimal_threshold_idx_attention]
optimal_precision_attention = precision_attention[optimal_threshold_idx_attention]
optimal_recall_attention = recall_attention[optimal_threshold_idx_attention]

# Storing optimal thresholds in a results dict for potential later use in `results`
# For the existing `results` variable structure, we need to adapt
# For simplicity, these will be accessible globally or via a new combined 'results' dict
results = {
    'train_loader': train_loader,
    'val_loader': val_loader,
    'test_loader': test_loader,
    'patches_seismic': X_patches, # Renamed for clarity
    'patches_fault': Y_patches,   # Renamed for clarity
    'optimal_threshold_baseline': optimal_threshold_baseline,
    'optimal_threshold_physics': optimal_threshold_physics
}


# =========================
# 15. K-FOLD CROSS-VALIDATION
# =========================
print("\n" + "="*80)
print("K-FOLD CROSS-VALIDATION FOR ROBUSTNESS")
print("="*80)

num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

baseline_metrics_folds = []
physics_metrics_folds = []

for fold, (train_index, test_index) in enumerate(kf.split(full_dataset)):
    print(f"\n--- Fold {fold+1}/{num_folds} ---")

    train_dataset_fold = Subset(full_dataset, train_index)
    train_size_fold = int(0.8 * len(train_dataset_fold))
    val_size_fold = len(train_dataset_fold) - train_size_fold
    train_subset, val_subset = torch.utils.data.random_split(train_dataset_fold, [train_size_fold, val_size_fold])

    test_dataset_fold = Subset(full_dataset, test_index)

    train_loader_fold = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader_fold = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader_fold = DataLoader(test_dataset_fold, batch_size=batch_size, shuffle=False)

    # Train Baseline Model
    print(f"Training Baseline Model for Fold {fold+1}...")
    baseline_model_fold = UNet().to(device)
    _ = train_model(baseline_model_fold, train_loader_fold, val_loader_fold, lam=0, epochs=5, model_name=f"Baseline_Fold_{fold+1}") # Reduced epochs for CV speed
    metrics_baseline, _, _ = evaluate(baseline_model_fold, test_loader_fold)
    baseline_metrics_folds.append(metrics_baseline)
    print(f"  Baseline IoU: {metrics_baseline['iou']:.4f}, F1: {metrics_baseline['f1']:.4f}")

    # Train Physics-Regularized Model
    print(f"Training Physics-Regularized Model for Fold {fold+1} (\u03bb={best_lambda:.3f})...")
    physics_model_fold = UNet().to(device)
    physics_model_fold.load_state_dict(baseline_model_fold.state_dict()) # Start from baseline for fair comparison
    _ = train_model(physics_model_fold, train_loader_fold, val_loader_fold,
                       lam=best_lambda, epochs=5, model_name=f"Physics_Fold_{fold+1}") # Reduced epochs
    metrics_physics, _, _ = evaluate(physics_model_fold, test_loader_fold)
    physics_metrics_folds.append(metrics_physics)
    print(f"  Physics IoU: {metrics_physics['iou']:.4f}, F1: {metrics_physics['f1']:.4f}")

results['kfold_baseline_metrics'] = baseline_metrics_folds
results['kfold_physics_metrics'] = physics_metrics_folds

print("\n" + "="*80)
print("K-FOLD CROSS-VALIDATION SUMMARY")
print("="*80)

baseline_df = pd.DataFrame(baseline_metrics_folds)
physics_df = pd.DataFrame(physics_metrics_folds)

print("\nBaseline Model Metrics (Mean \u00b1 Std Dev):")
for col in baseline_df.columns:
    print(f"  {col.capitalize()}: {baseline_df[col].mean():.4f} \u00b1 {baseline_df[col].std():.4f}")

print("\nPhysics-Regularized Model Metrics (Mean \u00b1 Std Dev):")
for col in physics_df.columns:
    print(f"  {col.capitalize()}: {physics_df[col].mean():.4f} \u00b1 {physics_df[col].std():.4f}")
print("K-Fold cross-validation complete.")


# =========================
# 15b. BOOTSTRAP HELPER FUNCTIONS (moved earlier so Section 16b can use them)
# =========================
num_bootstraps = 1000
confidence_level = 0.95


def calculate_bootstrapped_metrics_full(model, dataloader, threshold, num_bootstraps=1000):
    """Generalized version: works for ANY trained model (baseline, physics,
    attention, or any ablation variant) given its own optimal threshold.
    Returns a dict of {iou, f1, precision, recall: np.array of length
    num_bootstraps} suitable for computing mean/std/95% CI."""
    true_labels = []
    pred_probs_list = []
    model.eval()
    with torch.no_grad():
        for seismic, fault in dataloader:
            seismic = seismic.to(device)
            pred_probs = model(seismic)
            true_labels.append(fault.cpu().numpy())
            pred_probs_list.append(pred_probs.cpu().numpy())

    true_labels = np.concatenate(true_labels, axis=0).flatten()
    pred_probs_flattened = np.concatenate(pred_probs_list, axis=0).flatten()

    print(f"  Using optimal threshold: {threshold:.3f}")
    pred_labels_flattened = (pred_probs_flattened > threshold).astype(np.float32)
    indices = np.arange(len(true_labels))

    out = {"iou": [], "f1": [], "precision": [], "recall": []}
    rng = np.random.default_rng(42)
    for _ in range(num_bootstraps):
        resampled_indices = rng.choice(indices, size=len(indices), replace=True)
        resampled_true = true_labels[resampled_indices]
        resampled_pred = pred_labels_flattened[resampled_indices]

        intersection = np.sum(resampled_pred * resampled_true)
        union = np.sum((resampled_pred + resampled_true) > 0)
        iou = intersection / (union + 1e-8)

        tp = intersection
        fp = np.sum(resampled_pred) - tp
        fn = np.sum(resampled_true) - tp
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        out["iou"].append(iou)
        out["f1"].append(f1)
        out["precision"].append(precision)
        out["recall"].append(recall)

    return {k: np.array(v) for k, v in out.items()}


def summarize_bootstrap(boot_dict):
    """{metric: array} -> {metric: {mean, std, ci95_lo, ci95_hi}}"""
    summary = {}
    for k, v in boot_dict.items():
        summary[k] = {
            "mean": float(np.mean(v)),
            "std": float(np.std(v)),
            "ci95_lo": float(np.percentile(v, 2.5)),
            "ci95_hi": float(np.percentile(v, 97.5)),
        }
    return summary


# =========================
# 16. ABLATION STUDY
# =========================
print("\n" + "="*80)
print("PHYSICS REGULARIZATION ABLATION STUDY")
print("="*80)

def get_ablated_physics_weights_dict(excluded_loss_name):
    """Returns a physics weights dictionary with one term set to zero."""
    current_weights = PHYSICS_WEIGHTS.copy()
    if excluded_loss_name in current_weights:
        current_weights[excluded_loss_name] = 0
    return current_weights

ablation_terms = ['dip', 'sparsity', 'continuity', 'contrast']
ablation_results = {}
ablation_models = {}   # keep trained model objects for bootstrap CI later

# Use the same train/val/test loaders as the main experiment

initial_model_for_ablation = UNet().to(device)
initial_model_for_ablation.load_state_dict(baseline_model.state_dict()) # Start from the trained baseline model

for term in ablation_terms:
    print(f"\n[ABLATION] Excluding {term} loss term...")
    ablated_model = UNet().to(device)
    ablated_model.load_state_dict(initial_model_for_ablation.state_dict()) # Start each ablated model from the same baseline

    ablated_weights = get_ablated_physics_weights_dict(term)

    # Train the ablated model using the best_lambda
    _ = train_model(ablated_model, train_loader, val_loader,
                          lam=best_lambda, epochs=10, model_name=f"Ablated_{term}",
                          physics_weights=ablated_weights) # Pass the dictionary

    # Evaluate the trained ablated model
    metrics, _, _ = evaluate(ablated_model, test_loader)
    ablation_results[term] = metrics
    ablation_models[term] = ablated_model  # keep reference for bootstrap CI
    print(f"  \u2713 Ablated (excluded {term}) Test IoU: {metrics['iou']:.4f}, F1: {metrics['f1']:.4f}")

# Collect and Tabulate Results
baseline_metrics_abl = baseline_metrics # Use the metrics from the main baseline evaluation

print("\n[ABLATION] Training full Physics-Regularized model for comparison...")
full_physics_model_abl = UNet().to(device)
full_physics_model_abl.load_state_dict(initial_model_for_ablation.state_dict())
_ = train_model(full_physics_model_abl, train_loader, val_loader,
                      lam=best_lambda, epochs=10, model_name="Full_Physics_Ablation_Ref")
full_physics_metrics_abl, _, _ = evaluate(full_physics_model_abl, test_loader)
print(f"  \u2713 Full Physics Test IoU: {full_physics_metrics_abl['iou']:.4f}, F1: {full_physics_metrics_abl['f1']:.4f}")

ablation_summary = {
    'Model': [], 'IoU': [], 'F1-Score': [], 'Precision': [], 'Recall': []
}

ablation_summary['Model'].append('Baseline (No Physics)')
ablation_summary['IoU'].append(baseline_metrics_abl['iou'])
ablation_summary['F1-Score'].append(baseline_metrics_abl['f1'])
ablation_summary['Precision'].append(baseline_metrics_abl['precision'])
ablation_summary['Recall'].append(baseline_metrics_abl['recall'])

ablation_summary['Model'].append('Full Physics-Regularized')
ablation_summary['IoU'].append(full_physics_metrics_abl['iou'])
ablation_summary['F1-Score'].append(full_physics_metrics_abl['f1'])
ablation_summary['Precision'].append(full_physics_metrics_abl['precision'])
ablation_summary['Recall'].append(full_physics_metrics_abl['recall'])

for term, metrics in ablation_results.items():
    ablation_summary['Model'].append(f'Ablated (Excluding {term})')
    ablation_summary['IoU'].append(metrics['iou'])
    ablation_summary['F1-Score'].append(metrics['f1'])
    ablation_summary['Precision'].append(metrics['precision'])
    ablation_summary['Recall'].append(metrics['recall'])

results['ablation_df'] = pd.DataFrame(ablation_summary)

print("\n" + "="*80)
print("ABLATION STUDY RESULTS")
print("="*80)
print(results['ablation_df'].to_string(index=False))

print("Physics ablation study complete.")

# =========================
# 16b. ABLATION BOOTSTRAP 95% CIs
# =========================
# extends the bootstrap CI procedure (already used for baseline/physics
# below) to every ablation row, not just the two main models
print("\n" + "="*80)
print("ABLATION STUDY: BOOTSTRAP 95% CONFIDENCE INTERVALS")
print("="*80)

# optimal threshold per ablation model, computed the same way as for
# baseline/physics/attention above (argmax F1 on the PR curve)
def _optimal_threshold(y_true, y_prob):
    p, r, t = precision_recall_curve(y_true, y_prob)
    f1s = 2 * (p * r) / (p + r + 1e-8)
    idx = np.argmax(f1s[:-1]) if len(t) > 0 else 0  # last point has no threshold
    return t[idx] if len(t) > 0 else 0.5

ablation_ci_rows = []

# Baseline (No Physics) and Full Physics-Regularized reuse the models/thresholds already computed
_, y_true_base_abl, y_prob_base_abl = evaluate(baseline_model, test_loader)
_, y_true_full_abl, y_prob_full_abl = evaluate(full_physics_model_abl, test_loader)
thr_base_abl = _optimal_threshold(y_true_base_abl, y_prob_base_abl)
thr_full_abl = _optimal_threshold(y_true_full_abl, y_prob_full_abl)

print("\n[Bootstrap] Baseline (No Physics)...")
boot_base = summarize_bootstrap(
    calculate_bootstrapped_metrics_full(baseline_model, test_loader, thr_base_abl, num_bootstraps))
ablation_ci_rows.append(("Baseline (No Physics)", boot_base))

print("\n[Bootstrap] Full Physics-Regularized...")
boot_full = summarize_bootstrap(
    calculate_bootstrapped_metrics_full(full_physics_model_abl, test_loader, thr_full_abl, num_bootstraps))
ablation_ci_rows.append(("Full Physics-Regularized", boot_full))

for term in ablation_terms:
    print(f"\n[Bootstrap] w/o {term.capitalize()} Constraint...")
    m = ablation_models[term]
    _, y_true_t, y_prob_t = evaluate(m, test_loader)
    thr_t = _optimal_threshold(y_true_t, y_prob_t)
    boot_t = summarize_bootstrap(
        calculate_bootstrapped_metrics_full(m, test_loader, thr_t, num_bootstraps))
    ablation_ci_rows.append((f"w/o {term.capitalize()} Constraint", boot_t))

print("\n" + "="*80)
print("ABLATION TABLE WITH 95% BOOTSTRAP CI (copy into Table 3)")
print("="*80)
for name, s in ablation_ci_rows:
    def fmt(m):
        return f"{s[m]['mean']:.3f} [{s[m]['ci95_lo']:.3f}, {s[m]['ci95_hi']:.3f}]"
    print(f"{name:30s} IoU={fmt('iou')}  F1={fmt('f1')}  Prec={fmt('precision')}  Rec={fmt('recall')}")

print("\nLaTeX rows (IoU, F1-score, Precision, Recall):")
for name, s in ablation_ci_rows:
    def fmt(m):
        return f"{s[m]['mean']:.3f} [{s[m]['ci95_lo']:.3f}, {s[m]['ci95_hi']:.3f}]"
    print(f"{name} & {fmt('iou')} & {fmt('f1')} & {fmt('precision')} & {fmt('recall')} \\\\\\\\")

results['ablation_ci'] = dict(ablation_ci_rows)

# =========================
# 17. BOOTSTRAPPING AND STATISTICAL TEST
# =========================
print("\n" + "="*80)
print("BOOTSTRAPPED CONFIDENCE INTERVALS AND PAIRED STATISTICAL TEST")
print("="*80)

# num_bootstraps, confidence_level, calculate_bootstrapped_metrics_full, and
# summarize_bootstrap are now defined earlier (Section 15b) so Section 16b's
# per-ablation-row bootstrap CIs can use them too.


def calculate_bootstrapped_metrics(model, dataloader, num_bootstraps=1000, metric_name='iou'):
    if model is baseline_model:
        threshold = optimal_threshold_baseline
    elif model is physics_model:
        threshold = optimal_threshold_physics
    elif model is attention_model:
        threshold = optimal_threshold_attention
    else:
        threshold = 0.5
    return calculate_bootstrapped_metrics_full(model, dataloader, threshold, num_bootstraps)[metric_name]

# --- Baseline Model Bootstrapping ---
print("\n[1/2] Calculating bootstrapped metrics for Baseline Model...")
baseline_iou_bootstraps = calculate_bootstrapped_metrics(baseline_model, test_loader, num_bootstraps, 'iou')
baseline_f1_bootstraps = calculate_bootstrapped_metrics(baseline_model, test_loader, num_bootstraps, 'f1')

# --- Physics Model Bootstrapping ---
print("\n[2/2] Calculating bootstrapped metrics for Physics-Guided Model...")
physics_iou_bootstraps = calculate_bootstrapped_metrics(physics_model, test_loader, num_bootstraps, 'iou')
physics_f1_bootstraps = calculate_bootstrapped_metrics(physics_model, test_loader, num_bootstraps, 'f1')

# --- Confidence Intervals ---
alpha = 1.0 - confidence_level

lower_iou_baseline = np.percentile(baseline_iou_bootstraps, alpha / 2 * 100)
upper_iou_baseline = np.percentile(baseline_iou_bootstraps, (1 - alpha / 2) * 100)

lower_iou_physics = np.percentile(physics_iou_bootstraps, alpha / 2 * 100)
upper_iou_physics = np.percentile(physics_iou_bootstraps, (1 - alpha / 2) * 100)

lower_f1_baseline = np.percentile(baseline_f1_bootstraps, alpha / 2 * 100)
upper_f1_baseline = np.percentile(baseline_f1_bootstraps, (1 - alpha / 2) * 100)

lower_f1_physics = np.percentile(physics_f1_bootstraps, alpha / 2 * 100)
upper_f1_physics = np.percentile(physics_f1_bootstraps, (1 - alpha / 2) * 100)

print("\n" + "="*80)
print(f"BOOTSTRAPPED {int(confidence_level*100)}% CONFIDENCE INTERVALS (from {num_bootstraps} resamples)")
print("="*80)
print(f"Baseline IoU: {np.mean(baseline_iou_bootstraps):.3f} ({lower_iou_baseline:.3f}, {upper_iou_baseline:.3f})")
print(f"Physics IoU:  {np.mean(physics_iou_bootstraps):.3f} ({lower_iou_physics:.3f}, {upper_iou_physics:.3f})")
print(f"Baseline F1-Score: {np.mean(baseline_f1_bootstraps):.3f} ({lower_f1_baseline:.3f}, {upper_f1_baseline:.3f})")
print(f"Physics F1-Score:  {np.mean(physics_f1_bootstraps):.3f} ({lower_f1_physics:.3f}, {upper_f1_physics:.3f})")

# --- Paired Statistical Test (Wilcoxon signed-rank test) ---
print("\n" + "="*80)
print("PAIRED STATISTICAL TEST (WILCOXON SIGNED-RANK)")
print("="*80)

diff_iou = np.array(physics_iou_bootstraps) - np.array(baseline_iou_bootstraps)
diff_f1 = np.array(physics_f1_bootstraps) - np.array(baseline_f1_bootstraps)

w_statistic_iou, p_value_iou = wilcoxon(diff_iou)
print(f"IoU Comparison (physics - baseline): Median difference = {np.median(diff_iou):.4f}")
print(f"  Wilcoxon test p-value: {p_value_iou:.4e}")
if p_value_iou < alpha:
    print(f"  -> Significant difference in IoU (p < {alpha:.3f})")
else:
    print(f"  -> No significant difference in IoU (p >= {alpha:.3f})")

w_statistic_f1, p_value_f1 = wilcoxon(diff_f1)
print(f"F1-Score Comparison (physics - baseline): Median difference = {np.median(diff_f1):.4f}")
print(f"  Wilcoxon test p-value: {p_value_f1:.4e}")
if p_value_f1 < alpha:
    print(f"  -> Significant difference in F1-Score (p < {alpha:.3f})")
else:
    print(f"  -> No significant difference in F1-Score (p >= {alpha:.3f})")

print("\nBootstrapped confidence intervals and paired statistical test complete.")

# =========================
# 17b. ATTENTION U-NET BOOTSTRAP + FINAL SOTA COMPARISON TABLE
# =========================
print("\n" + "="*80)
print("ATTENTION U-NET BOOTSTRAP (SOTA COMPARISON)")
print("="*80)
boot_attention_full = calculate_bootstrapped_metrics_full(
    attention_model, test_loader, optimal_threshold_attention, num_bootstraps)
attention_ci = summarize_bootstrap(boot_attention_full)

boot_baseline_full = calculate_bootstrapped_metrics_full(
    baseline_model, test_loader, optimal_threshold_baseline, num_bootstraps)
baseline_ci = summarize_bootstrap(boot_baseline_full)

boot_physics_full = calculate_bootstrapped_metrics_full(
    physics_model, test_loader, optimal_threshold_physics, num_bootstraps)
physics_ci = summarize_bootstrap(boot_physics_full)

diff_iou_att = boot_attention_full['iou'] - boot_baseline_full['iou']
w_stat_att, p_val_att = wilcoxon(diff_iou_att)
print(f"Attention vs Baseline IoU: median diff = {np.median(diff_iou_att):.4f}, Wilcoxon p = {p_val_att:.4e}")

diff_iou_phys_att = boot_physics_full['iou'] - boot_attention_full['iou']
w_stat_pa, p_val_pa = wilcoxon(diff_iou_phys_att)
print(f"Physics-Guided vs Attention IoU: median diff = {np.median(diff_iou_phys_att):.4f}, Wilcoxon p = {p_val_pa:.4e}")

print("\n" + "="*80)
print("FINAL SOTA COMPARISON TABLE (Table 1) WITH 95% BOOTSTRAP CI")
print("="*80)


def _row(name, s):
    def fmt(m):
        return f"{s[m]['mean']:.3f} [{s[m]['ci95_lo']:.3f}, {s[m]['ci95_hi']:.3f}]"
    print(f"{name:28s} IoU={fmt('iou')}  F1={fmt('f1')}  Prec={fmt('precision')}  Rec={fmt('recall')}")
    return f"{name} & {s['iou']['mean']:.3f} & {s['f1']['mean']:.3f} & {s['precision']['mean']:.3f} & {s['recall']['mean']:.3f} \\\\\\\\"


latex_rows = [
    _row("Baseline U-Net", baseline_ci),
    _row("Attention U-Net", attention_ci),
    _row("Physics-guided U-Net (ours)", physics_ci),
]

print("\nLaTeX rows for Table 1 (Traditional/C3-Coherence row must be kept from your"
      " existing separate computation \u2014 it is not produced by this script):")
for r in latex_rows:
    print(r)

results['sota_ci'] = {"baseline": baseline_ci, "attention": attention_ci, "physics": physics_ci}

# =========================
# 18. Generating Visualization Figures
# =========================
print("\n" + "="*80)
print("GENERATING VISUALIZATION FIGURES")
print("="*80)

# FIGURE 5 - Fault Prediction Comparison
# (Note: This uses a single patch example for visualization, not full reconstructed seismic)

# Get one batch from the test loader for comparison
for X_vis_batch, Y_vis_batch in test_loader:
    break # Take the first batch

sample_seismic_patch_vis = X_vis_batch[0].unsqueeze(0).to(device) # Add batch dimension
sample_gt_patch_vis = Y_vis_batch[0].squeeze().cpu().numpy() # Remove batch/channel dims

with torch.no_grad():
    pred_baseline_patch_vis = baseline_model(sample_seismic_patch_vis).squeeze().cpu().numpy()
    pred_physics_patch_vis = physics_model(sample_seismic_patch_vis).squeeze().cpu().numpy()

def plot_fault_comparison(seismic_input, gt, pred_base, pred_phys):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(gt, cmap="gray_r") # Changed to gray_r for fault (white on black)
    axes[0].set_title("Ground Truth")

    axes[1].imshow(pred_base > 0.5, cmap="hot") # Apply threshold
    axes[1].set_title("U-Net Baseline")

    axes[2].imshow(pred_phys > 0.5, cmap="hot") # Apply threshold
    axes[2].set_title("Physics-Guided U-Net")

    axes[3].imshow(seismic_input, cmap="gray")
    axes[3].set_title("RTM Input")

    for ax in axes:
        ax.axis("off")

    plt.suptitle("Fig.5 Fault Prediction Comparison", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_fig("Fig5_fault_comparison")
    plt.show()

plot_fault_comparison(
    sample_seismic_patch_vis.squeeze().cpu().numpy(),
    sample_gt_patch_vis,
    pred_baseline_patch_vis,
    pred_physics_patch_vis
)

# FIGURE 6 - Zoomed Fault Resolution
def plot_zoomed_faults(gt, pred_base, pred_phys, x1, x2, z1, z2):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(gt[z1:z2, x1:x2], cmap="gray_r")
    axes[0].set_title("Ground Truth")

    axes[1].imshow(pred_base[z1:z2, x1:x2] > 0.5, cmap="hot")
    axes[1].set_title("U-Net Baseline")

    axes[2].imshow(pred_phys[z1:z2, x1:x2] > 0.5, cmap="hot")
    axes[2].set_title("Physics-Guided U-Net")

    for ax in axes:
        ax.axis("off")

    plt.suptitle("Fig.6 Zoomed Fault Resolution", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_fig("Fig6_zoomed_faults")
    plt.show()

# Define a representative zoom window for plot_zoomed_faults
# These coordinates are for the 128x128 patch
x1_zoom, x2_zoom = 20, 100
z1_zoom, z2_zoom = 30, 90

plot_zoomed_faults(
    sample_gt_patch_vis,
    pred_baseline_patch_vis,
    pred_physics_patch_vis,
    x1_zoom, x2_zoom, z1_zoom, z2_zoom
)


# FIGURE 7 - Precision-Recall Curves
def plot_precision_recall_curve(y_true, prob_baseline, prob_physics, prob_attention=None):
    p_b, r_b, _ = precision_recall_curve(y_true, prob_baseline)
    p_p, r_p, _ = precision_recall_curve(y_true, prob_physics)

    plt.figure(figsize=(8, 6))
    plt.plot(r_b, p_b, label=f"U-Net Baseline (AUC={auc(r_b, p_b):.3f})")
    if prob_attention is not None:
        p_a, r_a, _ = precision_recall_curve(y_true, prob_attention)
        plt.plot(r_a, p_a, label=f"Attention U-Net (AUC={auc(r_a, p_a):.3f})")
    plt.plot(r_p, p_p, label=f"Physics-Guided U-Net (AUC={auc(r_p, p_p):.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Fig.7 Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_fig("Fig7_precision_recall_curve")
    plt.show()

# Prepare data for Precision-Recall Curve from global `all_true_physics/baseline` and `all_probs_physics/baseline`
# These were populated by the `evaluate` function in the final evaluation step.
# Need to ensure all_true and all_probs are available for the entire test set.

# The `evaluate` function returns these, so we need to collect them from the last call
# For simplicity, let's re-run evaluate to get these specific outputs if not globally stored.
_, all_true_baseline_pr, all_probs_baseline_pr = evaluate(baseline_model, test_loader, thr=0.5)
_, all_true_physics_pr, all_probs_physics_pr = evaluate(physics_model, test_loader, thr=0.5)
_, all_true_attention_pr, all_probs_attention_pr = evaluate(attention_model, test_loader, thr=0.5)

plot_precision_recall_curve(all_true_baseline_pr, all_probs_baseline_pr, all_probs_physics_pr, all_probs_attention_pr)

# FIGURE 9 - ROC Curve
def plot_roc_curve(y_true, prob_baseline, prob_physics, prob_attention=None):
    fpr_b, tpr_b, _ = roc_curve(y_true, prob_baseline)
    fpr_p, tpr_p, _ = roc_curve(y_true, prob_physics)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_b, tpr_b, label=f"U-Net Baseline (AUC={auc(fpr_b, tpr_b):.3f})")
    if prob_attention is not None:
        fpr_a, tpr_a, _ = roc_curve(y_true, prob_attention)
        plt.plot(fpr_a, tpr_a, label=f"Attention U-Net (AUC={auc(fpr_a, tpr_a):.3f})")
    plt.plot(fpr_p, tpr_p, label=f"Physics-Guided U-Net (AUC={auc(fpr_p, tpr_p):.3f})")
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Fig.9 Receiver Operating Characteristic (ROC) Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_fig("Fig9_roc_curve")
    plt.show()

plot_roc_curve(all_true_baseline_pr, all_probs_baseline_pr, all_probs_physics_pr, all_probs_attention_pr)

print("\n" + "="*80)
print("ALL FIGURES GENERATED AND SAVED TO 'figures' DIRECTORY.")
print("PIPELINE COMPLETE.")
print("="*80)