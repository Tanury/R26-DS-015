"""
ppmi_validate.py
─────────────────
External validation of the DaTscan PD encoder on new PPMI subjects
not seen during training. Same preprocessing pipeline as training:
DICOM → NIfTI → peak transaxial slice → SBR normalisation → inference.
"""

import subprocess
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import (accuracy_score, f1_score,
                             roc_auc_score, confusion_matrix)

# ── Paths ──────────────────────────────────────────────────────────────────────
PD_DIR    = Path('/Users/tanuridissanayaka/Test Data/PD PD')
HC_DIR    = Path('/Users/tanuridissanayaka/Test Data/PD HC')
CKPT_PATH = Path('/Users/tanuridissanayaka/research/research_parent/final_neuro'
                 '/mri/parkinsons/models/checkpoints/dat_best_model.pth')

# ── Model (same as training) ───────────────────────────────────────────────────
class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.block(x)

class ResBlock2D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvBnRelu(ch, ch),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch)
        )
    def forward(self, x): return F.relu(x + self.block(x), inplace=True)

class DaTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem   = nn.Sequential(
            ConvBnRelu(1, 32, kernel=7, stride=2, padding=3),
            ConvBnRelu(32, 64, stride=2),
        )
        self.stage1 = nn.Sequential(ResBlock2D(64),  ConvBnRelu(64,  128, stride=2))
        self.stage2 = nn.Sequential(ResBlock2D(128), ConvBnRelu(128, 256, stride=2))
        self.stage3 = nn.Sequential(ResBlock2D(256), ConvBnRelu(256, 256, stride=2))
        self.pool   = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x):
        x = self.stem(x); x = self.stage1(x)
        x = self.stage2(x); x = self.stage3(x)
        return self.pool(x).flatten(1)

class DaTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone   = DaTCNN()
        self.projector  = nn.Sequential(
            nn.Linear(256, 256), nn.BatchNorm1d(256),
            nn.ReLU(inplace=True), nn.Linear(256, 256)
        )
        self.classifier = nn.Linear(256, 2)
    def forward(self, x):
        feat  = self.backbone(x)
        embed = self.projector(feat)
        z     = F.normalize(embed, p=2, dim=1)
        return self.classifier(z), z


# ── Preprocessing (identical to training pipeline) ────────────────────────────
def find_dcm(subject_dir: Path) -> Path | None:
    dcms = list(subject_dir.rglob('*.dcm'))
    return dcms[0] if dcms else None


def preprocess_dcm(dcm_path: Path) -> torch.Tensor | None:
    """DICOM → NIfTI → peak slice → SBR normalise → (1,1,224,224) tensor."""
    import nibabel as nib
    import cv2

    with tempfile.TemporaryDirectory() as tmp:
        # Step 1 — DICOM → NIfTI
        result = subprocess.run(
            ['dcm2niix', '-z', 'y', '-f', 'dat', '-o', tmp,
             str(dcm_path.parent)],
            capture_output=True
        )
        niis = list(Path(tmp).glob('*.nii.gz'))
        if not niis:
            return None

        # Step 2 — Load volume
        img  = nib.load(str(niis[0]))
        data = img.get_fdata(dtype=np.float32)
        if data.ndim == 4:
            data = data[..., 0]

        # Clamp to first 120 slices (canonical PPMI depth)
        if data.shape[1] > 120:
            data = data[:, :120, :]

        # Step 3 — Find peak transaxial slice (axis 1)
        lo = int(data.shape[1] * 0.30)
        hi = int(data.shape[1] * 0.70)
        slice_means = data[:, lo:hi, :].mean(axis=(0, 2))
        best_local  = int(np.argmax(slice_means))
        best_ax1    = lo + best_local
        w_lo = max(0, best_ax1 - 4)
        w_hi = min(data.shape[1], best_ax1 + 5)
        slice_2d = data[:, w_lo:w_hi, :].mean(axis=1)

        # Step 4 — Resize to 224×224
        resized = cv2.resize(slice_2d, (224, 224),
                             interpolation=cv2.INTER_LINEAR).astype(np.float32)

        # Step 5 — SBR normalisation
        h = resized.shape[0]
        ref_region = resized[int(h * 0.75):, :]
        ref_vals   = ref_region[ref_region > 0]
        if len(ref_vals) > 10:
            resized = resized / (ref_vals.mean() + 1e-8)
        else:
            nz = resized[resized > 0]
            if len(nz) < 100:
                return None
            p_low, p_high = np.percentile(nz, 1), np.percentile(nz, 99)
            resized = np.clip(resized, p_low, p_high)
            if p_high > p_low:
                resized = (resized - p_low) / (p_high - p_low)

        tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0)
        return tensor


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load model
    model = DaTEncoder().to(device)
    ckpt  = torch.load(str(CKPT_PATH), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'Checkpoint: epoch {ckpt["epoch"]}, PPMI val_acc={ckpt["val_acc"]:.4f}')

    # Collect subjects
    records = []
    for label_int, folder in [(1, PD_DIR), (0, HC_DIR)]:
        label_str = 'PD' if label_int == 1 else 'HC'
        subjects  = sorted([s for s in folder.iterdir() if s.is_dir()])
        for subj in subjects:
            dcm = find_dcm(subj)
            if dcm:
                records.append({
                    'subject_id': subj.name,
                    'dcm_path':   dcm,
                    'label':      label_int,
                    'label_str':  label_str,
                })
        print(f'{label_str}: {len(subjects)} subjects found')

    print(f'\nProcessing {len(records)} subjects...\n')

    all_preds  = []
    all_labels = []
    all_probs  = []
    failed     = []

    with torch.no_grad():
        for i, rec in enumerate(records):
            tensor = preprocess_dcm(rec['dcm_path'])
            if tensor is None:
                print(f'  ⚠ {rec["subject_id"]} — preprocessing failed, skipping')
                failed.append(rec['subject_id'])
                continue

            tensor = tensor.to(device)
            logits, _ = model(tensor)
            probs     = torch.softmax(logits, dim=1)
            pred      = logits.argmax(1).item()
            prob_pd   = probs[0, 1].item()

            all_preds.append(pred)
            all_labels.append(rec['label'])
            all_probs.append(prob_pd)

            pred_str = 'PD' if pred == 1 else 'HC'
            ok = '✓' if pred == rec['label'] else '✗'
            print(f'  [{i+1:03d}] {rec["subject_id"]:<10} '
                  f'true={rec["label_str"]} pred={pred_str} '
                  f'prob={prob_pd:.3f} {ok}')

    # Metrics
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    accuracy    = accuracy_score(all_labels, all_preds)
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    f1          = f1_score(all_labels, all_preds)
    auc         = roc_auc_score(all_labels, all_probs)

    print('\n' + '-'*55)
    print('  Validation of DaTscan PD Encoder')
    print('='*55)
    print(f'  Subjects processed : {len(all_labels)} '
          f'(PD={int(all_labels.sum())} HC={int((all_labels==0).sum())})')
    if failed:
        print(f'  Failed/skipped     : {len(failed)}')
    print(f'  Accuracy           : {accuracy:.4f}')
    print(f'  Sensitivity        : {sensitivity:.4f}  (PD recall)')
    print(f'  Specificity        : {specificity:.4f}  (HC recall)')
    print(f'  F1 Score           : {f1:.4f}')
    print(f'  AUC-ROC            : {auc:.4f}')
    print(f'\n  Confusion Matrix:')
    print(f'              Pred PD   Pred HC')
    print(f'  True PD  :  {tp:6d}   {fn:6d}')
    print(f'  True HC  :  {fp:6d}   {tn:6d}')
    print('-'*55)


if __name__ == '__main__':
    main()