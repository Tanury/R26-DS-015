"""
ntua_validate.py
─────────────────
External validation of the DaTscan PD encoder on the NTUA Parkinson Dataset.
Subjects: 55 PD + 23 Non-PD
Input : PNG slices from 0.DAT/s1/ (241×267 RGB)
Process: grayscale → resize 224×224 → log-normalise → model inference
Output : accuracy, sensitivity, specificity, F1, AUC-ROC
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix, classification_report)

# ── Paths ──────────────────────────────────────────────────────────────────────
NTUA_DIR  = Path('/Users/tanuridissanayaka/Test Data/ntua-parkinson-dataset-master')
CKPT_PATH = Path('/Users/tanuridissanayaka/research/research_parent/final_neuro/mri/parkinsons/models/checkpoints/dat_best_model.pth')

#sys.path.insert(0, str(Path('/Users/tanuridissanayaka/research/R26-DS-015')))
sys.path.insert(0, str(Path('/Users/tanuridissanayaka/research/research_parent/final_neuro')))

# ── Model definition (must match training) ────────────────────────────────────
class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.block(x)

class ResBlock2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBnRelu(channels, channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
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
        feat   = self.backbone(x)
        embed  = self.projector(feat)
        z_img  = F.normalize(embed, p=2, dim=1)
        logits = self.classifier(z_img)
        return logits, z_img


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_png(png_path: Path) -> torch.Tensor:
    """
    Load NTUA PNG → match our training preprocessing:
    grayscale → log-normalise → resize 224×224 → tensor
    """
    img  = Image.open(png_path).convert('L')           # RGB → grayscale
    arr  = np.array(img, dtype=np.float32) / 255.0     # [0, 1]

    # Log compression (same as our extract_bscans normalisation)
    SENTINEL = 3.0e38
    arr[arr >= SENTINEL] = 0.0
    log_img = np.log1p(arr * 1000)

    # Percentile clip
    p_low, p_high = np.percentile(log_img, 1), np.percentile(log_img, 99)
    clipped = np.clip(log_img, p_low, p_high)
    if p_high > p_low:
        norm = (clipped - p_low) / (p_high - p_low)
    else:
        norm = np.zeros_like(clipped)

    # Resize to 224×224
    img_norm = Image.fromarray((norm * 255).astype(np.uint8))
    img_resized = img_norm.resize((224, 224), Image.BILINEAR)
    tensor = torch.from_numpy(
        np.array(img_resized, dtype=np.float32) / 255.0
    ).unsqueeze(0)  # (1, 224, 224)

    return tensor


# ── Load data ─────────────────────────────────────────────────────────────────
def load_subjects(label_name: str, label_int: int):
    """Load all subjects from a label folder, pick middle slice from s1."""
    folder   = NTUA_DIR / label_name
    subjects = sorted([s for s in folder.iterdir() if s.is_dir()])
    records  = []

    for subj in subjects:
        dat_s1 = subj / '0.DAT' / 's1'
        if not dat_s1.exists():
            continue
        pngs = sorted(dat_s1.glob('*.png'))
        if not pngs:
            continue
        # Use middle slice (most informative striatal level)
        mid_png = pngs[len(pngs) // 2]
        records.append({
            'subject_id': subj.name,
            'png_path':   mid_png,
            'label':      label_int,
        })

    return records


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load model
    model = DaTEncoder().to(device)
    ckpt  = torch.load(str(CKPT_PATH), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f'Checkpoint loaded: epoch {ckpt["epoch"]}, '
          f'PPMI val_acc={ckpt["val_acc"]:.4f}')

    # Load subjects
    pd_records  = load_subjects('PD Patients',     label_int=1)
    hc_records  = load_subjects('Non PD Patients', label_int=0)
    all_records = pd_records + hc_records
    print(f'\nNTUA subjects — PD: {len(pd_records)}  HC: {len(hc_records)}')

    # Run inference
    all_preds  = []
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for rec in all_records:
            tensor = preprocess_png(rec['png_path']).unsqueeze(0).to(device)
            logits, z_img = model(tensor)
            probs = torch.softmax(logits, dim=1)
            pred  = logits.argmax(1).item()
            prob_pd = probs[0, 1].item()

            all_preds.append(pred)
            all_labels.append(rec['label'])
            all_probs.append(prob_pd)

    # Metrics
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    cm          = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    accuracy    = accuracy_score(all_labels, all_preds)
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    f1          = f1_score(all_labels, all_preds)
    auc         = roc_auc_score(all_labels, all_probs)

    print('\n' + '='*55)
    print('  NTUA EXTERNAL VALIDATION — DaTscan PD Encoder')
    print('='*55)
    print(f'  Subjects     : {len(all_records)} (PD={len(pd_records)} HC={len(hc_records)})')
    print(f'  Accuracy     : {accuracy:.4f}')
    print(f'  Sensitivity  : {sensitivity:.4f}  (PD recall)')
    print(f'  Specificity  : {specificity:.4f}  (HC recall)')
    print(f'  F1 Score     : {f1:.4f}')
    print(f'  AUC-ROC      : {auc:.4f}')
    print(f'\n  Confusion Matrix:')
    print(f'              Pred PD   Pred HC')
    print(f'  True PD  :  {tp:6d}   {fn:6d}')
    print(f'  True HC  :  {fp:6d}   {tn:6d}')
    print('='*55)

    # Per-subject breakdown
    print('\nPer-subject results:')
    print(f'{"Subject":<12} {"True":>6} {"Pred":>6} {"PD Prob":>8} {"OK":>4}')
    print('-' * 40)
    for rec, pred, prob in zip(all_records, all_preds, all_probs):
        true_label = 'PD' if rec['label'] == 1 else 'HC'
        pred_label = 'PD' if pred == 1 else 'HC'
        ok = '✓' if pred == rec['label'] else '✗'
        print(f'{rec["subject_id"]:<12} {true_label:>6} {pred_label:>6} '
              f'{prob:>8.3f} {ok:>4}')


if __name__ == '__main__':
    main()