import pandas as pd
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

DENOISED_DIR   = Path('/Users/tanuridissanayaka/research/R26-DS-015/mri/alzheimers/data/denoised')
COLLECTION_CSV = Path('/Users/tanuridissanayaka/research/R26-DS-015/mri/alzheimers/data/raw/collection.csv')
CLASSES = ['AD', 'MCI', 'CN']

print("\n" + "="*55)
print("  R26-DS-015 Alzheimer's Data Validation")
print("="*55)

# ── 1. Class distribution ──────────────────────────────────
print("\n[1] Class distribution:")
for cls in CLASSES:
    n = len(list((DENOISED_DIR / cls).glob('*.nii.gz')))
    print(f"    {cls}: {n} files")

# ── 2. Label consistency vs CSV ───────────────────────────
print("\n[2] Label consistency check:")
if COLLECTION_CSV.exists():
    df = pd.read_csv(COLLECTION_CSV)
    sub_col = next((c for c in ['Subject','subject_id'] if c in df.columns), None)
    grp_col = next((c for c in ['Group','group','DX'] if c in df.columns), None)
    if sub_col and grp_col:
        label_map = dict(zip(df[sub_col].astype(str), df[grp_col].astype(str)))
        mismatches = []
        for cls in CLASSES:
            for f in (DENOISED_DIR / cls).glob('*.nii.gz'):
                parts = f.name.split('_')
                if len(parts) >= 4:
                    sid = f'{parts[1]}_{parts[2]}_{parts[3]}'
                    true = label_map.get(sid)
                    if true and true != cls:
                        mismatches.append((f.name, cls, true))
        if mismatches:
            print(f"    ❌ {len(mismatches)} mismatches found:")
            for name, folder, csv_label in mismatches[:10]:
                print(f"       {name} → in {folder}/ but CSV says {csv_label}")
        else:
            print("    ✅ All labels match CSV")
else:
    print("    ⚠️  collection.csv not found — skipping")

# ── 3. Duplicate subjects ─────────────────────────────────
print("\n[3] Multi-visit subjects:")
all_subjects = []
for cls in CLASSES:
    for f in (DENOISED_DIR / cls).glob('*.nii.gz'):
        parts = f.name.split('_')
        if len(parts) >= 4:
            sid = f'{parts[1]}_{parts[2]}_{parts[3]}'
            all_subjects.append((sid, cls, f.name))

counts = Counter(s for s, _, _ in all_subjects)
multi = {k: v for k, v in counts.items() if v > 1}
print(f"    Subjects with multiple scans: {len(multi)}")
print(f"    ⚠️  Ensure train/test split is subject-level to avoid leakage")

# ── 4. Volume shapes ──────────────────────────────────────
print("\n[4] Volume shape check:")
shapes = {}
for cls in CLASSES:
    for f in (DENOISED_DIR / cls).glob('*.nii.gz'):
        try:
            shape = nib.load(str(f)).shape
            shapes.setdefault(shape, []).append(cls)
        except Exception as e:
            print(f"    ❌ Could not load {f.name}: {e}")

for shape, classes in sorted(shapes.items(), key=lambda x: -len(x[1])):
    print(f"    {shape}: {len(classes)} files  ({dict(Counter(classes))})")

# ── 5. Empty/near-empty volumes ───────────────────────────
print("\n[5] Empty volume check (< 5% non-zero voxels):")
empties = []
for cls in CLASSES:
    for f in (DENOISED_DIR / cls).glob('*.nii.gz'):
        try:
            data = nib.load(str(f)).get_fdata()
            pct = (data > 0).sum() / data.size * 100
            if pct < 5:
                empties.append((f.name, cls, round(pct, 1)))
        except Exception:
            pass
if empties:
    print(f"    ❌ {len(empties)} near-empty volumes found:")
    for name, cls, pct in empties:
        print(f"       {cls}/{name}: {pct}% non-zero  ← remove this file")
else:
    print("    ✅ No empty volumes found")

# ── 6. Intensity stats ────────────────────────────────────
print("\n[6] Intensity statistics (mean of non-zero voxels per class):")
for cls in CLASSES:
    means = []
    for f in (DENOISED_DIR / cls).glob('*.nii.gz'):
        try:
            data = nib.load(str(f)).get_fdata()
            nz = data[data > 0]
            if len(nz) > 0:
                means.append(nz.mean())
        except Exception:
            pass
    if means:
        print(f"    {cls}: mean={np.mean(means):.1f}  "
              f"std={np.std(means):.1f}  "
              f"min={np.min(means):.1f}  "
              f"max={np.max(means):.1f}")

# ── 7. Visual spot check ──────────────────────────────────
print("\n[7] Saving visual spot check images...")
for cls in CLASSES:
    files = list((DENOISED_DIR / cls).glob('*.nii.gz'))[:3]
    if not files:
        continue
    fig, axes = plt.subplots(len(files), 3,
                             figsize=(12, 4*len(files)),
                             facecolor='#0a0a0a')
    if len(files) == 1:
        axes = [axes]
    for row, f in enumerate(files):
        data = nib.load(str(f)).get_fdata().astype(np.float32)
        nz = data[data > 0]
        if len(nz) > 0:
            p1, p99 = np.percentile(nz, [1, 99])
            data = np.clip(data, p1, p99)
            data = (data - p1) / (p99 - p1 + 1e-8)
        cx, cy, cz = [s//2 for s in data.shape]
        for col, (sl, title) in enumerate([
            (np.rot90(data[:, :, cz]), 'Axial'),
            (np.rot90(data[:, cy, :]), 'Coronal'),
            (np.rot90(data[cx, :, :]), 'Sagittal'),
        ]):
            axes[row][col].imshow(sl, cmap='gray', aspect='auto')
            axes[row][col].set_title(
                f'{cls} — {title}\n{f.name[:30]}',
                color='#aaa', fontsize=8
            )
            axes[row][col].axis('off')
    plt.suptitle(f'{cls} — sample preprocessing outputs', color='white')
    plt.tight_layout()
    #out = f'mri/alzheimers/validation_{cls}.png'
    out = f'/Users/tanuridissanayaka/research/research_parent/final_neuro/mri/alzheimers/validation_{cls}.png'
    plt.savefig(out, dpi=100, facecolor='#0a0a0a')
    plt.close()
    print(f"    Saved: {out}")

print("\n" + "="*55)
print("  Validation complete")
print("="*55 + "\n")