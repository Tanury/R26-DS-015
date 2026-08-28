"""Load the exported R26-DS-015 EEG encoder and score one subject.

TorchScript needs no project code — only torch. Run from the folder that holds
the exported files.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
bundle = joblib.load(HERE / "neuro_risk_inference_bundle.joblib")
card = json.loads((HERE / "model_card.json").read_text(encoding="utf-8"))

model = torch.jit.load(str(HERE / "neuro_risk_encoder.torchscript.pt"))
model.eval()

planes, channels, times = bundle["input_shape"]
print(f"expects [batch, {planes}, {channels}, {times}]")
print(f"channel order : {bundle['channel_order'][:6]} ... ({len(bundle['channel_order'])} total)")
print(f"risk order    : {bundle['risk_conditions']}")

# Replace with real preprocessed epochs: shape [n_epochs, planes, channels, times],
# built with the settings in bundle["preprocessing_config"].
epochs = torch.randn(8, planes, channels, times)

with torch.no_grad():
    risk_scores, class_probabilities, z_eeg = model(epochs)

subject_risk = risk_scores.mean(dim=0)          # aggregate epochs -> subject
subject_z = z_eeg.mean(dim=0)
subject_z = subject_z / subject_z.norm().clamp_min(1e-12)

print("\nrisk scores (independent sigmoids, do not sum to 1):")
for name, value in zip(bundle["risk_conditions"], subject_risk.tolist()):
    band = ("Low" if value <= bundle["risk_bands"]["low_max"]
            else "Medium" if value <= bundle["risk_bands"]["medium_max"] else "High")
    print(f"  {name}-related EEG risk pattern: {value:.4f}  ({band})")

print(f"\nz_eeg: {tuple(subject_z.shape)}, L2 norm {float(subject_z.norm()):.4f}")
print("\n" + card["confound_disclosure"]["statement"])
print("\n" + card["intended_use"]["disclaimer"])
