"""
models/__init__.py
──────────────────
Singleton model loader.
All pkl artifacts are loaded ONCE at startup and reused across requests.
"""

import joblib
import pandas as pd
from pathlib import Path
from app.config import (
    DISEASE_MODEL_PATH, DISEASE_LABEL_ENC_PATH,
    DISEASE_CAT_ENC_PATH, DISEASE_IMPUTER_PATH, DISEASE_SHAP_PATH,
    RISK_MODEL_PATH, RISK_LABEL_MAP_PATH,
    RISK_CAT_ENC_PATH, RISK_IMPUTER_PATH, RISK_SHAP_PATH,
)


class ModelStore:
    """
    Holds all loaded models and preprocessing artifacts in memory.
    Instantiated once at application startup via load_models().
    """

    def __init__(self):
        # ── Disease Model ──────────────────────────────────────────────────
        self.disease_model      = None
        self.disease_label_enc  = None   # sklearn LabelEncoder
        self.disease_cat_enc    = None   # dict of {col: LabelEncoder}
        self.disease_imputer    = None   # sklearn SimpleImputer
        self.disease_shap_df    = None   # pd.DataFrame with feature importance

        # ── Risk Model ─────────────────────────────────────────────────────
        self.risk_model         = None
        self.risk_label_map     = None   # {"Low": 0, "Medium": 1, "High": 2}
        self.risk_cat_enc       = None
        self.risk_imputer       = None
        self.risk_shap_df       = None

        self._loaded = False

    def load(self):
        """Load all artifacts from disk. Called once at startup."""
        if self._loaded:
            return

        self._check_files()

        self.disease_model     = joblib.load(DISEASE_MODEL_PATH)
        self.disease_label_enc = joblib.load(DISEASE_LABEL_ENC_PATH)
        self.disease_cat_enc   = joblib.load(DISEASE_CAT_ENC_PATH)
        self.disease_imputer   = joblib.load(DISEASE_IMPUTER_PATH)
        self.disease_shap_df   = pd.read_csv(DISEASE_SHAP_PATH)

        self.risk_model        = joblib.load(RISK_MODEL_PATH)
        self.risk_label_map    = joblib.load(RISK_LABEL_MAP_PATH)
        self.risk_cat_enc      = joblib.load(RISK_CAT_ENC_PATH)
        self.risk_imputer      = joblib.load(RISK_IMPUTER_PATH)
        self.risk_shap_df      = pd.read_csv(RISK_SHAP_PATH)

        self._loaded = True
        print("✅ All models loaded successfully")

    def _check_files(self):
        """Raise a clear error if any model file is missing."""
        required = [
            DISEASE_MODEL_PATH, DISEASE_LABEL_ENC_PATH,
            DISEASE_CAT_ENC_PATH, DISEASE_IMPUTER_PATH, DISEASE_SHAP_PATH,
            RISK_MODEL_PATH, RISK_LABEL_MAP_PATH,
            RISK_CAT_ENC_PATH, RISK_IMPUTER_PATH, RISK_SHAP_PATH,
        ]
        missing = [str(p) for p in required if not Path(p).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing model files — run training notebooks first:\n"
                + "\n".join(f"  ✗ {m}" for m in missing)
            )

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# ── Global singleton ──────────────────────────────────────────────────────────
model_store = ModelStore()
