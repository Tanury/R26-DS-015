"""Build a platform-compatible artifact from the final training notebook."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


RANDOM_STATE = 42
TARGET_COLUMN = "label"
EXCLUDED_COLUMNS = ["sample_id", "synthetic_flag", TARGET_COLUMN]
OUTPUT_FILENAME = "neurological_risk_runtime_model.joblib"


def build_model() -> Path:
    backend_dir = Path(__file__).resolve().parents[1]
    data = pd.read_csv(backend_dir / "neuro_speech.csv")
    feature_columns = [
        column for column in data.columns if column not in EXCLUDED_COLUMNS
    ]
    categorical_columns = [
        column
        for column in feature_columns
        if pd.api.types.is_object_dtype(data[column])
        or pd.api.types.is_string_dtype(data[column])
    ]
    numeric_columns = [
        column
        for column in feature_columns
        if column not in categorical_columns
    ]

    features = data[feature_columns]
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(data[TARGET_COLUMN])
    train_features, _, train_labels, _ = train_test_split(
        features,
        labels,
        test_size=0.30,
        stratify=labels,
        random_state=RANDOM_STATE,
    )

    preprocessor = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns,
            ),
            (
                "num",
                SimpleImputer(strategy="median"),
                numeric_columns,
            ),
        ]
    )
    classifier = VotingClassifier(
        estimators=[
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=157,
                    max_depth=6,
                    min_samples_leaf=5,
                    max_features="sqrt",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
            (
                "xgb",
                xgb.XGBClassifier(
                    n_estimators=246,
                    max_depth=2,
                    learning_rate=np.float64(0.07442460308702238),
                    subsample=np.float64(0.7253152871382157),
                    colsample_bytree=np.float64(0.7439761626945283),
                    reg_lambda=np.float64(8.537614045478822),
                    reg_alpha=np.float64(0.053059844639466114),
                    random_state=RANDOM_STATE,
                    eval_metric="mlogloss",
                ),
            ),
            (
                "lr",
                LogisticRegression(
                    max_iter=3000,
                    C=0.2,
                    random_state=RANDOM_STATE,
                ),
            ),
        ],
        voting="soft",
    )
    pipeline = Pipeline(
        [
            ("pre", preprocessor),
            ("clf", classifier),
        ]
    )
    pipeline.fit(train_features, train_labels)

    output_path = backend_dir / "app" / "models" / OUTPUT_FILENAME
    joblib.dump(
        {
            "pipeline": pipeline,
            "label_encoder": label_encoder,
        },
        output_path,
    )

    loaded = joblib.load(output_path)
    loaded["pipeline"].predict_proba(train_features.iloc[[0]])
    return output_path


if __name__ == "__main__":
    model_path = build_model()
    print(f"Runtime model saved and verified: {model_path}")
