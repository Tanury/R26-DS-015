import pickle
import joblib
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from app.config import (
    DISEASE_MODEL_PATH,
    DISEASE_LABEL_ENCODER_PATH,
    DISEASE_NUM_IMPUTER_PATH,
    DISEASE_CAT_ENCODERS_PATH,
    DISEASE_SHAP_PATH,
    RISK_MODEL_PATH,
    RISK_LABEL_MAPPING_PATH,
    RISK_NUM_IMPUTER_PATH,
    RISK_CAT_ENCODERS_PATH,
    RISK_SHAP_PATH,
    ALL_FEATURE_COLS,
    CATEGORICAL_COLS,
    NUMERIC_COLS,
)

from app.services.feature_service import (
    get_disease_full_name,
    get_feature_type,
    format_feature_name,
)
from app.services.conclusion_service import generate_final_conclusion


class NeurologicalPredictionService:
    def __init__(self):
        # Disease classifier artifacts
        self.disease_model = self._load_pickle(DISEASE_MODEL_PATH)
        self.disease_label_encoder = self._load_pickle(DISEASE_LABEL_ENCODER_PATH)
        self.disease_num_imputer = self._load_pickle(DISEASE_NUM_IMPUTER_PATH)
        self.disease_cat_encoders = self._load_pickle(DISEASE_CAT_ENCODERS_PATH)

        # Risk classifier artifacts
        self.risk_model = self._load_pickle(RISK_MODEL_PATH)
        self.risk_label_mapping = self._load_pickle(RISK_LABEL_MAPPING_PATH)
        self.risk_num_imputer = self._load_pickle(RISK_NUM_IMPUTER_PATH)
        self.risk_cat_encoders = self._load_pickle(RISK_CAT_ENCODERS_PATH)

        # SHAP importance files
        self.disease_shap_df = self._load_shap_csv(DISEASE_SHAP_PATH)
        self.risk_shap_df = self._load_shap_csv(RISK_SHAP_PATH)

        self.input_features = ALL_FEATURE_COLS
        self.categorical_features = CATEGORICAL_COLS
        self.numerical_features = NUMERIC_COLS

    def _load_pickle(self, path):
        if not path.exists():
            raise FileNotFoundError(f"Required model artifact not found: {path}")

        # Prefer joblib for sklearn/xgboost artifacts; fall back to raw pickle.
        try:
            return joblib.load(path)
        except Exception:
            with open(path, "rb") as f:
                return pickle.load(f)

    def _load_shap_csv(self, path):
        if not path.exists():
            return pd.DataFrame(columns=["feature", "importance"])

        df = pd.read_csv(path)

        # Support both possible column names
        if "importance" not in df.columns:
            if "mean_abs_shap" in df.columns:
                df = df.rename(columns={"mean_abs_shap": "importance"})

        if "feature" not in df.columns or "importance" not in df.columns:
            return pd.DataFrame(columns=["feature", "importance"])

        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def _prepare_dataframe(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        clean_data = {feature: input_data.get(feature, None) for feature in self.input_features}

        # Fill missingness flags if not provided
        for feature in self.input_features:
            if feature.endswith("_missing") and clean_data.get(feature) is None:
                base_feature = feature[: -len("_missing")]
                clean_data[feature] = 1 if input_data.get(base_feature) is None else 0

        return pd.DataFrame([clean_data])

    def _apply_preprocessing(
        self,
        df: pd.DataFrame,
        num_imputer,
        cat_encoders,
    ) -> pd.DataFrame:
        processed = df.copy()

        # Ensure all expected columns exist
        for feature in self.input_features:
            if feature not in processed.columns:
                processed[feature] = None

        # Categorical encoding
        for col in self.categorical_features:
            processed[col] = processed[col].fillna("Missing").astype(str)

            if isinstance(cat_encoders, dict) and col in cat_encoders:
                encoder = cat_encoders[col]

                if hasattr(encoder, "classes_"):
                    known_classes = set(encoder.classes_)
                    processed[col] = processed[col].apply(
                        lambda x: x if x in known_classes else encoder.classes_[0]
                    )
                    processed[col] = encoder.transform(processed[col])
                elif isinstance(encoder, dict):
                    processed[col] = processed[col].map(encoder).fillna(0)
                else:
                    processed[col] = 0
            else:
                processed[col] = processed[col].map(
                    {"Male": 0, "Female": 1, "Other": 2, "Unknown": 0, "Missing": 0}
                ).fillna(0)

        # Numerical imputation
        if self.numerical_features:
            if not hasattr(num_imputer, "_fill_dtype") and hasattr(num_imputer, "_fit_dtype"):
                # Backward-compat for imputers persisted with older sklearn versions.
                num_imputer._fill_dtype = num_imputer._fit_dtype
            processed[self.numerical_features] = num_imputer.transform(
                processed[self.numerical_features]
            )

        return processed[self.input_features]

    def _decode_disease_label(self, encoded_prediction) -> str:
        if hasattr(self.disease_label_encoder, "inverse_transform"):
            return self.disease_label_encoder.inverse_transform([encoded_prediction])[0]

        if isinstance(self.disease_label_encoder, dict):
            reverse_mapping = {v: k for k, v in self.disease_label_encoder.items()}
            return reverse_mapping.get(encoded_prediction, str(encoded_prediction))

        return str(encoded_prediction)

    def _decode_risk_label(self, encoded_prediction) -> str:
        if hasattr(self.risk_label_mapping, "inverse_transform"):
            return self.risk_label_mapping.inverse_transform([encoded_prediction])[0]

        if isinstance(self.risk_label_mapping, dict):
            # Could be {0:"Low",1:"Medium",2:"High"} or {"Low":0,...}
            if encoded_prediction in self.risk_label_mapping:
                return self.risk_label_mapping[encoded_prediction]

            reverse_mapping = {v: k for k, v in self.risk_label_mapping.items()}
            return reverse_mapping.get(encoded_prediction, str(encoded_prediction))

        return str(encoded_prediction)

    def _predict_probabilities(
        self,
        model,
        X: pd.DataFrame,
        label_decoder,
        class_source=None,
    ) -> Tuple[str, float, Dict[str, float]]:
        encoded_pred = model.predict(X)[0]
        pred_label = label_decoder(encoded_pred)

        probabilities = {}

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]

            if hasattr(model, "classes_"):
                encoded_classes = list(model.classes_)
                class_labels = [label_decoder(cls) for cls in encoded_classes]
            elif class_source is not None and hasattr(class_source, "classes_"):
                class_labels = list(class_source.classes_)
            else:
                class_labels = [str(i) for i in range(len(probs))]

            probabilities = {
                str(class_labels[i]): round(float(probs[i]) * 100, 2)
                for i in range(len(probs))
            }

            confidence = round(float(np.max(probs)) * 100, 2)
        else:
            confidence = 0.0

        return str(pred_label), confidence, probabilities

    def _get_important_features(self, limit: int = 5):
        # Use risk SHAP for final explanation because the final clinical decision is risk-based.
        shap_df = self.risk_shap_df

        if shap_df.empty:
            shap_df = self.disease_shap_df

        top = shap_df.head(limit)

        important_features = []

        for _, row in top.iterrows():
            feature = str(row["feature"])
            important_features.append(
                {
                    "feature": feature,
                    "display_name": format_feature_name(feature),
                    "feature_type": get_feature_type(feature),
                    "importance": round(float(row["importance"]), 6),
                }
            )

        return important_features

    def _align_features(self, X: pd.DataFrame, model) -> pd.DataFrame:
        feature_names = None

        if hasattr(model, "feature_names_in_"):
            feature_names = list(model.feature_names_in_)
        elif hasattr(model, "estimators_"):
            for estimator in model.estimators_:
                if hasattr(estimator, "feature_names_in_"):
                    feature_names = list(estimator.feature_names_in_)
                    break

        if feature_names:
            for name in feature_names:
                if name not in X.columns:
                    X[name] = 0

            X = X.reindex(columns=feature_names, fill_value=0)

        return X

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        df = self._prepare_dataframe(input_data)

        # Disease preprocessing + prediction
        X_disease = self._apply_preprocessing(
            df=df,
            num_imputer=self.disease_num_imputer,
            cat_encoders=self.disease_cat_encoders,
        )
        X_disease = self._align_features(X_disease, self.disease_model)

        predicted_disease, disease_confidence, disease_probabilities = (
            self._predict_probabilities(
                model=self.disease_model,
                X=X_disease,
                label_decoder=self._decode_disease_label,
                class_source=self.disease_label_encoder,
            )
        )

        # Risk preprocessing + prediction
        X_risk = self._apply_preprocessing(
            df=df,
            num_imputer=self.risk_num_imputer,
            cat_encoders=self.risk_cat_encoders,
        )
        X_risk = self._align_features(X_risk, self.risk_model)

        predicted_risk, risk_confidence, risk_probabilities = (
            self._predict_probabilities(
                model=self.risk_model,
                X=X_risk,
                label_decoder=self._decode_risk_label,
                class_source=self.risk_label_mapping,
            )
        )

        disease_full_name = get_disease_full_name(predicted_disease)
        important_features = self._get_important_features(limit=5)

        conclusion_data = generate_final_conclusion(
            predicted_disease=predicted_disease,
            disease_full_name=disease_full_name,
            predicted_risk=predicted_risk,
            risk_confidence=risk_confidence,
            important_features=important_features,
        )

        return {
            "predicted_disease": predicted_disease,
            "disease_full_name": disease_full_name,
            "disease_confidence": disease_confidence,
            "predicted_risk": predicted_risk,
            "risk_confidence": risk_confidence,
            "disease_probabilities": disease_probabilities,
            "risk_probabilities": risk_probabilities,
            "conclusion": conclusion_data["conclusion"],
            "clinical_recommendation": conclusion_data["clinical_recommendation"],
            "important_features": important_features,
            "embedding_info": {
                "z_bio_dimension": 256,
                "embedding_available": True,
                "embedding_usage": (
                    "The biomarker representation is used internally for neurological "
                    "risk assessment and future multimodal fusion."
                ),
            },
            "model_summary": {
                "disease_model": "disease_model.pkl",
                "risk_model": "risk_model.pkl",
                "prediction_type": (
                    "Disease classification + risk severity classification"
                ),
            },
            "disclaimer": (
                "This is a clinical decision-support output, not a final medical diagnosis."
            ),
        }