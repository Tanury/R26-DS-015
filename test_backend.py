import os
import pickle
import joblib
import traceback
from pathlib import Path

import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "app" / "models"

API_URL = "http://127.0.0.1:8000/predict"


REQUIRED_FILES = [
    "disease_model.pkl",
    "disease_label_encoder.pkl",
    "disease_num_imputer.pkl",
    "disease_cat_encoders.pkl",
    "disease_shap_importance.csv",

    "risk_model.pkl",
    "risk_label_mapping.pkl",
    "risk_num_imputer.pkl",
    "risk_cat_encoders.pkl",
    "risk_shap_importance.csv",
]


SAMPLE_INPUT = {
    "age": 68,
    "sex": "Female",

    "moca_total_score": 21,
    "updrs_part_i": 8,
    "updrs_part_ii": 13,
    "updrs_part_iii": 28,
    "updrs_part_iv": 4,
    "disease_duration_years": 4.5,

    "amyloid_beta_42_pg_ml": 420.5,
    "amyloid_beta_40_pg_ml": 6200,
    "p_tau181_pg_ml": 3.2,
    "t_tau_pg_ml": 280,
    "nfl_pg_ml": 18.5,
    "gfap_pg_ml": 140,
    "alpha_synuclein_pg_ml": 920,

    "neuroinflam_score": 0.65,
    "tau_amyloid_ratio": 0.43
}


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def test_required_files():
    print_section("1. Checking required model files")

    all_ok = True

    for filename in REQUIRED_FILES:
        path = MODEL_DIR / filename

        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"✅ Found: {filename} | {size_kb:.2f} KB")
        else:
            print(f"❌ Missing: {filename}")
            all_ok = False

    return all_ok


def load_pickle_file(filename):
    path = MODEL_DIR / filename
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


def test_pickle_loading():
    print_section("2. Testing pickle loading")

    pickle_files = [
        "disease_model.pkl",
        "disease_label_encoder.pkl",
        "disease_num_imputer.pkl",
        "disease_cat_encoders.pkl",
        "risk_model.pkl",
        "risk_label_mapping.pkl",
        "risk_num_imputer.pkl",
        "risk_cat_encoders.pkl",
    ]

    all_ok = True

    for filename in pickle_files:
        try:
            obj = load_pickle_file(filename)
            print(f"✅ Loaded: {filename} | Type: {type(obj)}")

            if hasattr(obj, "classes_"):
                print(f"   Classes: {list(obj.classes_)}")

            if isinstance(obj, dict):
                print(f"   Dict keys: {list(obj.keys())[:10]}")

        except Exception as e:
            print(f"❌ Failed loading: {filename}")
            print(f"   Error: {e}")
            all_ok = False

    return all_ok


def test_csv_files():
    print_section("3. Testing SHAP CSV files")

    csv_files = [
        "disease_shap_importance.csv",
        "risk_shap_importance.csv",
    ]

    all_ok = True

    for filename in csv_files:
        try:
            path = MODEL_DIR / filename
            df = pd.read_csv(path)

            print(f"✅ Loaded: {filename}")
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {list(df.columns)}")
            print("   First rows:")
            print(df.head(5).to_string(index=False))

            if "feature" not in df.columns:
                print(f"❌ {filename} missing column: feature")
                all_ok = False

            if "importance" not in df.columns and "mean_abs_shap" not in df.columns:
                print(f"❌ {filename} missing importance column")
                all_ok = False

        except Exception as e:
            print(f"❌ Failed reading: {filename}")
            print(f"   Error: {e}")
            all_ok = False

    return all_ok


def test_direct_service_import():
    print_section("4. Testing FastAPI service import")

    try:
        from app.services.prediction_service import NeurologicalPredictionService

        print("✅ Successfully imported NeurologicalPredictionService")

        service = NeurologicalPredictionService()
        print("✅ Successfully initialized prediction service")

        result = service.predict(SAMPLE_INPUT)
        print("✅ Direct service prediction successful")
        print("Result:")
        print(result)

        required_output_keys = [
            "predicted_disease",
            "disease_full_name",
            "disease_confidence",
            "predicted_risk",
            "risk_confidence",
            "disease_probabilities",
            "risk_probabilities",
            "conclusion",
            "clinical_recommendation",
            "important_features",
            "disclaimer",
        ]

        missing_keys = [key for key in required_output_keys if key not in result]

        if missing_keys:
            print(f"❌ Missing output keys: {missing_keys}")
            return False

        print("✅ Output structure is correct")
        return True

    except Exception:
        print("❌ Service import or prediction failed")
        traceback.print_exc()
        return False


def test_api_connection():
    print_section("5. Testing FastAPI /predict endpoint")

    try:
        response = requests.post(API_URL, json=SAMPLE_INPUT, timeout=20)

        print(f"Status code: {response.status_code}")

        if response.status_code != 200:
            print("❌ API request failed")
            print("Response text:")
            print(response.text)
            return False

        data = response.json()

        print("✅ API request successful")
        print("Response JSON:")
        print(data)

        return True

    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to FastAPI server.")
        print("Make sure you started the backend first:")
        print("python run.py")
        return False

    except Exception:
        print("❌ API test failed")
        traceback.print_exc()
        return False


def main():
    print_section("FASTAPI BACKEND DEBUG TEST")

    print(f"Backend root: {BASE_DIR}")
    print(f"Model folder: {MODEL_DIR}")

    results = {
        "required_files": test_required_files(),
        "pickle_loading": test_pickle_loading(),
        "csv_files": test_csv_files(),
        "direct_service": test_direct_service_import(),
        "api_connection": test_api_connection(),
    }

    print_section("FINAL TEST SUMMARY")

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")

    if all(results.values()):
        print("\n🎉 All tests passed. Backend is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the failed section above.")


if __name__ == "__main__":
    main()