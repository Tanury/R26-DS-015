from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import NeurologicalInput, PredictionResponse
from app.services.prediction_service import NeurologicalPredictionService


app = FastAPI(
    title="Neurological Risk Assessment API",
    description=(
        "Disease and risk prediction API for Alzheimer’s Disease, Parkinson’s Disease, "
        "Multiple Sclerosis, and Control using clinical and fluid biomarker features."
    ),
    version="1.0.0",
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

prediction_service = NeurologicalPredictionService()


@app.get("/")
def root():
    return {
        "message": "Neurological Risk Assessment API is running",
        "status": "active",
        "docs": "/docs",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": True,
        "available_predictions": [
            "Disease classification: AD / PD / MS / Control",
            "Risk classification: Low / Medium / High",
        ],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_neurological_risk(input_data: NeurologicalInput):
    try:
        data = input_data.model_dump()
        result = prediction_service.predict(data)
        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )