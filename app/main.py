from pathlib import Path
import shutil
import uuid
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.services.audio_features import extract_audio_features
from app.services.audio_predictor import predict_audio_features

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app.main")


app = FastAPI(
    title="Dementia Speech Risk Screening API",
    description="Research prototype for speech-based dementia risk screening. Not for clinical diagnosis.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health")
def health_check():
    return {
        "status": "ok",
        "message": "Dementia speech backend is running",
        "clinical_note": "Research prototype only. Not a clinical diagnosis."
    }


@app.post("/api/v1/predict/audio-file")
async def predict_from_audio_file(file: UploadFile = File(...)):
    temp_file_path = None
    try:
        logger.debug("Received audio upload: %s", file.filename)
        allowed_extensions = [".mp3", ".wav", ".m4a", ".flac"]
        file_ext = Path(file.filename).suffix.lower()

        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail="Only .mp3, .wav, .m4a, and .flac audio files are supported."
            )

        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)

        temp_file_name = f"{uuid.uuid4()}{file_ext}"
        temp_file_path = temp_dir / temp_file_name

        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        features = extract_audio_features(str(temp_file_path))

        result = predict_audio_features(
            features=features,
            file_id=file.filename
        )

        logger.debug("Prediction result for %s: %s", file.filename, {"predicted": result.get("prediction", {})})

        return result

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        await file.close()
        if temp_file_path is not None:
            temp_file_path.unlink(missing_ok=True)