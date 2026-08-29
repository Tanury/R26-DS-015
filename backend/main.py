"""
main.py — R26-DS-015 Vision Encoder API
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers.analyze import router as analyze_ad_router
from backend.routers.preprocess import router as preprocess_router
from backend.routers.analyze_pd import router as analyze_pd_router
from backend.routers.analyze_ms import router as analyze_ms_eye_router

app = FastAPI(
    title="R26-DS-015 Vision Encoder API",
    description="Neurological Risk Assessment — Retinal & Brain MRI & DaTscan Feature Extraction",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze_ad_router,    prefix="/api")
app.include_router(preprocess_router, prefix="/api")
app.include_router(analyze_pd_router, prefix="/api")
app.include_router(analyze_ms_eye_router,    prefix="/api")  # MS endpoints are under /api/ms/analyze

@app.get("/")
def root():
    return {"project": "R26-DS-015", "status": "running", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok"}