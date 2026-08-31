import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[1]   # .../final_neuro/backend
_REPO_ROOT = _BACKEND_DIR.parent                      # .../final_neuro
for _p in (str(_REPO_ROOT), str(_BACKEND_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import logging
import re
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.api.routes.neurological_prediction import router as neurological_prediction_router
from app.core.config import settings
from app.core.logging import configure_logging

#new - tanuri

from backend.routers.analyze import router as neuroimaging_ad_router
from backend.routers.preprocess import router as neuroimaging_preprocess_router
from backend.routers.analyze_pd import router as neuroimaging_pd_router
from backend.routers.analyze_ms import router as neuroimaging_ms_router
#end of new - tanuri

configure_logging()
logger = logging.getLogger(__name__)
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Speech-feature-based neurological risk assessment API.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.frontend_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-Request-ID"],
    expose_headers=["X-Request-ID"],
)
app.include_router(api_router)

#new - tanuri
app.include_router(neuroimaging_ad_router, prefix="/api")
app.include_router(neuroimaging_preprocess_router, prefix="/api")
app.include_router(neuroimaging_pd_router, prefix="/api")
app.include_router(neuroimaging_ms_router, prefix="/api")
#end of new - tanuri

app.include_router(
    neurological_prediction_router,
    prefix="/neurological-risk",
    tags=["neurological-risk"],
)


def _request_id(request: Request) -> str:
    supplied_id = request.headers.get("X-Request-ID", "")
    if REQUEST_ID_PATTERN.fullmatch(supplied_id):
        return supplied_id
    return str(uuid4())


@app.middleware("http")
async def log_request(request: Request, call_next):
    request_id = _request_id(request)
    started_at = perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (perf_counter() - started_at) * 1000
        logger.exception(
            "request_failed request_id=%s method=%s path=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            elapsed_ms,
        )
        raise
    elapsed_ms = (perf_counter() - started_at) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request_complete request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response
