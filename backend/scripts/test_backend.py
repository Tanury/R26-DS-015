"""Smoke-test a running backend using only the Python standard library."""

import argparse
import json
import logging
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SAMPLE_FEATURES = {
    "mfcc_1_mean": -245.2,
    "mfcc_2_mean": 82.1,
    "mfcc_3_mean": 14.7,
    "pitch_mean": 178.4,
    "pitch_std": 32.8,
    "jitter": 0.9,
    "shimmer": 3.1,
    "hnr": 18.6,
    "speech_rate": 2.4,
    "pause_count": 12,
    "mean_pause_duration": 0.42,
    "mean_energy": 61.3,
    "spectral_centroid_mean": 1840.5,
    "zero_crossing_rate_mean": 0.08,
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("backend-smoke-test")


def request_json(url: str, payload: dict[str, float] | None = None) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "X-Request-ID": "smoke-test"},
        method="POST" if data is not None else "GET",
    )
    with urlopen(request, timeout=30) as response:
        body = json.loads(response.read().decode("utf-8"))
        logger.info(
            "%s -> %s request_id=%s",
            url,
            response.status,
            response.headers.get("X-Request-ID"),
        )
        return body


def main() -> int:
    parser = argparse.ArgumentParser(description="Test health and prediction endpoints.")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    base_url = args.url.rstrip("/")

    try:
        health = request_json(f"{base_url}/health/")
        if health != {"status": "ok"}:
            raise RuntimeError(f"Unexpected health response: {health}")

        prediction = request_json(f"{base_url}/predictions/", SAMPLE_FEATURES)
        required = {
            "predicted_class",
            "confidence_score",
            "risk_score",
            "risk_level",
            "probabilities",
            "observed_issues",
            "recommendations",
            "disclaimer",
        }
        missing = sorted(required - set(prediction))
        if missing:
            raise RuntimeError(f"Prediction response is missing: {', '.join(missing)}")
        logger.info(
            "prediction=%s confidence=%s risk_level=%s",
            prediction["predicted_class"],
            prediction["confidence_score"],
            prediction["risk_level"],
        )
        return 0
    except HTTPError as exc:
        logger.error(
            "HTTP %s: %s",
            exc.code,
            exc.read().decode("utf-8", errors="replace"),
        )
    except (URLError, TimeoutError) as exc:
        logger.error("Backend is unreachable: %s", exc)
    except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
        logger.error("Smoke test failed: %s", exc)
    return 1


if __name__ == "__main__":
    sys.exit(main())
