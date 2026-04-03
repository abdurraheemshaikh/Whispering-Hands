from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.service import HandSignPredictor


class PredictRequest(BaseModel):
    image: str = Field(..., description="Base64 or data URL encoded image")


class PredictResponse(BaseModel):
    gesture: str | None
    confidence: float | None

app = FastAPI(title="Whispering Hands API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_predictor() -> HandSignPredictor:
    return HandSignPredictor()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    try:
        result = get_predictor().predict_image(payload.image)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - safety net for model/runtime errors
        raise HTTPException(status_code=500, detail="Prediction failed") from exc

    return PredictResponse(gesture=result.gesture, confidence=result.confidence)
