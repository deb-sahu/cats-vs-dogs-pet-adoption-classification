"""FastAPI application for Cats vs Dogs classifier."""

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.responses import Response

from src.api.predict import Predictor
from src.api.schemas import ErrorResponse, HealthResponse, ModelInfoResponse, PredictionResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PREDICTION_REQUESTS = Counter(
    "prediction_requests_total", "Total number of prediction requests", ["status", "prediction"]
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction request latency in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
MODEL_LOADED = Gauge("model_loaded", "Whether the model is currently loaded (1=yes, 0=no)")
PREDICTION_CONFIDENCE = Histogram(
    "prediction_confidence",
    "Distribution of prediction confidence scores",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
)

predictor: Optional[Predictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown."""
    global predictor

    model_path = os.environ.get("MODEL_PATH", "models/model.pt")
    logger.info(f"Loading model from {model_path}...")

    predictor = Predictor()

    if Path(model_path).exists():
        success = predictor.load_model(model_path)
        if success:
            logger.info("Model loaded successfully")
            MODEL_LOADED.set(1)
        else:
            logger.warning("Failed to load model")
            MODEL_LOADED.set(0)
    else:
        logger.warning(f"Model file not found: {model_path}")
        MODEL_LOADED.set(0)

    yield

    logger.info("Shutting down...")


app = FastAPI(
    title="Cats vs Dogs Classifier API",
    description="Binary image classification API for pet adoption platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )

    return response


@app.get("/", tags=["Info"])
async def root():
    """API root with basic information."""
    return {
        "name": "Cats vs Dogs Classifier API",
        "version": "1.0.0",
        "description": "Binary image classification for pet adoption platform",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "metrics": "/metrics",
            "docs": "/docs",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint for Kubernetes liveness/readiness probes.

    Returns service status and whether the model is loaded.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=predictor.is_loaded() if predictor else False,
        version="1.0.0",
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    tags=["Prediction"],
)
async def predict(file: UploadFile = File(..., description="Image file to classify")):
    """
    Classify an uploaded image as cat or dog.

    Accepts JPEG or PNG images. Returns classification result with confidence scores.
    """
    if not predictor or not predictor.is_loaded():
        PREDICTION_REQUESTS.labels(status="error", prediction="none").inc()
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")

    if not file.content_type or not file.content_type.startswith("image/"):
        PREDICTION_REQUESTS.labels(status="error", prediction="none").inc()
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Please upload an image.",
        )

    start_time = time.time()

    try:
        contents = await file.read()

        result = predictor.predict_from_bytes(contents)

        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)
        PREDICTION_REQUESTS.labels(status="success", prediction=result["label"]).inc()
        PREDICTION_CONFIDENCE.observe(result["confidence"])

        logger.info(
            f"Prediction: {result['label']} "
            f"(confidence: {result['confidence']:.3f}, latency: {latency:.3f}s)"
        )

        return PredictionResponse(**result)

    except Exception as e:
        PREDICTION_REQUESTS.labels(status="error", prediction="none").inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """
    Prometheus metrics endpoint.

    Exposes application metrics for monitoring.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Info"])
async def model_info():
    """Get information about the loaded model."""
    return ModelInfoResponse(
        model_type="SimpleCNN", input_size=[224, 224], classes=["cat", "dog"], version="1.0.0"
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500, content={"error": "Internal server error", "detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
