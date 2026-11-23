"""FastAPI Application Server for TransitPulse-AI Platform."""

import time
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from api.routes.frontend import HTML_CONTENT, router as frontend_router
from api.routes.monitoring import router as monitoring_router
from api.routes.predict import router as predict_router
from api.routes.routes_stops import router as routes_router
from api.routes.scenario import router as scenario_router
from transitpulse.config import settings
from transitpulse.utils.logger import get_logger

logger = get_logger("transitpulse.api")

app = FastAPI(
    title="TransitPulse-AI API",
    description=(
        "Production REST API for spatiotemporal transit delay prediction, "
        "meteorological and event shockwave quantification, scenario simulation, "
        "and SHAP game-theoretic explainability in Bengaluru, India."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Measures and logs request processing latency."""
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time_ms = (time.perf_counter() - start_time) * 1000.0
    response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"
    return response


@app.get("/health", tags=["System"])
def health_check():
    """System liveness and readiness probe."""
    return {
        "status": "healthy",
        "service": "transitpulse-api",
        "timestamp": datetime.now().isoformat(),
        "environment": settings.ENVIRONMENT,
        "version": "1.0.0",
        "city": settings.CITY_NAME,
        "timezone": settings.TIMEZONE,
    }


@app.get("/", tags=["System"])
def root(request: Request):
    """API Root with service overview and routing catalog (serves interactive UI for browsers)."""
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return HTMLResponse(content=HTML_CONTENT)
    return {
        "service": "TransitPulse-AI API",
        "description": "Bengaluru Transit Delay Prediction & Spatiotemporal Analysis Platform",
        "version": "1.0.0",
        "status": "online",
        "web_app": "/app",
        "documentation": "/docs",
        "health": "/health",
        "city": settings.CITY_NAME,
        "environment": settings.ENVIRONMENT,
        "dataset_mode": "simulated",
        "endpoints": {
            "web_app": "/app",
            "health": "/health",
            "docs": "/docs",
            "routes": "/api/v1/routes",
            "stops": "/api/v1/stops",
            "predict": "/api/v1/predict",
            "scenario": "/api/v1/scenario",
            "model_info": "/api/v1/model/info",
            "model_metrics": "/api/v1/model/metrics",
        },
    }


# Include Modular Routers
app.include_router(frontend_router)
app.include_router(routes_router)
app.include_router(predict_router)
app.include_router(scenario_router)
app.include_router(monitoring_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=False)
