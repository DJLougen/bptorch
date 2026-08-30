"""FastAPI application entrypoint for bpTorch."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from neural_blueprint.api.routes.models import router as models_router
from neural_blueprint.api.routes.registry import router as registry_router
from neural_blueprint.api.routes.samples import router as samples_router
from neural_blueprint.api.routes.sessions import router as sessions_router
from neural_blueprint.api.routes.sessions import ws_router
from neural_blueprint.api.routes.tester import router as tester_router


class HealthResponse(BaseModel):
    status: str
    version: str
    runtime: str
    torch_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup initialization
    yield
    # Teardown / cleanup


def create_app() -> FastAPI:
    app = FastAPI(
        title="bpTorch API",
        version="0.1.0",
        description="Executable Blueprint-style Neural-Network Architecture Runtime API",
        lifespan=lifespan,
    )

    allowed_origins = [
        origin.strip()
        for origin in os.getenv(
            "NEURAL_BLUEPRINT_CORS_ORIGINS",
            "http://127.0.0.1:5173,http://localhost:5173",
        ).split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(registry_router)
    app.include_router(models_router)
    app.include_router(sessions_router)
    app.include_router(tester_router)
    app.include_router(ws_router)
    app.include_router(samples_router)

    @app.get("/api/v1/health", response_model=HealthResponse)
    async def health():
        import torch

        return HealthResponse(
            status="ok",
            version="0.1.0",
            runtime="pytorch",
            torch_version=torch.__version__,
        )

    return app

app = create_app()
