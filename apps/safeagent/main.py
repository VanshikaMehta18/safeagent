"""SafeAgent FastAPI application entry point."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from api.routes import router
from config import get_settings
from retrieval.vector_store import VectorStore

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup."""
    settings = get_settings()
    logger.info("Starting SafeAgent v%s", settings.app_version)

    if settings.langsmith_tracing and settings.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project

    store = VectorStore()
    await store.ensure_index()
    app.state.vector_store = store
    logger.info("FAISS index ready at %s", settings.faiss_index_path)
    yield
    logger.info("Shutting down SafeAgent")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    application = FastAPI(
        title="SafeAgent",
        description="Agentic pipeline with constitutional safety layer",
        version=settings.app_version,
        lifespan=lifespan,
    )
    application.include_router(router)
    return application


app = create_app()
