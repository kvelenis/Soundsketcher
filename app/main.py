from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api import audio, experiments, pages, questionnaires
from app.core.config import Settings, get_settings
from app.core.prefix import BasePathMiddleware

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    # Do not pre-warm MATLAB here.  The MATLAB engine can load its bundled
    # libexpat before matplotlib/MOSQITO imports pyexpat, which then crashes
    # MOSQITO with an undefined XML_SetAllocTrackerActivationThreshold symbol.
    # Feature extraction imports MOSQITO before starting MATLAB, preserving the
    # safer library load order while keeping the showcase routes lightweight.
    logger.info("Startup kept lightweight; MATLAB starts lazily during feature extraction.")
    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.state.settings = settings
    if settings.base_path:
        app.add_middleware(BasePathMiddleware, base_path=settings.base_path)

    if settings.sandbox_static_dir.exists():
        app.mount(
            "/sandbox-static",
            StaticFiles(directory=str(settings.sandbox_static_dir)),
            name="sandbox_static",
        )

    app.mount("/static", StaticFiles(directory=str(settings.static_dir)), name="static")

    if settings.static_upload_dir.exists():
        app.mount(
            "/static_uploads",
            StaticFiles(directory=str(settings.static_upload_dir)),
            name="static_uploads",
        )

    settings.cache_root.mkdir(parents=True, exist_ok=True)
    app.mount("/user_data", StaticFiles(directory=str(settings.cache_root)), name="uploads")

    app.include_router(pages.router)
    app.include_router(audio.router)
    app.include_router(questionnaires.router)
    app.include_router(experiments.router)

    return app


app = create_app()
