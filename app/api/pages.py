import os

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app.core.config import get_settings
from app.core.templates import get_templates
from app.core.templates import app_base_path, app_path


router = APIRouter(tags=["pages"])
templates = get_templates()


PAGE_ROUTES = {
    "/": "index.html",
    "/objectifier-upload-page": "objectifier-upload-page.html",
    "/analyze": "analyze.html",
    "/wav2vec": "wav2vec.html",
    "/clip-clap": "clip-clap.html",
    "/high_level_features": "high_level_features.html",
    "/high_level_mosqito_2": "high_level_mosqito_2.html",
    "/sound_image_shape": "sound_image_shape.html",
    "/sound_image_shape_stimuli": "sound_image_shape_stimuli.html",
    "/sound_image_texture": "sound_image_texture.html",
    "/sound_indefinite_pilot": "noisy_peak_experiment_ui.html",
    "/sound_indefinite_pilot_2": "pairwise_pitch_experiment.html",
    "/sound_indefinite_pilot_3": "pairwise_pitch_experiment_2_balanced.html",
    "/evaluation": "soundsketcher_evaluation.html",
    "/questionnaire-soundsketcher": "soundsketcher-questionaire.html",
    "/experiments": "experiments.html",
    "/noise_tonal_preference": "noise_tonal_preference_showcase.html",
    "/indefinite_pitch": "indefinite_pitch.html",
    "/test": "test.html",
    "/indefinite_pitch_results": "indefinite_pitch_results.html",
    "/sound_image_shape_results": "sound_image_shape_results.html",
    "/sound_image_texture_results": "sound_image_texture_results.html",
}


@router.get("/healthz")
async def healthz():
    return {"status": "ok"}


def _path_check(path, *, expect_file: bool = False, writable: bool = False) -> dict:
    exists = path.exists()
    is_expected_type = path.is_file() if expect_file else path.is_dir()
    result = {
        "exists": exists,
        "ok": exists and is_expected_type,
    }
    if writable:
        result["writable"] = exists and path.is_dir() and os.access(path, os.W_OK)
        result["ok"] = result["ok"] and result["writable"]
    return result


@router.get("/deployment-info")
async def deployment_info(request: Request):
    settings = get_settings()
    checks = {
        "templates": _path_check(settings.templates_dir),
        "static": _path_check(settings.static_dir),
        "sandbox_static": _path_check(settings.sandbox_static_dir),
        "cache_root": _path_check(settings.cache_root, writable=True),
        "main_module": _path_check(
            settings.sandbox_static_dir / "js" / "main.module.mjs",
            expect_file=True,
        ),
        "app_css": _path_check(
            settings.sandbox_static_dir / "css" / "app-chrome.css",
            expect_file=True,
        ),
    }
    status = "ok" if all(check["ok"] for check in checks.values()) else "degraded"
    return {
        "status": status,
        "app": settings.app_name,
        "base_path": app_base_path(request) or settings.base_path,
        "checks": checks,
    }


@router.get("/favicon.ico", include_in_schema=False)
async def favicon(request: Request):
    return RedirectResponse(url=app_path(request, "/sandbox-static/assets/favicon.svg"))


def add_page_route(path: str, template_name: str) -> None:
    async def page(request: Request):
        return templates.TemplateResponse(request, template_name, {"request": request})

    page.__name__ = f"page_{path.strip('/').replace('/', '_') or 'index'}"
    router.add_api_route(path, page, methods=["GET"], response_class=HTMLResponse)


for route_path, route_template in PAGE_ROUTES.items():
    add_page_route(route_path, route_template)
