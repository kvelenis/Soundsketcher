from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from app.core.templates import get_templates
from app.core.templates import app_path


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
