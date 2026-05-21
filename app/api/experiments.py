from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.services.stimuli import StimulusService


router = APIRouter(prefix="", tags=["experiments"])
stimuli = StimulusService(get_settings().static_dir)


SOUND_LIST_ROUTES = {
    "/get_sounds": "indefinite_pitch",
    "/get_sounds_training": "indefinite_pitch/training",
    "/get_sounds_shape": "image_shape",
    "/get_sounds_shape_training": "image_shape/training",
    "/get_sounds_texture": "image_texture",
    "/get_sounds_texture_training": "image_texture/training",
}


def add_sound_list_route(path: str, relative_dir: str) -> None:
    async def sound_list():
        return JSONResponse(content=stimuli.list_audio_files(relative_dir))

    sound_list.__name__ = f"list_{relative_dir.replace('/', '_')}"
    router.add_api_route(path, sound_list, methods=["GET"])


for route_path, route_dir in SOUND_LIST_ROUTES.items():
    add_sound_list_route(route_path, route_dir)
