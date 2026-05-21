from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.core.config import get_settings
from app.services.stimuli import StimulusService
from app.storage.responses import ResponseStore


router = APIRouter(prefix="", tags=["questionnaires"])
settings = get_settings()
store = ResponseStore(settings.response_root)
stimuli = StimulusService(settings.static_dir)


class ShapeResponseData(BaseModel):
    session_id: str
    sound: str
    response: dict[str, Any]


class TextureResponseData(BaseModel):
    session_id: str
    sound: str
    response: dict[str, Any]


class ResponseData(BaseModel):
    session_id: str
    sound: str
    response: dict[str, Any]


class UserInfo(BaseModel):
    session_id: str
    gender: str
    age: int
    music_experience: str
    plays_instrument: bool
    instrument_name: str = ""
    knows_pitch: bool
    hearing_condition: str
    absolute_pitch: bool


class NoiseTonalPreferenceUserInfo(BaseModel):
    session_id: str
    gender: str
    age: int
    music_education_years: int
    plays_instrument: bool
    instrument_name: str = ""
    consent_given: bool
    consent_version: str = ""


class NoiseTonalPreferenceResponse(BaseModel):
    session_id: str
    order_index: int
    total_sounds: int
    stimulus_id: str
    audio_filenames: list[str]
    slider_value: float
    timestamp: str
    mode: str


class SupplementaryUserInfo(BaseModel):
    session_id: str
    gender: str
    age: int
    music_education_years: int
    plays_instrument: bool
    instrument_name: str = ""
    consent_given: bool
    consent_version: str = ""


class SupplementaryResponseValues(BaseModel):
    smoothness: int
    roughness: int
    sharpness: int
    timestamp: str


class SupplementaryResponseData(BaseModel):
    session_id: str
    sound: str
    order_index: int
    response: SupplementaryResponseValues


def dump_model(model: BaseModel) -> dict[str, Any]:
    return model.model_dump()


@router.post("/save_shape_response")
async def save_shape_response(data: ShapeResponseData):
    store.append_session_json("responses_shape", data.session_id, dump_model(data))
    return {"status": "ok"}


@router.post("/save_texture_response")
async def save_texture_response(data: TextureResponseData):
    store.append_session_json("responses_texture", data.session_id, dump_model(data))
    return {"status": "ok"}


@router.post("/save_responses_indefinite_pitch")
async def save_responses_indefinite_pitch(data: ResponseData):
    store.append_session_json("responses_indefinite_pitch", data.session_id, dump_model(data))
    return {"status": "ok"}


@router.post("/save_indefinite_pitch_user_info")
async def save_indefinite_pitch_user_info(info: UserInfo):
    store.write_session_json("responses_indefinite_pitch/user_info", info.session_id, dump_model(info))
    return {"status": "ok", "message": "User info saved successfully."}


@router.post("/save_image_shape_user_info")
async def save_image_shape_user_info(info: UserInfo):
    store.write_session_json("responses_shape/user_info", info.session_id, dump_model(info))
    return {"status": "ok", "message": "User info saved successfully."}


@router.post("/save_image_texture_user_info")
async def save_image_texture_user_info(info: UserInfo):
    store.write_session_json("responses_texture/user_info", info.session_id, dump_model(info))
    return {"status": "ok", "message": "User info saved successfully."}


@router.get("/get_all_responses_indefinite_pitch")
async def get_all_responses_indefinite_pitch():
    return JSONResponse(content=store.read_all_json_entries("responses_indefinite_pitch"))


@router.get("/get_all_responses_shape")
async def get_all_responses_shape():
    return JSONResponse(content=store.read_all_json_entries("responses_shape"))


@router.get("/get_all_responses_texture")
async def get_all_responses_texture():
    return JSONResponse(content=store.read_all_json_entries("responses_texture"))


@router.get("/questionnaires/noise-tonal-preference/v1/stimuli")
async def get_noise_tonal_preference_stimuli():
    sample_dir = settings.static_dir / "noise_tonal_preference_samples"
    if not sample_dir.exists():
        return JSONResponse(
            status_code=404,
            content={"error": f"Sample directory not found: {sample_dir}"},
        )

    noise_tonal_stimuli = stimuli.noise_tonal_preference_stimuli()
    return JSONResponse(
        content={
            "total_sounds": len(noise_tonal_stimuli),
            "stimuli": noise_tonal_stimuli,
        }
    )


@router.post("/questionnaires/noise-tonal-preference/v1/save-user-info")
async def save_noise_tonal_preference_user_info(payload: NoiseTonalPreferenceUserInfo):
    store.append_jsonl(
        "questionnaire_results/noise_tonal_preference/user_info.jsonl",
        dump_model(payload),
    )
    return {"status": "ok"}


@router.post("/questionnaires/noise-tonal-preference/v1/save-response")
async def save_noise_tonal_preference_response(payload: NoiseTonalPreferenceResponse):
    store.append_jsonl(
        "questionnaire_results/noise_tonal_preference/responses.jsonl",
        dump_model(payload),
    )
    return {"status": "ok"}


@router.get("/get_sounds_supplementary")
async def get_sounds_supplementary():
    return JSONResponse(content=stimuli.list_audio_files("image_shape"))


@router.post("/save_supplementary_user_info")
async def save_supplementary_user_info(data: SupplementaryUserInfo):
    store.write_session_json("responses_supplementary_users", data.session_id, dump_model(data))
    return {"status": "ok"}


@router.post("/save_supplementary_response")
async def save_supplementary_response(data: SupplementaryResponseData):
    store.append_session_json("responses_supplementary", data.session_id, dump_model(data))
    return {"status": "ok"}
