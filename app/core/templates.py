from fastapi.templating import Jinja2Templates
from jinja2 import ChoiceLoader, FileSystemLoader

from app.core.config import get_settings


def app_base_path(request) -> str:
    return request.scope.get("root_path", "").rstrip("/")


def app_path(request, path: str) -> str:
    base_path = app_base_path(request)
    normalized_path = path if path.startswith("/") else f"/{path}"
    if not base_path:
        return normalized_path
    if normalized_path == "/":
        return f"{base_path}/"
    return f"{base_path}{normalized_path}"


def sandbox_static_path(request, path: str) -> str:
    return app_path(request, f"/sandbox-static/{path.lstrip('/')}")


def get_templates() -> Jinja2Templates:
    settings = get_settings()
    templates = Jinja2Templates(directory=str(settings.templates_dir))
    sandbox_templates_dir = settings.project_root / "refactor_sandbox" / "templates"
    templates.env.loader = ChoiceLoader(
        [
            FileSystemLoader(str(sandbox_templates_dir)),
            FileSystemLoader(str(settings.templates_dir)),
        ]
    )
    templates.env.globals["app_path"] = app_path
    templates.env.globals["app_base_path"] = app_base_path
    templates.env.globals["sandbox_static_path"] = sandbox_static_path
    return templates
