from starlette.types import ASGIApp, Receive, Scope, Send


class BasePathMiddleware:
    """Serve the app both at / and at a configured URL prefix.

    The reverse proxy can pass /soundsketcher/... through unchanged.  This
    middleware strips the public prefix for routing, while setting root_path so
    URL generation can include the prefix again.
    """

    def __init__(self, app: ASGIApp, base_path: str) -> None:
        self.app = app
        self.base_path = "/" + base_path.strip("/") if base_path else ""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not self.base_path or scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path == self.base_path:
            scope = dict(scope)
            scope["root_path"] = self.base_path
            scope["path"] = "/"
        elif path.startswith(f"{self.base_path}/"):
            scope = dict(scope)
            scope["root_path"] = self.base_path
            scope["path"] = path[len(self.base_path) :] or "/"

        await self.app(scope, receive, send)
