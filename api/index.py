import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from app.main import app as _app


class _VercelPathMiddleware:
    def __init__(self, inner):
        self.inner = inner

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http":
            p = scope.get("path", "")
            if p.startswith("/api/index.py"):
                p = p[len("/api/index.py"):]
                scope["path"] = p if p else "/"
            elif p == "/api":
                scope["path"] = "/"
            elif p.startswith("/api/"):
                p = p[len("/api"):]
                scope["path"] = p if p else "/"
        await self.inner(scope, receive, send)


app = _VercelPathMiddleware(_app)

__all__ = ["app"]
