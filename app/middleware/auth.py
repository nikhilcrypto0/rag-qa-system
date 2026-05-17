import logging
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from app.config import get_settings

logger = logging.getLogger(__name__)

EXEMPT_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        settings = get_settings()

        if settings.auth_mode == "none" or request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        if settings.auth_mode == "api_key":
            key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
            valid_keys = {k.strip() for k in settings.api_keys.split(",") if k.strip()}
            if not key or key not in valid_keys:
                raise HTTPException(status_code=401, detail="Invalid or missing API key.")
            return await call_next(request)

        if settings.auth_mode == "email_domain":
            email = request.headers.get("X-User-Email", "").strip().lower()
            domain = settings.allowed_email_domain.strip().lower()
            if not email or not email.endswith(f"@{domain}"):
                raise HTTPException(
                    status_code=403,
                    detail=f"Access restricted to @{domain} university accounts."
                )
            return await call_next(request)

        return await call_next(request)
