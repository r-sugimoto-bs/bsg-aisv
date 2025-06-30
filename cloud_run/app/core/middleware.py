import os
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # preflight request の場合は、BEARER_TOKEN のチェックを行わない
        if request.method == 'OPTIONS':
            response = await call_next(request)
            response.status_code = 200
            return response

        auth_header = request.headers.get('Authorization')
        expected_auth_header = os.environ.get('BEARER_TOKEN')
        if auth_header != f"Bearer {expected_auth_header}":
            raise HTTPException(status_code=403, detail="Forbidden")

        response = await call_next(request)
        return response