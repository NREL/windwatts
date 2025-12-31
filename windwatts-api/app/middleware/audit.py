from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import json
from typing import Dict, Any
import logging
import uuid
import time

logger = logging.getLogger(__name__)


class AuditMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        request_id = str(uuid.uuid4())

        # Get request details
        request_data = await self._get_request_data(request)
        request_size = len(json.dumps(request_data).encode("utf-8"))

        try:
            # Process the request
            response = await call_next(request)

            # Calculate duration
            duration_ms = int((time.time() - start_time) * 1000)

            # Get response size
            response_body = b""
            async for chunk in response.body_iterator:
                response_body += chunk
            response_size = len(response_body)

            # Only log successful requests to audit log
            if response.status_code < 400:
                self._log_request(
                    request=request,
                    response=response,
                    duration_ms=duration_ms,
                    request_size=request_size,
                    response_size=response_size,
                    request_id=request_id,
                )

            # Reconstruct response
            return Response(
                content=response_body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type,
            )

        except Exception:
            raise

    async def _get_request_data(self, request: Request) -> Dict[str, Any]:
        """Extract relevant data from the request"""
        try:
            body = await request.body()
            if body:
                try:
                    return json.loads(body)
                except json.JSONDecodeError:
                    return {"raw_body": body.decode()}
        except Exception:
            return {}
        return {}

    def _get_client_type(self, user_agent: str) -> str:
        """Determine client type from user agent"""
        if not user_agent:
            return "unknown"
        user_agent = user_agent.lower()
        if "mobile" in user_agent or "android" in user_agent or "iphone" in user_agent:
            return "mobile"
        if "postman" in user_agent or "curl" in user_agent:
            return "api"
        return "web"

    def _log_request(
        self,
        request: Request,
        response,
        duration_ms: int,
        request_size: int,
        response_size: int,
        request_id: str,
    ):
        """Create audit log entry for successful requests only"""
        print("Logging request")
        print("Request ID", request_id)
        print("Duration", duration_ms)
        print("Request size", request_size)
        print("Response size", response_size)
        print("Request", request)
        print("Response", response)
        # db: Session = SessionLocal()
        # try:
        #     # Get user ID from request if available
        #     user_id = None
        #     if hasattr(request.state, "user"):
        #         user_id = str(request.state.user.id)

        #     # Extract path information
        #     path = str(request.url.path)

        #     # Create audit log entry
        #     audit_log = AuditLog(
        #         user_id=user_id,
        #         action="api_request",
        #         resource=path,
        #         method=request.method,
        #         status_code=response.status_code,
        #         ip_address=request.client.host if request.client else None,
        #         user_agent=request.headers.get("user-agent"),
        #         duration_ms=duration_ms,
        #         request_size_bytes=request_size,
        #         response_size_bytes=response_size,
        #         request_id=request_id,
        #         log_metadata={
        #             "query_params": dict(request.query_params),
        #             "path_params": request.path_params
        #         }
        #     )

        #     db.add(audit_log)
        #     db.commit()
        # except Exception as e:
        #     logger.error(f"Failed to create audit log: {str(e)}")
        #     db.rollback()
        # finally:
        #     db.close()
