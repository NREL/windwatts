import time
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import logging
import uuid

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        request_id = str(uuid.uuid4())

        # Log request start
        logger.info(
            f"Request started: {request.method} {request.url.path} | "
            f"ID: {request_id} | "
            f"Client: {request.client.host if request.client else 'unknown'}"
        )

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

            # Log request completion
            logger.info(
                f"Request completed: {request.method} {request.url.path} | "
                f"Status: {response.status_code} | "
                f"Duration: {duration_ms}ms | "
                f"Size: {response_size} bytes"
            )

            # Reconstruct response
            return Response(
                content=response_body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type,
            )

        except Exception as e:
            # Log error
            logger.error(
                f"Request failed: {request.method} {request.url.path} | "
                f"Error: {str(e)} | "
                f"Duration: {int((time.time() - start_time) * 1000)}ms",
                exc_info=True,
            )
            raise
