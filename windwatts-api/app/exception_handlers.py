from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import traceback
from app.logging_setup import logger


async def log_unhandled_exceptions(request: Request, exc: Exception):
    logger.error(f"❌ ERROR: {request.method} {request.url}\n{traceback.format_exc()}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


async def log_validation_errors(request: Request, exc: RequestValidationError):
    logger.warning(
        f"⚠️ VALIDATION ERROR: {request.method} {request.url} | {exc.errors()}"
    )
    # Remove 'input' field from all error details
    filtered_errors = []
    for err in exc.errors():
        filtered = {k: v for k, v in err.items() if k != "input"}
        filtered_errors.append(filtered)
    return JSONResponse(status_code=422, content={"detail": filtered_errors})
