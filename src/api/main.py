"""
FastAPI application entrypoint with middleware, lifecycle, and error handling.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..config import Environment, get_settings
from .dependencies import lifespan_dependencies
from .routes import health_router, posts_router, settings_router
from .schemas import ErrorDetail, ErrorResponse

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------

LOGGER = logging.getLogger(__name__)


def configure_logging(level: str = "INFO") -> None:
    """Configure root logger with structured format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )


# -----------------------------------------------------------------------------
# Application Lifespan
# -----------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan context manager.

    Handles startup and shutdown of shared resources.
    """
    settings = get_settings()
    configure_logging(settings.logging.level)

    LOGGER.info(
        "Starting %s v%s in %s environment",
        settings.app.name,
        settings.app.version,
        settings.app.environment.value,
    )

    async with lifespan_dependencies(settings):
        yield

    LOGGER.info("Application shutdown complete.")


# -----------------------------------------------------------------------------
# Application Factory
# -----------------------------------------------------------------------------


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns a fully configured FastAPI instance with all routes and middleware.
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.app.name,
        version=settings.app.version,
        description="Personalized news aggregation service with ML classification",
        docs_url="/docs" if settings.app.debug else None,
        redoc_url="/redoc" if settings.app.debug else None,
        openapi_url="/openapi.json" if settings.app.debug else None,
        lifespan=lifespan,
    )

    # -------------------------------------------------------------------------
    # CORS Middleware
    # -------------------------------------------------------------------------

    allowed_origins = ["*"] if settings.app.environment == Environment.LOCAL else []
    if settings.app.environment in (Environment.DEVELOPMENT, Environment.STAGING):
        allowed_origins = [
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
        ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    # -------------------------------------------------------------------------
    # Request Logging Middleware
    # -------------------------------------------------------------------------

    @app.middleware("http")
    async def request_logging_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Log request details and add request ID header."""
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        start_time = time.perf_counter()

        # Attach request_id to request state for access in handlers
        request.state.request_id = request_id

        try:
            response = await call_next(request)
        except Exception:
            LOGGER.exception(
                "Unhandled exception for %s %s [%s]",
                request.method,
                request.url.path,
                request_id,
            )
            raise

        duration_ms = (time.perf_counter() - start_time) * 1000

        LOGGER.info(
            "%s %s -> %d (%.2fms) [%s]",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            request_id,
        )

        response.headers["X-Request-ID"] = request_id
        return response

    # -------------------------------------------------------------------------
    # Exception Handlers
    # -------------------------------------------------------------------------

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request,
        exc: StarletteHTTPException,
    ) -> JSONResponse:
        """Handle HTTP exceptions with consistent error format."""
        request_id = getattr(request.state, "request_id", None)

        error_response = ErrorResponse(
            error=exc.__class__.__name__,
            message=str(exc.detail),
            details=[],
            request_id=request_id,
        )

        return JSONResponse(
            status_code=exc.status_code,
            content=error_response.model_dump(exclude_none=True),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        """Handle validation errors with detailed field information."""
        request_id = getattr(request.state, "request_id", None)

        details = [
            ErrorDetail(
                field=".".join(str(loc) for loc in error.get("loc", [])),
                message=error.get("msg", "Validation error"),
                code=error.get("type"),
            )
            for error in exc.errors()
        ]

        error_response = ErrorResponse(
            error="ValidationError",
            message="Request validation failed",
            details=details,
            request_id=request_id,
        )

        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=error_response.model_dump(exclude_none=True),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        """Handle unexpected exceptions."""
        request_id = getattr(request.state, "request_id", None)

        LOGGER.exception(
            "Unhandled exception: %s [%s]",
            str(exc),
            request_id,
        )

        # Hide internal errors in production
        message = str(exc) if settings.app.debug else "An internal error occurred"

        error_response = ErrorResponse(
            error="InternalServerError",
            message=message,
            details=[],
            request_id=request_id,
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response.model_dump(exclude_none=True),
        )

    # -------------------------------------------------------------------------
    # Route Registration
    # -------------------------------------------------------------------------

    app.include_router(health_router)
    app.include_router(posts_router)
    app.include_router(settings_router)

    return app


# -----------------------------------------------------------------------------
# Application Instance
# -----------------------------------------------------------------------------

app = create_app()


__all__ = ["app", "create_app"]
