from ai_gateway.api.routers.admin import router as admin_router
from ai_gateway.api.routers.assistant import router as assistant_router
from ai_gateway.api.routers.health import router as health_router

__all__ = ["admin_router", "assistant_router", "health_router"]