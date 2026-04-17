"""Public HTTP mountable routers for observation and interaction.

``make_read_router`` returns a FastAPI ``APIRouter`` the developer mounts
into their own app alongside their own auth dependency. Dendrux never
owns the server process.
"""

from __future__ import annotations

from dendrux.http.read_router import make_read_router

__all__ = ["make_read_router"]
