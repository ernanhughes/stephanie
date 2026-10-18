"""SIS template adapter for the Starlette request-first template API.

The existing route modules use the older ``TemplateResponse(name, context)``
call shape. Keep those call sites working while translating at the application
boundary to the current ``TemplateResponse(request, name, context)`` API.
"""

from __future__ import annotations

from fastapi.templating import Jinja2Templates
from starlette.requests import Request


class SISTemplates(Jinja2Templates):
    def TemplateResponse(self, *args, **kwargs):
        if args and isinstance(args[0], str):
            if len(args) < 2 or not isinstance(args[1], dict):
                raise TypeError("Legacy SIS TemplateResponse requires (name, context)")
            name, context, *remaining = args
            request = context.get("request")
            if not isinstance(request, Request):
                raise TypeError("SIS template context must include a Starlette Request")
            args = (request, name, context, *remaining)
        return super().TemplateResponse(*args, **kwargs)
