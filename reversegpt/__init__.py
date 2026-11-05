"""ReverseGPT: turn natural-language requests into chained API calls.

A RestGPT-style planner breaks the request into steps, and a Reverse Chain
resolver works backwards from the API that finishes each step to the APIs
that produce its missing arguments.
"""

from reversegpt.config import Settings, load_settings
from reversegpt.pipeline import ReverseGPT, RunResult, ToolCall
from reversegpt.registry import ApiRegistry, ApiSpec

__all__ = [
    "ApiRegistry",
    "ApiSpec",
    "ReverseGPT",
    "RunResult",
    "Settings",
    "ToolCall",
    "load_settings",
]

__version__ = "1.0.0"
