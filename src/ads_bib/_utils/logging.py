"""Logging helpers for consistent, low-noise runtime output."""

from __future__ import annotations

import logging
import os


def suppress_noisy_third_party_logs() -> None:
    """Suppress repetitive third-party transport logs while keeping pipeline logs."""
    try:
        for noisy_logger_name in ("httpx", "httpcore", "LiteLLM", "litellm"):
            logging.getLogger(noisy_logger_name).setLevel(logging.WARNING)

        os.environ.setdefault("LITELLM_LOG", "WARNING")
        os.environ.setdefault("LITELLM_VERBOSE", "False")

        try:
            import litellm
        except Exception:
            return

        set_verbose_attr = getattr(litellm, "set_verbose", None)
        if callable(set_verbose_attr):
            try:
                set_verbose_attr(False)
            except Exception:
                pass
        elif set_verbose_attr is not None:
            try:
                setattr(litellm, "set_verbose", False)
            except Exception:
                pass

        if hasattr(litellm, "suppress_debug_info"):
            try:
                litellm.suppress_debug_info = True
            except Exception:
                pass
    except Exception:
        return
