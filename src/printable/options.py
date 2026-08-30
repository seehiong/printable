"""Shared option parsing, usable from both the CLI and the web API."""

from __future__ import annotations


def parse_options(pairs: list[str]) -> dict:
    """Turn key=value strings into a dict, coercing obvious literals."""
    out: dict = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"bad option {pair!r}; expected key=value")
        key, _, raw = pair.partition("=")
        if raw.lower() in ("true", "false"):
            out[key] = raw.lower() == "true"
        else:
            try:
                out[key] = int(raw)
            except ValueError:
                try:
                    out[key] = float(raw)
                except ValueError:
                    out[key] = raw
    return out
