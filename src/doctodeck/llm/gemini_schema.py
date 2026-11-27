"""Translate a Pydantic JSON Schema into the dialect Gemini's `responseSchema` accepts.

Gemini's schema is an OpenAPI 3.0 subset: type names in capitals ("OBJECT", "STRING"), no `$ref`/`$defs`,
no `additionalProperties`. Pydantic's `model_json_schema()` puts every nested model under `$defs` and points
at it with `$ref`, so the translator inlines those references. Mirrors AnkiX's `GeminiSchemas.cs`.
"""

from typing import Any

# Keywords Gemini's responseSchema doesn't take. "title" is the schema keyword Pydantic adds to every
# model and field, not a property *named* title: property names are handled separately below.
_DROPPED = frozenset({"$schema", "$defs", "additionalProperties", "title", "default"})


def to_gemini_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a new schema in Gemini's dialect. The input is never modified."""
    return _convert(schema, schema.get("$defs", {}), in_progress=())


def _convert(node: dict[str, Any], defs: dict[str, Any], in_progress: tuple[str, ...]) -> dict[str, Any]:
    if "$ref" in node:
        name = node["$ref"].rsplit("/", 1)[-1]  # "#/$defs/PlanOutputModule" -> "PlanOutputModule"
        if name in in_progress:
            # A model that contains itself (a tree). Inlining would never end; Gemini's dialect can't express it.
            raise ValueError(f"Recursive schema '{name}' can't be inlined for Gemini.")
        siblings = {k: v for k, v in node.items() if k != "$ref"}  # e.g. a description next to the $ref
        return _convert(defs[name] | siblings, defs, (*in_progress, name))

    converted: dict[str, Any] = {}
    for key, value in node.items():
        if key in _DROPPED:
            continue
        if key == "type":
            converted["type"] = value.upper()  # "object" -> "OBJECT"
        elif key == "properties":
            # Keys here are property names (which may well be "title"), values are schemas.
            converted["properties"] = {name: _convert(sub, defs, in_progress) for name, sub in value.items()}
        elif key == "items":
            converted["items"] = _convert(value, defs, in_progress)
        elif key in ("anyOf", "oneOf", "allOf"):
            converted[key] = [_convert(sub, defs, in_progress) for sub in value]
        else:
            converted[key] = value  # description, required, enum, format...
    return converted
