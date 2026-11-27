import json

import pytest
from pydantic import BaseModel

from doctodeck.llm.gemini_schema import to_gemini_schema
from doctodeck.pipeline.models import CardsOutput, PlanOutput


def test_nested_models_are_inlined_and_types_capitalised():
    schema = to_gemini_schema(PlanOutput.model_json_schema())

    module = schema["properties"]["modules"]["items"]
    assert schema["type"] == "OBJECT"
    assert module["type"] == "OBJECT"
    assert module["properties"]["sectionIndices"] == {
        "type": "ARRAY",
        "items": {"type": "INTEGER"},
        "description": "0-based indices of the outline sections in this module.",
    }
    assert module["required"] == ["title", "description", "estimatedCards", "sectionIndices"]


def test_no_keyword_gemini_rejects_is_left_anywhere():
    text = json.dumps(to_gemini_schema(CardsOutput.model_json_schema()))

    for keyword in ("$ref", "$defs", "additionalProperties", '"object"', '"string"'):
        assert keyword not in text


def test_a_property_named_title_survives_while_the_title_keyword_is_dropped():
    schema = to_gemini_schema(PlanOutput.model_json_schema())

    module = schema["properties"]["modules"]["items"]
    assert "title" not in schema and "title" not in module  # Pydantic's "title": "PlanOutput" keyword
    assert module["properties"]["title"]["type"] == "STRING"  # the field called title


def test_the_input_schema_is_not_modified():
    original = PlanOutput.model_json_schema()
    snapshot = json.dumps(original)

    to_gemini_schema(original)

    assert json.dumps(original) == snapshot


def test_a_recursive_model_is_rejected_instead_of_looping_forever():
    class Node(BaseModel):
        name: str
        children: list["Node"]

    with pytest.raises(ValueError, match="Recursive"):
        to_gemini_schema(Node.model_json_schema())
