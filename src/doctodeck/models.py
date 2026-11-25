"""Data shapes shared by the parser and the API.

Pydantic models instead of plain dataclasses because they are also the API's JSON contract:
`alias_generator=to_camel` makes `word_count` go out as `wordCount`, which is what the page's
JavaScript reads. Python code keeps snake_case; the wire keeps camelCase.
"""

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class ApiModel(BaseModel):
    # frozen=True: a parsed document is a value, nobody should mutate it after the fact.
    # populate_by_name=True: tests and code can still build models with snake_case names.
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, frozen=True)


class DocumentSection(ApiModel):
    title: str
    level: int
    word_count: int
    snippet: str


class ParsedDocument(ApiModel):
    sections: list[DocumentSection]
    clean_text: str
    word_count: int
    page_count: int
