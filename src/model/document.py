from typing import Generic, List, TypeVar

from pydantic import BaseModel, Field

T = TypeVar('T', bound=BaseModel)

class Document(BaseModel, Generic[T]):
    id: str = Field(description='The unique identifier for the document')
    page_texts: List[str] = Field([], description='The OCR text for each page in the document')
    target: T = Field(description='The object containing the target information to extract from the document')

    def __reduce__(self):
        return (Document, (self.id, self.target, self.page_texts))