import json
from dataclasses import dataclass
from functools import reduce
from typing import Generic, List, Optional, Tuple, TypeVar, no_type_check

T = TypeVar("T", int, float)

@dataclass(frozen=True)
class BBox(Generic[T]):
    left: T
    top: T
    right: T
    bottom: T

    def to_absolute_coords(self, width: float, height: float) -> "BBox[int]":
        return BBox(
            round(self.left * width),
            round(self.top * height),
            round(self.right * width),
            round(self.bottom * height),
        )

    def to_relative_coords(self, width: float, height: float) -> "BBox[float]":
        return BBox(
            self.left / width,
            self.top / height,
            self.right / width,
            self.bottom / height,
        )

    def has_valid_relative_coords(self) -> bool:
        return 0 <= self.left <= self.right <= 1 and 0 <= self.top <= self.bottom <= 1

    def to_tuple(self) -> Tuple[T, T, T, T]:
        return self.left, self.top, self.right, self.bottom

    def intersects(self, other: "BBox") -> bool:
        if self.left > other.right or other.left > self.right:
            return False
        if self.top > other.bottom or other.top > self.bottom:
            return False
        return True

    def union(self, *others: "BBox[T]") -> "BBox[T]":
        if not others:
            return self
        return self.__class__(
            min(self, *others, key=lambda bbox: bbox.left).left,
            min(self, *others, key=lambda bbox: bbox.top).top,
            max(self, *others, key=lambda bbox: bbox.right).right,
            max(self, *others, key=lambda bbox: bbox.bottom).bottom,
        )

    def __and__(self, other: "BBox[T]") -> "BBox[T]":
        l1, t1, r1, b1 = self.to_tuple()
        l2, t2, r2, b2 = other.to_tuple()
        l, t, r, b = max(l1, l2), max(t1, t2), min(r1, r2), min(b1, b2)
        return self.__class__(l, t, r, b) if l < r and t < b else self.zero_bbox()

    def __or__(self, other: "BBox[T]") -> "BBox[T]":
        return self.union(other)

    @classmethod
    def zero_bbox(cls) -> "BBox[T]":
        return cls(0, 0, 0, 0)

    @property
    def width(self) -> T:
        return self.right - self.left

    @property
    def height(self) -> T:
        return self.bottom - self.top

    @property
    def size(self) -> Tuple[T, T]:
        return self.width, self.height

    @property
    def area(self) -> T:
        return self.width * self.height

    @property
    def centroid(self) -> Tuple[T, T]:
        ctr = ((self.left + self.right) / 2), ((self.top + self.bottom) / 2)
        return (round(ctr[0]), round(ctr[1])) if isinstance(self.left, int) else ctr

    @no_type_check
    def intersection(self, *others: "BBox[T]") -> "BBox[T]":
        return reduce(self.__class__.__and__, others, self)

@dataclass
class OcrResult:
    doc_id: str
    words: List[str]
    bboxes: List[BBox] # 1000x1000 normalized bouding boxes
    snapped_bboxes: Optional[List[BBox]] = None # 1000x1000 normalized bouding boxes

    # None if this is a GT
    confidences: Optional[List[float]] = None

@dataclass
class Field:
    doc_id: str
    type: str
    text: str
    bboxes: List[BBox] # 1000x1000 normalized bouding boxes
    snapped_bboxes: List[BBox]
    words: List[str]

    # Used if the field is a prediction
    p_ner: Optional[float] = None
    p_ocr: Optional[float] = None

    # eq if text is equal
    def __eq__(self, other):
        return self.text == other.text
    
    # hash by text
    def __hash__(self):
        return hash(self.text)

def load_ocr_result(filepath) -> OcrResult:
    with open(filepath, 'r') as f:
        js = json.loads(f.read())

    js['bboxes'] = [BBox(**bbox) for bbox in js['bboxes']]
    js['snapped_bboxes'] = [BBox(**bbox) for bbox in js['snapped_bboxes']]
    return OcrResult(**js)

def load_fields(filepath) -> List[Field]:
    with open(filepath, 'r') as f:
        js = json.loads(f.read())
    
    for f in js:
        f['bboxes'] = [BBox(**bbox) for bbox in f['bboxes']]
        f['snapped_bboxes'] = [BBox(**bbox) for bbox in f['snapped_bboxes']]
    return [Field(**f) for f in js]
