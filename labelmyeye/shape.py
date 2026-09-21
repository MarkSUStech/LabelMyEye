"""Polygon shape model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

Point = Tuple[float, float]


@dataclass
class Shape:
    label: str = ""
    points: List[Point] = field(default_factory=list)
    shape_type: str = "polygon"
    group_id: Optional[int] = None
    description: str = ""
    flags: dict = field(default_factory=dict)
    # which image layer this shape belongs to: "after" | "before" | "single"
    layer: str = "after"
    selected: bool = False

    def copy(self) -> "Shape":
        return Shape(
            label=self.label,
            points=[(float(x), float(y)) for x, y in self.points],
            shape_type=self.shape_type,
            group_id=self.group_id,
            description=self.description,
            flags=dict(self.flags),
            layer=self.layer,
            selected=False,
        )

    def to_labelme(self) -> dict:
        return {
            "label": self.label,
            "points": [[float(x), float(y)] for x, y in self.points],
            "group_id": self.group_id,
            "description": self.description,
            "shape_type": self.shape_type,
            "flags": self.flags,
            "mask": None,
        }

    @classmethod
    def from_labelme(cls, data: dict, layer: str = "after") -> "Shape":
        return cls(
            label=data.get("label", ""),
            points=[(float(p[0]), float(p[1])) for p in data.get("points", [])],
            shape_type=data.get("shape_type", "polygon"),
            group_id=data.get("group_id"),
            description=data.get("description", "") or "",
            flags=data.get("flags") or {},
            layer=layer,
        )

    def close_enough(self, p: Point, q: Point, thresh: float) -> bool:
        return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 <= thresh * thresh
