"""Undo / redo stack."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, List, Optional


class History:
    def __init__(self, limit: int = 80):
        self._undo: List[Any] = []
        self._redo: List[Any] = []
        self._limit = limit
        self._snapshot_fn: Optional[Callable[[], Any]] = None
        self._restore_fn: Optional[Callable[[Any], None]] = None

    def bind(self, snapshot: Callable[[], Any], restore: Callable[[Any], None]) -> None:
        self._snapshot_fn = snapshot
        self._restore_fn = restore

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()

    def push(self) -> None:
        if not self._snapshot_fn:
            return
        self._undo.append(self._snapshot_fn())
        if len(self._undo) > self._limit:
            self._undo.pop(0)
        self._redo.clear()

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> bool:
        if not self._undo or not self._snapshot_fn or not self._restore_fn:
            return False
        self._redo.append(self._snapshot_fn())
        state = self._undo.pop()
        self._restore_fn(deepcopy(state))
        return True

    def redo(self) -> bool:
        if not self._redo or not self._snapshot_fn or not self._restore_fn:
            return False
        self._undo.append(self._snapshot_fn())
        state = self._redo.pop()
        self._restore_fn(deepcopy(state))
        return True
