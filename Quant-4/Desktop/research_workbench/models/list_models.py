"""Stable-role Qt models fed exclusively by application-service DTOs."""
from __future__ import annotations

from typing import Any

from PySide6.QtCore import QAbstractListModel, QModelIndex, Qt


class DictListModel(QAbstractListModel):
    def __init__(self, roles: tuple[str, ...], parent: Any = None):
        super().__init__(parent)
        self._rows: list[dict[str, Any]] = []
        self._roles = {Qt.UserRole + index + 1: role.encode() for index, role in enumerate(roles)}

    def roleNames(self) -> dict[int, bytes]:  # noqa: N802 - Qt API
        return self._roles

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: B008,N802 - Qt API
        return 0 if parent.isValid() else len(self._rows)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:  # noqa: N802 - Qt API
        if not index.isValid() or not 0 <= index.row() < len(self._rows):
            return None
        name = self._roles.get(role)
        return self._rows[index.row()].get(name.decode()) if name else None

    def replace(self, rows: list[dict[str, Any]]) -> None:
        self.beginResetModel()
        self._rows = [dict(row) for row in rows]
        self.endResetModel()

    def append(self, row: dict[str, Any]) -> None:
        position = len(self._rows)
        self.beginInsertRows(QModelIndex(), position, position)
        self._rows.append(dict(row))
        self.endInsertRows()


class ResearchRunListModel(DictListModel):
    def __init__(self) -> None:
        super().__init__(("runId", "question", "status", "mode"))


class ResearchGraphModel(DictListModel):
    def __init__(self) -> None:
        super().__init__(("stageId", "name", "status", "x", "y", "critical"))


class EvidenceNexusModel(DictListModel):
    def __init__(self) -> None:
        super().__init__(("evidenceId", "claim", "origin", "status"))


class VerificationMatrixModel(DictListModel):
    def __init__(self) -> None:
        super().__init__(("level", "status", "evidenceCount"))


class EventStreamModel(DictListModel):
    def __init__(self) -> None:
        super().__init__(("sequence", "eventType", "runId", "summary", "occurredAt"))


class GovernanceGateModel(DictListModel):
    def __init__(self) -> None:
        super().__init__(("dimension", "status", "reason", "critical"))
