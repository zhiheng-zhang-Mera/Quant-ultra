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

    def update_where(self, role_name: str, value: Any, changes: dict[str, Any]) -> bool:
        for position, row in enumerate(self._rows):
            if row.get(role_name) == value:
                row.update(changes)
                index = self.index(position, 0)
                self.dataChanged.emit(index, index, list(self._roles))
                return True
        return False

    def rows(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._rows]


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

    def append(self, row: dict[str, Any]) -> None:
        super().append(row)
        overflow = len(self._rows) - 1000
        if overflow > 0:
            self.beginRemoveRows(QModelIndex(), 0, overflow - 1)
            del self._rows[:overflow]
            self.endRemoveRows()


class GovernanceGateModel(DictListModel):
    def __init__(self) -> None:
        super().__init__(("dimension", "status", "reason", "critical"))


class AgentActivityModel(DictListModel):
    def __init__(self) -> None: super().__init__(("agentId", "role", "status", "activity"))


class ExperimentModel(DictListModel):
    def __init__(self) -> None: super().__init__(("experimentId", "status", "specHash", "trialCount"))


class TwinImplementationModel(DictListModel):
    def __init__(self) -> None: super().__init__(("level", "status", "agreement", "diagnostic"))


class StatisticalModel(DictListModel):
    def __init__(self) -> None: super().__init__(("metric", "value", "status", "evidenceId"))


class RobustnessModel(DictListModel):
    def __init__(self) -> None: super().__init__(("attack", "status", "detail", "evidenceId"))


class GeneralizationModel(DictListModel):
    def __init__(self) -> None: super().__init__(("axis", "status", "parameterHash", "evidenceId"))


class KernelPhaseModel(DictListModel):
    def __init__(self) -> None: super().__init__(("phase", "status", "duration", "artifact"))


class MemoryModel(DictListModel):
    def __init__(self) -> None: super().__init__(("memoryId", "type", "result", "failureReason"))


class ReportModel(DictListModel):
    def __init__(self) -> None: super().__init__(("path", "sha256", "status", "size"))
