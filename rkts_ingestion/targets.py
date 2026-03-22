"""Default pilot targets for the RKTS ingestion pass."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PilotTarget:
    source_family: str
    collection_code: str
    volume_number: str


PILOT_TARGETS: tuple[PilotTarget, ...] = (
    PilotTarget("Kanjur", "KanjurDerge", "001"),
    PilotTarget("Kanjur", "KanjurDerge", "045"),
    PilotTarget("Tanjur", "TanjurDerge", "001"),
    PilotTarget("Tanjur", "TanjurDerge", "173"),
    PilotTarget("Old Tantra", "NGBGdg", "001"),
    PilotTarget("Old Tantra", "NGBGdg", "013"),
    PilotTarget("Bon Canon", "BonBkz", "001"),
    PilotTarget("Bon Canon", "BonBkz", "050"),
)
