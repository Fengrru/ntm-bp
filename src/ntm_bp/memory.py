"""MemoryState and fact data models for NTM-BP.

Merged lightweight port of the research codebase's
``ntm.core.types``, ``ntm.core.fact`` and ``ntm.core.memory_state``.
The copy-on-write storage hook and propagation-mode enum are dropped:
this package ships only the state needed by the propagation engine.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from ntm_bp.graph import DependencyGraph


class FactStatus(Enum):
    """Status of a fact in the memory system."""

    ACTIVE = "ACTIVE"
    """Fact is active and its belief is trusted."""

    STALE = "STALE"
    """Fact invalidated by a cascade (binary system). Must be re-derived."""

    REEVAL_REQUIRED = "REEVAL_REQUIRED"
    """Belief dropped below threshold; re-evaluation recommended."""

    PENDING = "PENDING"
    """Fact created but not yet evaluated."""


@dataclass(frozen=True)
class FactVersion:
    """An immutable version of a fact's value at a point in time.

    Attributes:
        value: The factual value (e.g., "Acme Corp 信誉良好").
        confidence: Confidence in [0.0, 1.0] assigned to this value.
        timestamp: Global time at which this version was created.
    """

    value: str
    confidence: float
    timestamp: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Confidence must be in [0.0, 1.0], got {self.confidence}"
            )
        if self.timestamp < 0:
            raise ValueError(f"Timestamp must be non-negative, got {self.timestamp}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "value": self.value,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FactVersion":
        """Deserialize from dictionary."""
        return cls(
            value=data["value"],
            confidence=data["confidence"],
            timestamp=data["timestamp"],
        )


@dataclass
class Fact:
    """A single fact in the memory system with version history.

    Each fact carries a monotonic version number that increments on
    every update via set_version(), enabling precise tracking of which
    version was current during propagation.

    Attributes:
        fact_id: Unique identifier for this fact.
        current_version: The most recent version of this fact.
        version: Monotonically increasing version number (>= 1).
        status: Current status (ACTIVE, STALE, REEVAL_REQUIRED, ...).
        metadata: Optional key-value metadata.
    """

    fact_id: str
    current_version: FactVersion
    version: int = 1
    status: FactStatus = FactStatus.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Archived (old) versions; current_version is NOT stored here.
    _versions: List[FactVersion] = field(default_factory=list, repr=False)

    @property
    def value(self) -> str:
        """Shorthand for current_version.value."""
        return self.current_version.value

    @property
    def confidence(self) -> float:
        """Shorthand for current_version.confidence."""
        return self.current_version.confidence

    @property
    def timestamp(self) -> int:
        """Shorthand for current_version.timestamp."""
        return self.current_version.timestamp

    @property
    def version_history(self) -> List[FactVersion]:
        """All versions (archived + current) as an immutable view."""
        return list(self._versions) + [self.current_version]

    def set_version(self, version: FactVersion) -> None:
        """Archive the old current version, set a new one, bump the counter."""
        self._versions.append(copy.deepcopy(self.current_version))
        self.current_version = version
        self.version += 1

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "fact_id": self.fact_id,
            "current_version": self.current_version.to_dict(),
            "version": self.version,
            "status": self.status.value,
            "metadata": copy.deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Fact":
        """Deserialize from dictionary."""
        return cls(
            fact_id=data["fact_id"],
            current_version=FactVersion.from_dict(data["current_version"]),
            version=data.get("version", 1),
            status=FactStatus(data["status"]),
            metadata=data.get("metadata", {}),
        )

    def __hash__(self) -> int:
        return hash(self.fact_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Fact):
            return NotImplemented
        return self.fact_id == other.fact_id


@dataclass
class MemoryState:
    """The central state of the NTM memory system.

    Attributes:
        facts: Dictionary mapping fact_id -> Fact.
        belief: Dictionary mapping fact_id -> current belief [0, 1].
        time: Global time counter, incremented on each update.
        tau: Re-evaluation threshold. When belief < tau, the fact is
            marked REEVAL_REQUIRED by the engine.
        graph: The dependency graph (required by the engine).
    """

    facts: Dict[str, Fact] = field(default_factory=dict)
    belief: Dict[str, float] = field(default_factory=dict)
    time: int = 0
    tau: float = 0.7
    graph: Optional[DependencyGraph] = None

    # ------------------------------------------------------------------
    # Fact management
    # ------------------------------------------------------------------

    def add_fact(
        self,
        fact_id: str,
        value: str,
        confidence: float,
        status: FactStatus = FactStatus.ACTIVE,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Fact:
        """Add a new fact to the memory system.

        Raises:
            ValueError: If fact_id already exists.
        """
        if fact_id in self.facts:
            raise ValueError(f"Fact '{fact_id}' already exists in memory.")

        version = FactVersion(value=value, confidence=confidence, timestamp=self.time)
        fact = Fact(
            fact_id=fact_id,
            current_version=version,
            status=status,
            metadata=metadata or {},
        )
        self.facts[fact_id] = fact
        self.belief[fact_id] = confidence
        return fact

    def get_fact(self, fact_id: str) -> Optional[Fact]:
        """Retrieve a fact by ID, or None if not found."""
        return self.facts.get(fact_id)

    def has_fact(self, fact_id: str) -> bool:
        """Check if a fact exists in memory."""
        return fact_id in self.facts

    # ------------------------------------------------------------------
    # Belief management
    # ------------------------------------------------------------------

    def get_belief(self, fact_id: str) -> Optional[float]:
        """Get the current belief (confidence) of a fact, or None."""
        return self.belief.get(fact_id)

    def set_belief(self, fact_id: str, confidence: float) -> None:
        """Set the belief for a fact (used by propagation algorithms).

        Raises:
            KeyError: If fact_id does not exist.
            ValueError: If confidence is outside [0, 1].
        """
        if fact_id not in self.facts:
            raise KeyError(f"Fact '{fact_id}' not found in memory.")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0.0, 1.0], got {confidence}")
        self.belief[fact_id] = confidence

    # ------------------------------------------------------------------
    # Status management
    # ------------------------------------------------------------------

    def mark_stale(self, fact_id: str) -> None:
        """Mark a fact as STALE and set its belief to 0.0 (binary system)."""
        if fact_id in self.facts:
            self.facts[fact_id].status = FactStatus.STALE
            self.belief[fact_id] = 0.0

    def mark_reeval(self, fact_id: str) -> None:
        """Mark a fact as REEVAL_REQUIRED (probabilistic system).

        The belief remains unchanged; the system should re-evaluate the
        fact before relying on its value.
        """
        if fact_id in self.facts:
            self.facts[fact_id].status = FactStatus.REEVAL_REQUIRED

    def mark_active(self, fact_id: str) -> None:
        """Reset a fact's status to ACTIVE."""
        if fact_id in self.facts:
            self.facts[fact_id].status = FactStatus.ACTIVE

    def get_stale_facts(self) -> List[str]:
        """Return IDs of all STALE facts."""
        return [
            fid for fid, fact in self.facts.items()
            if fact.status == FactStatus.STALE
        ]

    def get_reeval_facts(self) -> List[str]:
        """Return IDs of all REEVAL_REQUIRED facts."""
        return [
            fid for fid, fact in self.facts.items()
            if fact.status == FactStatus.REEVAL_REQUIRED
        ]

    # ------------------------------------------------------------------
    # Update operation
    # ------------------------------------------------------------------

    def update_fact(self, fact_id: str, new_value: str, new_confidence: float) -> None:
        """Perform a memory update: new version + belief update.

        Call ``NtmBpEngine.propagate()`` afterwards to cascade the change
        through the dependency graph.

        Raises:
            KeyError: If fact_id does not exist.
        """
        if fact_id not in self.facts:
            raise KeyError(f"Fact '{fact_id}' not found in memory.")

        fact = self.facts[fact_id]
        new_version = FactVersion(
            value=new_value,
            confidence=new_confidence,
            timestamp=self.time,
        )
        fact.set_version(new_version)
        self.belief[fact_id] = new_confidence
        self.time += 1

    # ------------------------------------------------------------------
    # Snapshot & serialization
    # ------------------------------------------------------------------

    def to_snapshot(self) -> Dict[str, Any]:
        """Create a serializable snapshot of the current memory state."""
        return {
            "time": self.time,
            "tau": self.tau,
            "facts": {fid: fact.to_dict() for fid, fact in self.facts.items()},
            "belief": dict(self.belief),
            "graph": self.graph.to_dict() if self.graph else None,
        }

    @classmethod
    def from_snapshot(cls, data: Dict[str, Any]) -> "MemoryState":
        """Restore a MemoryState from a snapshot dictionary."""
        state = cls(time=data["time"], tau=data["tau"])
        for fid, fact_data in data["facts"].items():
            state.facts[fid] = Fact.from_dict(fact_data)
        state.belief.update(data["belief"])
        if data.get("graph") is not None:
            state.graph = DependencyGraph.from_dict(data["graph"])
        return state

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def fact_count(self) -> int:
        """Number of facts currently in memory."""
        return len(self.facts)

    @property
    def active_fact_count(self) -> int:
        """Number of facts with ACTIVE status."""
        return sum(
            1 for fact in self.facts.values() if fact.status == FactStatus.ACTIVE
        )

    def __repr__(self) -> str:
        return (
            f"MemoryState(facts={self.fact_count}, time={self.time}, "
            f"tau={self.tau})"
        )
