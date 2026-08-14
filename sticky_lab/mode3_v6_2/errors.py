"""Fail-closed exception taxonomy for the V6.2 protocol."""

from __future__ import annotations


class V62Error(RuntimeError):
    """Base class for V6.2 failures."""


class CandidateRejected(V62Error):
    """The only family that a worker may convert into a candidate rejection."""


class CandidateRejectedRadius(CandidateRejected):
    pass


class CandidateRejectedTokenRealization(CandidateRejected):
    pass


class CandidateRejectedDegenerateCluster(CandidateRejected):
    pass


class ProtocolViolation(V62Error):
    pass


class RoleLeakage(ProtocolViolation):
    pass


class ManifestMismatch(ProtocolViolation):
    pass


class ShapeMismatch(ProtocolViolation):
    pass


class NumericalNonFinite(ProtocolViolation):
    pass


class CacheCorruption(ProtocolViolation):
    pass
