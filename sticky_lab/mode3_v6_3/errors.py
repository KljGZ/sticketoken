"""Fail-closed exception taxonomy for the V6.3 protocol."""

class V63Error(RuntimeError):
    """Base class for all V6.3 failures."""


class CandidateRejected(V63Error):
    """Only this family may be converted into a candidate-level rejection."""


class CandidateRejectedRadius(CandidateRejected):
    pass


class CandidateRejectedTokenRealization(CandidateRejected):
    pass


class CandidateRejectedNonFiniteEmbedding(CandidateRejected):
    pass


class CandidateRejectedDegenerateFit(CandidateRejected):
    pass


class ProtocolViolation(V63Error):
    pass


class RoleLeakage(ProtocolViolation):
    pass


class ManifestMismatch(ProtocolViolation):
    pass


class ShapeMismatch(ProtocolViolation):
    pass


class CacheCorruption(ProtocolViolation):
    pass


class UnexpectedException(ProtocolViolation):
    pass


class BudgetLedgerMismatch(ProtocolViolation):
    pass


class DuplicateEncoderCallConflict(ProtocolViolation):
    pass


class NumericalNonFinite(ProtocolViolation):
    pass


class TokenizerHashMismatch(ProtocolViolation):
    pass


class ModelRevisionMismatch(ProtocolViolation):
    pass


class BudgetHardStop(V63Error):
    """Raised before a model call that would cross the registered hard stop."""
