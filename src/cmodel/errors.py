"""Errors raised by the strict architecture C-model."""


class CModelError(RuntimeError):
    """Base class for strict C-model failures."""


class ConfigurationError(CModelError):
    """Configuration is incomplete or inconsistent."""


class BackendError(CModelError):
    """External simulator integration failed."""


class BackendProtocolError(BackendError):
    """External simulator violated the declared protocol."""


class BackendVersionError(BackendError):
    """External simulator version or commit does not match the configuration."""


class MappingError(CModelError):
    """Operator cannot be mapped under the declared mapping."""


class ResourceError(CModelError):
    """A resource request is invalid or cannot be serviced."""


class DeadlockError(CModelError):
    """Simulation cannot make forward progress."""
