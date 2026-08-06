"""Strict C-model exception hierarchy.

The production simulator is deliberately fail-closed: unsupported operations,
missing external tools, malformed traces, and invalid resource mappings are
reported as errors instead of silently switching to an analytical model.
"""


class CModelError(RuntimeError):
    """Base exception for all strict C-model failures."""


class ConfigurationError(CModelError):
    """Raised when an architecture or backend configuration is invalid."""


class UnsupportedOperatorError(CModelError):
    """Raised when the functional IR contains an unsupported operation."""


class BackendUnavailableError(CModelError):
    """Raised when a required external backend cannot be executed."""


class BackendExecutionError(CModelError):
    """Raised when an external backend exits unsuccessfully."""


class BackendOutputError(CModelError):
    """Raised when backend output is missing, ambiguous, or malformed."""


class MappingError(CModelError):
    """Raised when an operation cannot be mapped onto the declared hardware."""


class ResourceError(CModelError):
    """Raised when a micro-operation requests an invalid resource."""


class DeadlockError(CModelError):
    """Raised when the micro-operation graph cannot make progress."""
