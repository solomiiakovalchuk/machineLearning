"""Controlled exceptions. Each carries a stable CLI exit code."""


class ProjectError(Exception):
    """Base class; the CLI turns subclasses into a stderr message and an exit code."""

    exit_code = 1


class ConfigError(ProjectError):
    exit_code = 3


class InputFileError(ProjectError):
    """Input file is missing, unreadable or not valid JSON/CSV."""

    exit_code = 4


class InputValidationError(ProjectError):
    """Record violates the schema contract (blocks prediction)."""

    exit_code = 5


class BatchSizeError(InputValidationError):
    exit_code = 6


class ArtifactIntegrityError(ProjectError):
    """A hash, version or cross-file consistency check failed."""

    exit_code = 7


class RebuildError(ProjectError):
    exit_code = 8
