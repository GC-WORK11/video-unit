"""VideoUnit error classes."""


class VideoUnitError(Exception):
    """Base exception for VideoUnit errors."""

    pass


class EvaluationError(VideoUnitError):
    """Raised when video evaluation fails."""

    pass


class ContractError(VideoUnitError):
    """Raised when contract validation fails."""

    pass


class ConnectionError(VideoUnitError):
    """Raised when connection to backend fails."""

    pass


class TimeoutError(VideoUnitError):
    """Raised when a request times out."""

    pass
