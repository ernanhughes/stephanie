# stephanie/services/bus/errors.py
class BusError(Exception):
    """Base exception for all bus-related errors."""
    pass

class BusConnectionError(BusError):
    """Raised when connection to bus backend fails."""
    pass

class BusPublishError(BusError):
    """Raised when message publishing fails."""
    pass

class BusSubscribeError(BusError):
    """Raised when subscription fails."""
    pass

class BusRequestError(BusError):
    """Raised when a request/reply operation fails."""
    pass
