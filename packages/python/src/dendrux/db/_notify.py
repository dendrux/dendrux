"""Internal PostgreSQL notification channel validation."""


def validate_channel(channel: str) -> str:
    """Validate a PostgreSQL notification channel without echoing its value."""
    if (
        not isinstance(channel, str)
        or not channel
        or "\x00" in channel
        or len(channel.encode()) >= 64
    ):
        raise ValueError("channel must contain 1–63 UTF-8 bytes and no NUL.")
    return channel
