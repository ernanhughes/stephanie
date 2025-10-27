from datetime import datetime

def datetimeformat(value, fmt="%Y-%m-%d %H:%M:%S"):
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value).strftime(fmt)
    if isinstance(value, datetime):
        return value.strftime(fmt)
    return str(value)


def short_dt(value, fmt="%Y-%m-%d %H:%M", tz="Europe/Dublin"):
    if value is None:
        return ""
    # accept datetime or ISO string
    if isinstance(value, str):
        try:
            # minimal ISO parse
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            return value[:16]  # last resort trim
    try:
        if value.tzinfo is None:
            value = value.replace(tzinfo=ZoneInfo("UTC"))
        value = value.astimezone(ZoneInfo(tz))
        return value.strftime(fmt)
    except Exception:
        return str(value)

