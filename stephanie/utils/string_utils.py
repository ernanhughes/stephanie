
def trunc(s: str | None, n: int = 200) -> str | None:
    if not isinstance(s, str):
        return s
    return s if len(s) <= n else s[:n] + "…"
