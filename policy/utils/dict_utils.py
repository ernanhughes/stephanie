def deep_get(d: dict, *keys, default=0.0):
    """
    Safely retrieve nested dict values.

    Example:
        deep_get(row, "energy", "geometry", "spectral", "participation_ratio")
    """
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k)
        if d is None:
            return default
    return d
