def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ("true", "1", "yes", "y")