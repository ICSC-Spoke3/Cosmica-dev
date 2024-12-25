def first_value_with_key_in(d, keys):
    for k, v in d.items():
        if k in keys:
            return v
    return []