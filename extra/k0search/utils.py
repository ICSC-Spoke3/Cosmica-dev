def first_value_with_key_in(d, keys):
    """
    Return the first value in the dictionary that has a key in the list of keys.
    :param d: dictionary
    :param keys: list of keys
    :return: value
    """
    
    for k, v in d.items():
        if k in keys:
            return v
    return []