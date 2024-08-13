class DictToObj:
    """
    A class to convert nested dictionaries to objects (to handle custom config objects).
    """
    def __init__(self, d):
        for key, value in d.items():
            if isinstance(value, dict):
                d[key] = DictToObj(value)
        self.__dict__.update(d)