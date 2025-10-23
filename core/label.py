class Label:
    def __init__(self, **kwargs):
        self._data = dict(kwargs)
        self._hash = hash(frozenset(self._data.items()))

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def items(self):
        return self._data.items()

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def matches(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self._data:
                raise ValueError(f"Trying to match on unknown namespace {k}")
            elif self._data[k] != v:
                return False
        return True

    def drop(self, *keys):
        new_mapping = {k: v for k, v in self._data.items() if k not in keys}
        return Label(**new_mapping)

    def __eq__(self, other):
        if not isinstance(other, Label):
            return NotImplemented
        return self._data == other._data

    def __hash__(self):
        return self._hash

    def __repr__(self):
        return f"Label({self._data})"

    def column(self):
        return str(self._data)