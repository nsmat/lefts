from immutables import Map as FrozenDict
from typing import Any, Dict


class Label:
    def __init__(self, mapping: Dict[str, Any]):
        """
        mapping: leaf PoMap name -> label value (single column per leaf)
        """
        self._mapping: FrozenDict = FrozenDict(mapping)

        # canonical form for hashing / equality
        self._canonical = tuple(sorted(self._mapping.items()))

    def __hash__(self):
        return hash(self._canonical)

    def __eq__(self, other):
        return isinstance(other, Label) and self._canonical == other._canonical

    def __getitem__(self, pomap_name: str):
        return self._mapping[pomap_name]

    def __contains__(self, pomap_name: str):
        return pomap_name in self._mapping

    def as_dict(self) -> Dict[str, Any]:
        """Return a regular dict copy."""
        return dict(self._mapping)

    def matches_partial(self, partial: Dict[str, Any]) -> bool:
        """Return True if all specified leaf labels match."""
        for pomap_name, value in partial.items():
            if pomap_name not in self._mapping or self._mapping[pomap_name] != value:
                return False
        return True

    def __repr__(self):
        return f"Label({self.as_dict()})"
