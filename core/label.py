from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional


@dataclass(frozen=True)
class Label:
    """
    Immutable, hashable label that stores a mapping:
      { pomap_name: { leaf_field: value, ... }, ... }

    Internal representation is canonicalized to make equality/hash stable.
    """
    _mapping: Tuple[Tuple[str, Tuple[Tuple[str, Any], ...]], ...]  # ((pomap, ((k,v),...)), ...)

    @staticmethod
    def from_dict(mapping: Dict[str, Dict[str, Any]]) -> "Label":
        items: List[Tuple[str, Tuple[Tuple[str, Any], ...]]] = []
        for pomap_name in sorted(mapping.keys()):
            lbl = mapping[pomap_name] or {}
            lbl_items = tuple(sorted(lbl.items()))
            items.append((pomap_name, lbl_items))
        return Label(tuple(items))

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return {pomap: dict(tuples) for pomap, tuples in self._mapping}

    def for_pomap(self, pomap_name: str) -> Optional[Dict[str, Any]]:
        for p, tuples in self._mapping:
            if p == pomap_name:
                return dict(tuples)
        return None

    def merged_with(self, other: "Label") -> "Label":
        a = self.to_dict()
        b = other.to_dict()
        intersect = set(a).intersection(b)
        if intersect:
            raise ValueError(f"Cannot merge Labels: overlapping pomap names {intersect}")
        merged = {**a, **b}
        return Label.from_dict(merged)

    def matches_partial(self, partial: dict[str, dict]) -> bool:
        """
        Return True if this label contains at least all (pomap, subdict)
        pairs specified in `partial`.
        Example partial = {"p1": {"val": "a"}}
        """
        for pname, subdict in partial.items():
            # If the pomap isn't in this label, fail
            if pname not in dict(self._mapping):
                return False
            # Check all k/v pairs in subdict
            this_dict = dict(dict(self._mapping)[pname])
            for k, v in subdict.items():
                if this_dict.get(k) != v:
                    return False
        return True

    def __repr__(self):
        return f"Label({self.to_dict()})"

