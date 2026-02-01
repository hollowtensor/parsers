"""Registry for dynamically registering and retrieving parsers and extractors."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from parsers.methods.base import BaseParser
    from parsers.extraction.base import BaseExtractor


class _Registry[T]:
    """Generic registry for named components."""

    def __init__(self) -> None:
        self._registry: dict[str, type[T]] = {}

    def register(self, name: str, cls: type[T]) -> None:
        self._registry[name] = cls

    def get(self, name: str) -> type[T]:
        if name not in self._registry:
            available = ", ".join(self._registry.keys())
            raise KeyError(f"Unknown component '{name}'. Available: {available}")
        return self._registry[name]

    def list(self) -> list[str]:
        return list(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._registry


ParserRegistry: _Registry[BaseParser] = _Registry()
ExtractorRegistry: _Registry[BaseExtractor] = _Registry()
