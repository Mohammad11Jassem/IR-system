import re
import unicodedata
from dataclasses import dataclass


@dataclass
class QueryProcessorConfig:
    lowercase: bool = True
    normalize_unicode: bool = True
    preserve_hyphen: bool = True
    collapse_whitespace: bool = True


class QueryProcessor:
    """
    Biomedical-safe query processor.

    This processor intentionally performs light normalization only.
    Terrier/PyTerrier still performs its own query analysis internally.

    We avoid aggressive preprocessing such as stemming, lemmatization,
    and stopword removal because biomedical terms can be sensitive:
    BRCA1, BRCA-1, IL-2, TNF-alpha, p53, etc.
    """

    def __init__(self, config: QueryProcessorConfig = QueryProcessorConfig()):
        self.config = config

        if self.config.preserve_hyphen:
            # Keep letters, numbers, underscore, whitespace, and hyphen.
            self._punct_pattern = re.compile(r"[^\w\s\-]")
        else:
            self._punct_pattern = re.compile(r"[^\w\s]")

        self._space_pattern = re.compile(r"\s+")

    def process(self, query: str) -> str:
        if query is None:
            return ""

        query = str(query)

        if self.config.normalize_unicode:
            query = unicodedata.normalize("NFKC", query)

        if self.config.lowercase:
            query = query.lower()

        query = self._punct_pattern.sub(" ", query)

        if self.config.collapse_whitespace:
            query = self._space_pattern.sub(" ", query).strip()

        return query

    def is_empty(self, query: str) -> bool:
        return len(self.process(query)) == 0