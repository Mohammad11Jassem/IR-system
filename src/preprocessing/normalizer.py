import re
import unicodedata
from dataclasses import dataclass


@dataclass
class NormalizerConfig:
    lowercase: bool = True
    remove_punctuation: bool = True
    normalize_unicode: bool = True
    collapse_whitespace: bool = True


class Normalizer:
    """
    Production-grade text normalizer for IR systems.
    Works for both documents and queries.
    """

    def __init__(self, config: NormalizerConfig = NormalizerConfig()):
        self.config = config

        # keep biomedical-safe punctuation (important!)
        self._punct_pattern = re.compile(r"[^\w\s\-]")  
        # keeps: letters, numbers, underscore, spaces, hyphen

        self._space_pattern = re.compile(r"\s+")

    def normalize(self, text: str) -> str:
        if text is None:
            return ""

        text = str(text)

        # 1. Unicode normalization (important for scientific text)
        if self.config.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)

        # 2. Lowercasing
        if self.config.lowercase:
            text = text.lower()

        # 3. Remove punctuation (SAFE for biomedical IR)
        if self.config.remove_punctuation:
            text = self._punct_pattern.sub(" ", text)

        # 4. Collapse whitespace
        if self.config.collapse_whitespace:
            text = self._space_pattern.sub(" ", text).strip()

        return text

    def normalize_batch(self, texts):
        return [self.normalize(t) for t in texts]