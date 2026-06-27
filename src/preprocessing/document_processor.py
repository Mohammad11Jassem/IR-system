import re
import unicodedata
from dataclasses import dataclass
from typing import List

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


@dataclass
class DocumentProcessorConfig:
    lowercase: bool = True
    normalize_unicode: bool = True
    preserve_hyphen: bool = True
    collapse_whitespace: bool = True

    tokenize: bool = True
    remove_stopwords: bool = True


class DocumentProcessor:
    """
    Document/query preprocessing for traditional IR models.

    This class is used for BOTH:
    - Terrier document indexing
    - Terrier query processing
    """

    def __init__(
        self,
        config: DocumentProcessorConfig = DocumentProcessorConfig(),
    ):
        self.config = config

        if self.config.preserve_hyphen:
            self._punct_pattern = re.compile(r"[^\w\s\-]")
        else:
            self._punct_pattern = re.compile(r"[^\w\s]")

        self._space_pattern = re.compile(r"\s+")
        self.stopwords = set(stopwords.words("english"))

    def normalize(self, text: str) -> str:
        if text is None:
            return ""

        text = str(text)

        if self.config.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)

        if self.config.lowercase:
            text = text.lower()

        text = self._punct_pattern.sub(" ", text)

        if self.config.collapse_whitespace:
            text = self._space_pattern.sub(" ", text).strip()

        return text

    def tokenize(self, text: str) -> List[str]:
        return word_tokenize(text)

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        return [
            token
            for token in tokens
            if token not in self.stopwords
        ]

    def process(self, text: str) -> List[str]:
        text = self.normalize(text)

        if not self.config.tokenize:
            return [text] if text else []

        tokens = self.tokenize(text)

        if self.config.remove_stopwords:
            tokens = self.remove_stopwords(tokens)

        return tokens

    def process_to_text(self, text: str) -> str:
        return " ".join(self.process(text))

    def is_empty(self, text: str) -> bool:
        return len(self.process(text)) == 0