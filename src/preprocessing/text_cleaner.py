import re
import unicodedata
from html import unescape


class EmbeddingTextCleaner:
    """
    Light text cleaner for embedding models.

    We intentionally avoid aggressive preprocessing because BERT/SentenceTransformer
    has its own tokenizer and benefits from natural sentence structure.
    """

    def __init__(
        self,
        normalize_unicode: bool = True,
        remove_html_tags: bool = True,
        collapse_whitespace: bool = True,
        max_chars: int | None = 4000,
    ):
        self.normalize_unicode = normalize_unicode
        self.remove_html_tags = remove_html_tags
        self.collapse_whitespace = collapse_whitespace
        self.max_chars = max_chars

        self._html_tag_pattern = re.compile(r"<[^>]+>")
        self._control_chars_pattern = re.compile(r"[\x00-\x1f\x7f-\x9f]")
        self._space_pattern = re.compile(r"\s+")

    def clean(self, text: str | None) -> str:
        if text is None:
            return ""

        text = str(text)

        if self.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)

        text = unescape(text)

        if self.remove_html_tags:
            text = self._html_tag_pattern.sub(" ", text)

        text = self._control_chars_pattern.sub(" ", text)

        if self.collapse_whitespace:
            text = self._space_pattern.sub(" ", text).strip()

        if self.max_chars is not None and len(text) > self.max_chars:
            text = text[: self.max_chars]

        return text