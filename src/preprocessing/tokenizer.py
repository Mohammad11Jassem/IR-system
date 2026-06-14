import re
from typing import List


class Tokenizer:
    """
    IR-safe tokenizer for biomedical text (Medline dataset).
    Designed to preserve scientific meaning.
    """

    def __init__(self):
        # Matches:
        # - words (gene, cancer)
        # - numbers (123)
        # - biomedical tokens (TNF-alpha, IL-2, BRCA1)
        self.pattern = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []

        text = text.lower()

        tokens = self.pattern.findall(text)

        return tokens

    def tokenize_batch(self, texts):
        return [self.tokenize(t) for t in texts]