"""Test doubles shared by the test suite.

* ``HashingEncoder`` replaces the sentence-transformers model. It hashes words
  into a fixed-size bag-of-words vector, so texts that share words get similar
  vectors. That is enough to test retrieval logic deterministically.
* ``FakeLLM`` replaces the API client and returns (or raises) scripted replies.
* ``build_pdf`` writes real (tiny) PDF files, so the loader is tested against
  actual PDF bytes without needing a PDF-writing library.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np

from notemind.bm25_index import STOPWORDS
from notemind.config import GenerationConfig
from notemind.embedder import l2_normalize
from notemind.llm_client import LLMClient


class HashingEncoder:
    """Deterministic stand-in for the embedding model (implements TextEncoder).

    Bag-of-words over non-stopwords, hashed into ``dimension`` buckets.
    """

    model_name = "test-hashing-encoder"

    def __init__(self, dimension: int = 256) -> None:
        self._dimension = dimension
        self.calls = 0  # how many times encode() ran, to test caching

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self.calls += 1
        matrix = np.zeros((len(texts), self._dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                if word in STOPWORDS:  # like a real model, ignore filler words
                    continue
                bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self._dimension
                matrix[row, bucket] += 1.0
        return l2_normalize(matrix)

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


class FakeLLM(LLMClient):
    """Scripted LLM. Each item in ``replies`` is returned (or raised) in order."""

    provider = "fake"

    def __init__(self, replies: Sequence[object] = (), config: GenerationConfig | None = None) -> None:
        super().__init__("fake-model", config or GenerationConfig(max_retries=0))
        self.replies = list(replies)
        self.calls: list[dict] = []

    def _call(self, system: str, user: str, *, json_mode: bool, max_tokens: int) -> str:
        self.calls.append({"system": system, "user": user, "json_mode": json_mode, "max_tokens": max_tokens})
        if not self.replies:
            raise AssertionError("FakeLLM called more times than scripted")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return str(reply)


def build_pdf(pages: Sequence[str]) -> bytes:
    """Write a minimal, valid PDF with one page per string ("" = blank page)."""
    objects: dict[int, bytes] = {}
    kids: list[int] = []
    next_id = 4  # 1 = catalog, 2 = page tree, 3 = font
    for text in pages:
        page_id, content_id = next_id, next_id + 1
        next_id += 2
        ops = ["BT", "/F1 11 Tf", "14 TL", "50 800 Td"]
        for line in text.split("\n") if text else []:
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            ops.append(f"({escaped}) Tj T*")
        ops.append("ET")
        stream = "\n".join(ops).encode("latin-1")
        objects[content_id] = b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents {content_id} 0 R "
            "/Resources << /Font << /F1 3 0 R >> >> >>"
        ).encode()
        kids.append(page_id)
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[2] = (
        f"<< /Type /Pages /Kids [{' '.join(f'{k} 0 R' for k in kids)}] /Count {len(kids)} >>"
    ).encode()
    objects[3] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for obj_id in sorted(objects):
        offsets[obj_id] = len(out)
        out += f"{obj_id} 0 obj\n".encode() + objects[obj_id] + b"\nendobj\n"
    xref_at = len(out)
    size = max(objects) + 1
    out += f"xref\n0 {size}\n0000000000 65535 f \n".encode()
    for obj_id in range(1, size):
        out += f"{offsets[obj_id]:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n".encode()
    return bytes(out)
