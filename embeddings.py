from __future__ import annotations

import hashlib
import math
import os
import re
from typing import List

from config import EMBEDDING_BACKEND, EMBEDDING_MODEL


class EmbeddingGenerator:
    def __init__(self, model: str = EMBEDDING_MODEL, local_dimension: int = 256):
        self.model = model
        self.local_dimension = local_dimension
        self.client = None
        self.backend = "local"

        api_key = os.getenv("OPENAI_API_KEY")
        if EMBEDDING_BACKEND == "openai" and api_key:
            try:
                from openai import OpenAI
            except Exception:
                self.client = None
            else:
                self.client = OpenAI(api_key=api_key)
                self.backend = "openai"

    def generate_embedding(self, text: str) -> List[float]:
        if self.backend == "openai" and self.client is not None:
            try:
                response = self.client.embeddings.create(input=text, model=self.model)
            except Exception:
                return self._generate_local_embedding(text)
            return list(response.data[0].embedding)
        return self._generate_local_embedding(text)

    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        if self.backend == "openai" and self.client is not None:
            all_embeddings: List[List[float]] = []
            for index in range(0, len(texts), batch_size):
                batch = texts[index : index + batch_size]
                try:
                    response = self.client.embeddings.create(input=batch, model=self.model)
                except Exception:
                    return [self._generate_local_embedding(text) for text in texts]
                all_embeddings.extend(list(item.embedding) for item in response.data)
            return all_embeddings
        return [self._generate_local_embedding(text) for text in texts]

    def _generate_local_embedding(self, text: str) -> List[float]:
        vector = [0.0] * self.local_dimension
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            slot = int.from_bytes(digest[:4], "big") % self.local_dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[slot] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]
