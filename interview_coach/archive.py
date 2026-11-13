"""Chroma-backed archive ingestion and company/role-filtered retrieval."""

from __future__ import annotations

import csv
import hashlib
import io
import math
from pathlib import Path
import re
from typing import Any

import chromadb
from chromadb.api.types import EmbeddingFunction
from chromadb.api.types import Documents, Embeddings

REQUIRED_COLUMNS = {"company", "role", "type", "content"}
EMBEDDING_DIMENSIONS = 384


class LocalFeatureEmbedding(EmbeddingFunction[Documents]):
    """Deterministic, dependency-free text vectors for offline archive search."""

    def __init__(self) -> None:
        pass

    @staticmethod
    def name() -> str:
        return "interview-studio-local-features"

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "LocalFeatureEmbedding":
        return LocalFeatureEmbedding()

    def get_config(self) -> dict[str, Any]:
        return {}

    def default_space(self) -> str:
        return "cosine"

    def __call__(self, input: Documents) -> Embeddings:
        embeddings: Embeddings = []
        for document in input:
            words = re.findall(r"[a-z0-9]+", document.lower())
            features = [f"word:{word}" for word in words]
            features.extend(
                f"pair:{left}_{right}" for left, right in zip(words, words[1:])
            )
            vector = [0.0] * EMBEDDING_DIMENSIONS
            for feature in features:
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "little") % EMBEDDING_DIMENSIONS
                sign = 1.0 if digest[4] & 1 else -1.0
                vector[bucket] += sign
            norm = math.sqrt(sum(value * value for value in vector))
            embeddings.append(
                [value / norm for value in vector] if norm else vector
            )
        return embeddings


class InterviewArchive:
    """Persistent vector index for interview questions and feedback examples."""

    def __init__(self, persist_directory: str | Path) -> None:
        self.client = chromadb.PersistentClient(path=str(persist_directory))
        self.embedding_function = LocalFeatureEmbedding()
        self.collection = self.client.get_or_create_collection(
            name="interview_archive",
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_function,
        )

    @property
    def count(self) -> int:
        return self.collection.count()

    def ingest_csv(self, csv_content: bytes, *, source_label: str, dataset: str) -> int:
        """Validate and upsert a CSV archive; return the number of new records."""
        try:
            text = csv_content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("The archive must be a UTF-8 encoded CSV file.") from exc

        reader = csv.DictReader(io.StringIO(text))
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(
                "CSV is missing required columns: " + ", ".join(sorted(missing))
            )

        records: dict[str, tuple[str, dict[str, str]]] = {}
        for line_number, row in enumerate(reader, start=2):
            company = (row.get("company") or "").strip()
            role = (row.get("role") or "").strip()
            kind = (row.get("type") or "").strip().lower()
            content = (row.get("content") or "").strip()
            if not all((company, role, kind, content)):
                raise ValueError(
                    f"CSV row {line_number} has an empty company, role, type, or content."
                )
            if kind not in {"question", "feedback"}:
                raise ValueError(
                    f"CSV row {line_number} has unsupported type '{kind}'; use question or feedback."
                )
            source = (row.get("source") or "").strip() or source_label
            round_name = (row.get("round") or "").strip()
            stable_values = [company, role, kind, content, source]
            if round_name:
                stable_values.append(round_name)
            stable_key = "\0".join(stable_values)
            record_id = hashlib.sha256(stable_key.encode("utf-8")).hexdigest()
            metadata = {
                "company": company,
                "role": role,
                "kind": kind,
                "source": source,
                "dataset": dataset,
            }
            if round_name:
                metadata["round"] = round_name
            records[record_id] = (
                content,
                metadata,
            )

        if not records:
            raise ValueError("The CSV contains no interview records.")

        ids = list(records)
        existing = set(self.collection.get(ids=ids, include=[])["ids"])
        new_ids = [record_id for record_id in ids if record_id not in existing]
        if not new_ids:
            return 0

        self.collection.add(
            ids=new_ids,
            documents=[records[record_id][0] for record_id in new_ids],
            metadatas=[records[record_id][1] for record_id in new_ids],
        )
        return len(new_ids)

    def retrieve(
        self,
        *,
        company: str,
        role: str,
        query: str,
        kind: str | None = None,
        limit: int = 4,
        exclude_content: list[str] | None = None,
        round_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Prefer exact role evidence; broaden to company-only evidence if needed."""
        if self.count == 0:
            return []

        role_filter: dict[str, Any] = {
            "$and": [
                {"company": {"$eq": company.strip()}},
                {"role": {"$eq": role.strip()}},
            ]
        }
        company_conditions: list[dict[str, Any]] = [
            {"company": {"$eq": company.strip()}}
        ]
        if kind:
            role_filter["$and"].append({"kind": {"$eq": kind}})
            company_conditions.append({"kind": {"$eq": kind}})
        if round_name:
            role_filter["$and"].append({"round": {"$eq": round_name}})
            company_conditions.append({"round": {"$eq": round_name}})
        company_filter: dict[str, Any] = (
            company_conditions[0]
            if len(company_conditions) == 1
            else {"$and": company_conditions}
        )

        matches = self._query(
            query,
            role_filter,
            min(limit, self.count),
            exact_role=True,
            exclude_content=exclude_content,
        )
        if matches:
            return matches
        return self._query(
            query,
            company_filter,
            min(limit, self.count),
            exact_role=False,
            exclude_content=exclude_content,
        )

    def _query(
        self,
        query: str,
        where: dict[str, Any],
        limit: int,
        *,
        exact_role: bool,
        exclude_content: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        query_options: dict[str, Any] = {
            "query_texts": [query],
            "n_results": limit,
            "where": where,
            "include": ["documents", "metadatas"],
        }
        exclusions = [content for content in (exclude_content or []) if content]
        if exclusions:
            filters = [{"$not_contains": content} for content in exclusions]
            query_options["where_document"] = (
                filters[0] if len(filters) == 1 else {"$and": filters}
            )
        result = self.collection.query(**query_options)
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        return [
            {"content": document, "metadata": metadata, "exact_role": exact_role}
            for document, metadata in zip(documents, metadatas)
        ]
