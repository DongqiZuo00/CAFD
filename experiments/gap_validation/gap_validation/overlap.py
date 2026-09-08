from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from datasketch import MinHash


def normalize_prompt(text: str) -> str:
    text = text.casefold().replace("\u00a0", " ")
    text = re.sub(r"https?://(?:www\.)?", "", text)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"[^\w\d]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _shingles(text: str, width: int) -> set[str]:
    tokens = normalize_prompt(text).split()
    if len(tokens) < width:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[index : index + width]) for index in range(len(tokens) - width + 1)}


def minhash(text: str, width: int = 8, num_perm: int = 256) -> MinHash:
    value = MinHash(num_perm=num_perm, seed=42)
    for shingle in sorted(_shingles(text, width)):
        value.update(shingle.encode())
    return value


@dataclass(frozen=True)
class OverlapMatch:
    training_id: str
    training_source: str
    target_id: str
    target_source: str
    match_type: str
    similarity: float


def find_overlaps(
    training: Iterable[dict[str, Any]],
    targets: Iterable[dict[str, Any]],
    *,
    threshold: float = 0.80,
    ngram_size: int = 8,
    num_perm: int = 256,
) -> list[OverlapMatch]:
    target_rows = list(targets)
    exact_prompts: dict[str, list[dict[str, Any]]] = {}
    exact_ids: dict[str, list[dict[str, Any]]] = {}
    exact_urls: dict[str, list[dict[str, Any]]] = {}
    target_hashes: list[tuple[dict[str, Any], MinHash]] = []
    for row in target_rows:
        exact_prompts.setdefault(normalize_prompt(str(row.get("prompt", ""))), []).append(row)
        exact_ids.setdefault(str(row.get("id", "")), []).append(row)
        url = str(row.get("url", ""))
        if url:
            exact_urls.setdefault(url, []).append(row)
        target_hashes.append((row, minhash(str(row.get("prompt", "")), ngram_size, num_perm)))

    matches: list[OverlapMatch] = []
    seen: set[tuple[str, str, str]] = set()
    for row in training:
        training_id = str(row.get("id", ""))
        training_source = str(row.get("source", ""))
        candidates: list[tuple[dict[str, Any], str, float]] = []
        normalized = normalize_prompt(str(row.get("prompt", "")))
        candidates.extend((target, "normalized_prompt_exact", 1.0) for target in exact_prompts.get(normalized, []))
        candidates.extend((target, "problem_id_exact", 1.0) for target in exact_ids.get(training_id, []))
        url = str(row.get("url", ""))
        candidates.extend((target, "problem_url_exact", 1.0) for target in exact_urls.get(url, []) if url)
        training_hash = minhash(str(row.get("prompt", "")), ngram_size, num_perm)
        for target, target_hash in target_hashes:
            similarity = training_hash.jaccard(target_hash)
            if similarity >= threshold:
                candidates.append((target, "minhash_near_duplicate", similarity))
        solution = str(row.get("solution", "")).strip()
        solution_hash = hashlib.sha256(solution.encode()).hexdigest() if solution else None
        for target in target_rows:
            target_solution = str(target.get("solution", "")).strip()
            if solution_hash and target_solution and hashlib.sha256(target_solution.encode()).hexdigest() == solution_hash:
                candidates.append((target, "reference_solution_exact", 1.0))
        for target, match_type, similarity in candidates:
            key = (training_id, str(target.get("id", "")), match_type)
            if key in seen:
                continue
            seen.add(key)
            matches.append(
                OverlapMatch(
                    training_id=training_id,
                    training_source=training_source,
                    target_id=str(target.get("id", "")),
                    target_source=str(target.get("source", "")),
                    match_type=match_type,
                    similarity=float(similarity),
                )
            )
    return matches


def match_dicts(matches: Iterable[OverlapMatch]) -> list[dict[str, Any]]:
    return [asdict(item) for item in matches]

