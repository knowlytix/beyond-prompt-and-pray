"""Graph-RAG retriever for search_policy.

Vendored from llm-tutorial Chapter 32, which uses kg-memory's QueryEngine
plus graph-driven alias expansion to route customer-vocabulary queries
("credit card", "SSN", "chargeback") onto canonical policy ids. The
retrieval logic walks the trained GMS store rather than counting tokens.

The pipeline per query:

  1. Expand the query — append every routed ``policy_id`` for any token
     that hits the ``ALIAS_TO_POLICY`` lexicon (built from the
     ``(policy_id, has_alias, alias)`` triples in the store).
  2. Gather GMS context via ``QueryEngine._gather_gms_context`` —
     dual-embedding GeometricSearch plus multi-hop path finding over the
     trained graph; the result is a list of ``FACT: h | r | t`` lines.
  3. Vote on policy by walking back from each FACT line's head/tail
     entities to its parent policy id (direct match, has_alias hop, or
     in_section prefix).

The store is ingested once from ``data/bank_policies.md`` by
``scripts/build_policy_rag_store.py``. ``has_alias`` triples come from
the Policy Aliases table; the same compendium drives both retrieval and
the Chapter 31 LoRA's canonical-summary template (so the policy id this
retriever picks lands the LoRA on its training distribution).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import torch

# GMS backend is optional (open-core). Import lazily so this module loads without
# a knowlytix license; the GMS-backed retriever raises a clear error when used.
# See glassloop.gms for the install pointer.
try:
    from knowlytix.knowledge.query import (
        DocGMSConfig,
        GMSExpertStore,
        LLMBackend,
        QueryEngine,
    )

    _GMS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the backend
    _GMS_AVAILABLE = False
    DocGMSConfig = GMSExpertStore = QueryEngine = None  # type: ignore[assignment]

    class LLMBackend:  # minimal stand-in so subclasses below still define
        """Placeholder base. The real GMS-backed retriever requires the licensed
        ``knowlytix`` package; see ``glassloop.gms`` for install instructions."""

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_STORE = _REPO_ROOT / "data" / "gms_policy_store"
_DEFAULT_POLICIES_DIR = _REPO_ROOT / "data" / "policies"

POLICY_IDS: list[str] = [
    "overdraft",
    "disputes",
    "fee_reversal",
    "account_closure",
    "pii_handling",
    "regulatory_escalation",
]


class _NoOpLLM(LLMBackend):
    """Stand-in LLMBackend for the offline path (AGENTLAB_USE_LLM_RAG=0).
    ``_gather_gms_context`` never calls the LLM in gms_only retrieval mode, so
    with synthesis disabled the retriever runs with zero GPU/LLM cost."""

    @property
    def model_name(self) -> str:
        return "noop"

    def call(self, system: str, user: str, max_tokens: int = 2048) -> str:
        return ""


class QwenLLMBackend(LLMBackend):
    """Qwen2.5-3B-Instruct as the Graph-RAG generator. GMS does the graph
    retrieval; this backend turns the retrieved policy context into a grounded,
    cited answer. Wraps the shared ``QwenAdapter`` (weights are process-cached,
    so constructing per call is cheap and reuses the loaded model)."""

    _MODEL = "Qwen/Qwen2.5-3B-Instruct"

    @property
    def model_name(self) -> str:
        return self._MODEL

    def call(self, system: str, user: str, max_tokens: int = 512) -> str:
        from glassloop.models.qwen_adapter import QwenAdapter

        return QwenAdapter(
            model=self._MODEL, system=system, max_new_tokens=max_tokens
        ).complete(user, max_tokens=max_tokens)


# The generator is grounded: it must answer from the retrieved policy text only,
# so the synthesized answer carries no facts the GMS store did not surface.
_SYNTH_SYSTEM = (
    "You are a retail-bank policy assistant. Using ONLY the policy text "
    "provided, answer the customer's question in one or two sentences and name "
    "the policy. Do not invent any fee amount, deadline or term that is not in "
    "the provided text. If the policy text does not answer the question, say so."
)


def _build_alias_lexicon(store: GMSExpertStore) -> dict[str, str]:
    """alias_lowercased -> policy_id, from every ``(policy_id, has_alias, alias)`` triple."""
    out: dict[str, str] = {}
    for h, r, t in store.triples:
        if r == "has_alias":
            out[t.lower()] = h
    return out


class PolicyRagRetriever:
    def __init__(
        self,
        store_path: Path | None = None,
        policies_dir: Path | None = None,
        device: torch.device | None = None,
        synthesize: bool | None = None,
    ) -> None:
        store_path = Path(store_path) if store_path is not None else _DEFAULT_STORE
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = DocGMSConfig(store_path=str(store_path), ingest_mode="regex")
        store = GMSExpertStore(config, device=device)
        if not store.load():
            raise RuntimeError(
                f"failed to load GMS policy store at {store_path!s}; "
                "run scripts/build_policy_rag_store.py first."
            )
        self.store = store
        self.alias_to_policy = _build_alias_lexicon(store)
        # AGENTLAB_USE_LLM_RAG=0 -> retrieval only, no Qwen synthesis (offline,
        # zero-GPU). Default is the full Graph RAG with a grounded generator.
        if synthesize is None:
            synthesize = os.environ.get("AGENTLAB_USE_LLM_RAG", "1") != "0"
        self.synthesize = synthesize
        self.llm: LLMBackend = QwenLLMBackend() if synthesize else _NoOpLLM()
        self.engine = QueryEngine(store=store, llm=self.llm, config=config)
        self.policies_dir = (
            Path(policies_dir) if policies_dir is not None else _DEFAULT_POLICIES_DIR
        )

    def _synthesize_answer(self, query: str, policy_id: str, policy_text: str) -> str:
        """Grounded generation: Qwen answers the query from the retrieved policy
        text only. Returns '' on any failure so retrieval still stands alone."""
        if not self.synthesize or not policy_text.strip():
            return ""
        user = (
            f"Policy ({policy_id}):\n{policy_text.strip()}\n\n"
            f"Customer question: {query}\n\nGrounded answer:"
        )
        try:
            return self.llm.call(_SYNTH_SYSTEM, user, max_tokens=160).strip()
        except Exception:
            return ""

    def _expand_query(self, query: str) -> str:
        """Append routed policy_ids for any alias token that hits the lexicon."""
        additions: list[str] = []
        for w in re.findall(r"[a-zA-Z0-9_-]+", query):
            wc = w.lower()
            if wc in self.alias_to_policy:
                additions.append(self.alias_to_policy[wc])
        if additions:
            return query + "  " + " ".join(additions)
        return query

    def _policy_for_entity(self, ent: str) -> str | None:
        """Walk a bare entity back to its parent policy via three hops:
        direct match, has_alias hop, or in_section prefix."""
        e = ent.lower()
        if e in POLICY_IDS:
            return e
        if "/" in e and e.split("/")[0] in POLICY_IDS:
            return e.split("/")[0]
        for h, r, _t in self.store.query_triples(tail=ent):
            if r == "has_alias" and h in POLICY_IDS:
                return h
        for _h, r, t in self.store.query_triples(head=ent):
            if r == "in_section":
                section = t.lower()
                for pid in POLICY_IDS:
                    if section.startswith(pid):
                        return pid
        return None

    def _policy_from_context(self, context: str) -> str | None:
        """Tally policy votes from the GMS context: direct mentions plus
        entity-walked votes for each FACT line."""
        counts = {pid: 0 for pid in POLICY_IDS}
        for line in context.splitlines():
            for pid in POLICY_IDS:
                if re.search(rf"(?<![a-z_]){pid}(?![a-z_])", line.lower()):
                    counts[pid] += 1
            if line.startswith("FACT:"):
                parts = [p.strip() for p in line[5:].split("|")]
                if len(parts) == 3:
                    h, _r, t = parts
                    for ent in (h, t):
                        pid = self._policy_for_entity(ent)
                        if pid is not None:
                            counts[pid] += 1
        if not any(counts.values()):
            return None
        return max(counts, key=counts.get)

    def search(self, query: str, k: int = 3) -> list[dict]:
        """Route query → GMS context → policy id, return top-1 (k retained for API parity)."""
        expanded = self._expand_query(query)
        context = self.engine._gather_gms_context(expanded)
        primary = self._policy_from_context(context)
        if primary is None:
            return []
        policy_path = self.policies_dir / f"{primary}.txt"
        text = policy_path.read_text() if policy_path.exists() else ""
        answer = self._synthesize_answer(query, primary, text)
        return [{
            "id": primary,
            "text": text,
            "answer": answer,
            "score": 1.0,
            "expanded_query": expanded,
            "gms_context_lines": len(context.splitlines()),
        }]


_DEFAULT_RETRIEVER: PolicyRagRetriever | None = None


def get_default_retriever() -> PolicyRagRetriever:
    global _DEFAULT_RETRIEVER
    if _DEFAULT_RETRIEVER is None:
        _DEFAULT_RETRIEVER = PolicyRagRetriever()
    return _DEFAULT_RETRIEVER
