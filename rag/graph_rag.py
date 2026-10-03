"""GraphRAG-lite: connect facts that are split across chunks.

THE PROBLEM THIS SOLVES
------------------------
Plain vector/BM25 retrieval treats each chunk independently. If chunk 4 says
"... (i) as Chief Operating Officer and (ii) resigned from the Board" and the
antecedent for "(ii)" -- WHO resigned -- is only unambiguous if you also read
the sentence naming "Guy F. Cipriani" nearby, a similarity search over chunk 6
alone ("... and (ii) resigned from the Board.") has no way to recover that
link unless both chunks happen to get retrieved together AND the LLM correctly
stitches them at generation time. That's fragile.

THE FIX: A KNOWLEDGE GRAPH
---------------------------
We ask the LLM to read each chunk once and extract explicit facts as
(subject, relation, object) triples, resolving pronouns/references like
"(ii)" to what they actually mean using the chunk's own context. Those
triples become edges in a graph: "Guy F. Cipriani" --[resigned from]--> "the
Board". At query time we find which graph nodes (entities) the question
mentions, walk one hop out, and hand those facts to the LLM alongside the
normal retrieved chunks -- so it sees the resolved fact directly instead of
having to re-derive it from scattered, ambiguous chunk text.

This is NOT full GraphRAG (Microsoft's version also builds community
summaries for whole-graph "tell me everything about X" questions). This is
the practical subset that fixes cross-chunk entity/reference linking, which is
exactly the symptom you're describing.

COST NOTE: building the graph costs one LLM call per chunk (extraction), on
top of the embedding calls ingestion already makes. That's why this is a
separate, explicit "Build knowledge graph" step in the app rather than
something that happens automatically on every PDF ingest.
"""
import json
import re

from .llm_client import multi_key_chat_complete

TRIPLE_EXTRACTION_PROMPT = """Extract factual relationships from the text below as a JSON array of triples.
Each triple: {{"subject": "...", "relation": "...", "object": "..."}}

Rules:
- Use short, consistent entity names (e.g. always "Guy F. Cipriani", never mix "he" / "Mr. Cipriani" / "the officer" for the same person).
- Resolve pronouns and references like "(ii)", "the above", "he", "it" to the actual entity they refer to, using the surrounding text.
- Only extract clear, explicit facts stated in the text. Skip vague or uncertain statements.
- Return ONLY the JSON array, nothing else. If there are no clear facts, return [].

Text:
{text}
"""


def extract_triples(text: str, base_url: str, api_keys: list[str], model: str) -> list[dict]:
    raw = multi_key_chat_complete(
        base_url,
        api_keys,
        system="You are a precise information-extraction system. You output only valid JSON, nothing else.",
        user=TRIPLE_EXTRACTION_PROMPT.format(text=text),
        model=model,
        max_tokens=600,
    )
    # Models sometimes wrap JSON in prose or ```json fences -- pull out the array.
    match = re.search(r"\[.*\]", raw or "", re.DOTALL)
    if not match:
        return []
    try:
        triples = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return [
        t for t in triples
        if isinstance(t, dict) and {"subject", "relation", "object"} <= t.keys()
        and all(isinstance(t[k], str) and t[k].strip() for k in ("subject", "relation", "object"))
    ]


class KnowledgeGraph:
    """A tiny in-memory graph: node -> [(relation, other_node, source_chunk_id), ...]."""

    def __init__(self):
        self.edges: dict[str, list[tuple[str, str, str]]] = {}
        self.skipped: list[tuple[str, str]] = []  # (chunk_id, reason) for chunks that failed

    def add_triple(self, subject: str, relation: str, obj: str, chunk_id: str):
        subject, relation, obj = subject.strip(), relation.strip(), obj.strip()
        self.edges.setdefault(subject, []).append((relation, obj, chunk_id))
        self.edges.setdefault(obj, []).append((f"<- {relation}", subject, chunk_id))

    def build(self, documents: list[dict], base_url: str, api_keys: list[str], model: str, progress_cb=None):
        self.skipped = []
        for i, doc in enumerate(documents):
            try:
                triples = extract_triples(doc["text"], base_url, api_keys, model)
            except Exception as exc:  # noqa: BLE001 - one failed chunk should not abort the whole graph
                self.skipped.append((doc["id"], str(exc)[:200]))
                triples = []
            for triple in triples:
                self.add_triple(triple["subject"], triple["relation"], triple["object"], doc["id"])
            if progress_cb:
                progress_cb(i + 1, len(documents))

    def matching_nodes(self, query: str) -> list[str]:
        """Which graph entities does the query mention? Simple case-insensitive
        substring matching -- good enough for names/titles. A production system
        would use embedding similarity or a proper NER pass on the query too."""
        q = query.lower()
        hits = []
        for node in self.edges:
            node_l = node.lower()
            if node_l in q or any(w in q for w in node_l.split() if len(w) > 3):
                hits.append(node)
        return hits

    def related_facts(self, query: str, max_facts: int = 8) -> list[tuple[str, str, str, str]]:
        """(subject, relation, object, chunk_id) facts one hop from any entity
        the query mentions -- these can come from chunks the vector/BM25
        retriever never surfaced."""
        facts, seen = [], set()
        for node in self.matching_nodes(query):
            for relation, other, chunk_id in self.edges.get(node, []):
                key = (node, relation, other)
                if key in seen:
                    continue
                seen.add(key)
                facts.append((node, relation, other, chunk_id))
                if len(facts) >= max_facts:
                    return facts
        return facts

    def all_triples(self) -> list[tuple[str, str, str, str]]:
        """Every (subject, relation, object, chunk_id), deduplicated, for display."""
        seen, out = set(), []
        for node, rels in self.edges.items():
            for relation, other, chunk_id in rels:
                if relation.startswith("<- "):
                    continue  # skip the reverse-edge duplicates for display
                key = (node, relation, other, chunk_id)
                if key not in seen:
                    seen.add(key)
                    out.append(key)
        return out

    def stats(self) -> str:
        n_relationships = len(self.all_triples())
        return f"knowledge graph: {len(self.edges)} entities, {n_relationships} relationships"
