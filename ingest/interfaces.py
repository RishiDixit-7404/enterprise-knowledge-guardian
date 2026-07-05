import datetime
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class FilingSource(ABC):
    @abstractmethod
    def fetch_filings(self, ticker: str, date_from: datetime.date, date_to: datetime.date) -> List[Dict[str, Any]]:
        pass

class NewsSource(ABC):
    @abstractmethod
    def fetch_news(self, query: str, date_from: datetime.date, date_to: datetime.date) -> List[Dict[str, Any]]:
        pass

class EmbeddingModel(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        pass

class LLMClient(ABC):
    @abstractmethod
    def complete(self, prompt: str) -> str:
        pass

    @abstractmethod
    def structured_call(self, schema: Dict[str, Any], messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def extract_relationships(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def generate_and_verify_claims(self, question: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Drafts factual claims based on the question and provided chunks, and verifies if each claim is supported."""
        pass

    @abstractmethod
    def generate_response(self, question: str, verified_claims: List[Dict[str, Any]]) -> str:
        """Renders the final text response from the supported verified claims, with inline citations."""
        pass


class FakeFilingSource(FilingSource):
    def fetch_filings(self, ticker: str, date_from: datetime.date, date_to: datetime.date) -> List[Dict[str, Any]]:
        """Returns mock SEC filings with predictable text content."""
        return [
            {
                "source": "edgar",
                "external_id": f"0000320193-23-000106-{ticker}",
                "title": f"{ticker} Annual Report 2023 Form 10-K",
                "company": f"Apple Inc. / {ticker}",
                "doc_type": "10-K",
                "published_at": datetime.datetime(2023, 10, 31, 16, 0, 0),
                "url": f"https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/{ticker.lower()}-20231031.htm",
                "raw_path": f"/tmp/fixtures/{ticker.lower()}_10k.txt",
                "text_content": f"This is the 10-K filing text for {ticker}. Apple Inc. designs consumer electronics. The CEO is Tim Cook. Apple Inc. is regulated by the SEC. Apple faces market risks and supply chain constraints."
            }
        ]

class FakeNewsSource(NewsSource):
    def fetch_news(self, query: str, date_from: datetime.date, date_to: datetime.date) -> List[Dict[str, Any]]:
        """Returns mock news articles containing structured claims."""
        return [
            {
                "source": "news",
                "external_id": f"news-art-10023-{query}",
                "title": f"Regulatory investigations targeting {query}",
                "company": query,
                "doc_type": "news",
                "published_at": datetime.datetime(2023, 11, 15, 9, 30, 0),
                "url": f"https://www.financialnews.com/articles/regulatory-{query.lower()}",
                "raw_path": f"/tmp/fixtures/news_{query.lower()}.txt",
                "text_content": f"According to regulators, {query} is under inspection. The SEC is checking compliance issues at {query}."
            }
        ]

class FakeEmbeddingModel(EmbeddingModel):
    def embed_text(self, text: str) -> List[float]:
        """Returns a deterministic list of floats of size 384."""
        val = (sum(ord(c) for c in text) % 100) / 100.0
        return [val] * 384

class FakeLLMClient(LLMClient):
    def complete(self, prompt: str) -> str:
        """Returns a mock completion response."""
        return f"Fake response for prompt containing: {prompt[:30]}..."

    def structured_call(self, schema: Dict[str, Any], messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Returns mock schema-valid entities and relationships for graph extraction."""
        # Deprecated: use extract_entities and extract_relationships instead
        return {"entities": [], "relationships": []}

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Returns deterministic entities using keywords and a steering example."""
        # Single training example required by PLAN.md
        steering_example = [{"prompt": "Apple Inc is based in Cupertino.", "completion": '[{"name": "Apple Inc.", "type": "Company", "attributes": {}}]'}]
        
        text_lower = text.lower()
        entities = []
        if "apple" in text_lower or "aapl" in text_lower or "iphone" in text_lower or "cook" in text_lower:
            entities.extend([
                {"name": "Apple Inc.", "type": "Company", "attributes": {"ticker": "AAPL"}},
                {"name": "Tim Cook", "type": "Person", "attributes": {"role": "CEO"}},
                {"name": "SEC", "type": "Regulator", "attributes": {}}
            ])
        if "tesla" in text_lower or "elon" in text_lower or "musk" in text_lower:
            entities.extend([
                {"name": "Tesla Inc.", "type": "Company", "attributes": {"ticker": "TSLA"}},
                {"name": "Elon Musk", "type": "Person", "attributes": {"role": "CEO"}},
                {"name": "SEC", "type": "Regulator", "attributes": {}}
            ])
        if not entities:
            entities.append({"name": "Generic Entity", "type": "Company", "attributes": {}})
        return entities

    def extract_relationships(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Returns deterministic relationships based on extracted entities and a steering example."""
        # Single training example required by PLAN.md
        steering_example = [{"prompt": "Tim Cook is CEO of Apple Inc.", "completion": '[{"type": "HAS_CEO", "from": "Apple Inc.", "to": "Tim Cook", "confidence": 0.9}]'}]
        
        rels = []
        entity_names = [e["name"].lower() for e in entities]
        if "apple inc." in entity_names and "tim cook" in entity_names:
            rels.append({"type": "HAS_CEO", "from": "Apple Inc.", "to": "Tim Cook", "confidence": 0.9})
        if "apple inc." in entity_names and "sec" in entity_names:
            rels.append({"type": "REGULATED_BY", "from": "Apple Inc.", "to": "SEC", "confidence": 0.95})
        if "tesla inc." in entity_names and "elon musk" in entity_names:
            rels.append({"type": "HAS_CEO", "from": "Tesla Inc.", "to": "Elon Musk", "confidence": 0.9})
        if "tesla inc." in entity_names and "sec" in entity_names:
            rels.append({"type": "REGULATED_BY", "from": "Tesla Inc.", "to": "SEC", "confidence": 0.8})
        return rels

    def generate_and_verify_claims(self, question: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Returns deterministic claims for testing. Will return unsupported claims if chunks are insufficient."""
        claims = []
        q_lower = question.lower()
        combined_text = " ".join([c.get("text", "").lower() for c in chunks])
        
        # Determine some mock claims based on the question
        if "apple" in q_lower:
            # Claim 1: Apple designs consumer electronics
            is_supported_1 = "consumer electronics" in combined_text
            chunk_ids_1 = [c["constituent_chunk_ids"][0] for c in chunks if "consumer electronics" in c.get("text", "").lower()] if is_supported_1 else []
            claims.append({
                "claim": "Apple Inc. designs consumer electronics.",
                "supported": is_supported_1,
                "evidence_chunk_ids": chunk_ids_1
            })
            
            # Claim 2: Apple makes cars (Unsupported claim to trigger retry logic)
            claims.append({
                "claim": "Apple Inc. manufactures electric cars.",
                "supported": False,
                "evidence_chunk_ids": []
            })
            
        if "tesla" in q_lower or "tsla" in q_lower:
            is_supported = "inspection" in combined_text
            chunk_ids = [c["constituent_chunk_ids"][0] for c in chunks if "inspection" in c.get("text", "").lower()] if is_supported else []
            claims.append({
                "claim": "Tesla is under inspection by regulators.",
                "supported": is_supported,
                "evidence_chunk_ids": chunk_ids
            })
            
        if not claims:
            # Fallback claim
            claims.append({
                "claim": "The documents provide general corporate information.",
                "supported": True,
                "evidence_chunk_ids": chunks[0]["constituent_chunk_ids"] if chunks else []
            })
            
        return claims

    def generate_response(self, question: str, verified_claims: List[Dict[str, Any]]) -> str:
        """Deterministic response generation from supported claims."""
        if not verified_claims:
            return "I could not find sufficient evidence to answer the question."
            
        response_parts = []
        for claim in verified_claims:
            if not claim.get("supported"):
                continue
            
            # Create inline citations like [1], [2]
            cites = claim.get("evidence_chunk_ids", [])
            cite_str = "".join([f"[{cid}]" for cid in cites])
            
            response_parts.append(f"{claim['claim']} {cite_str}")
            
        return " ".join(response_parts)


class SentenceTransformerEmbeddingModel(EmbeddingModel):
    """Real embedding model using sentence-transformers all-MiniLM-L6-v2 (local, CPU, $0)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> List[float]:
        """Returns a 384-dimensional embedding vector."""
        return self._model.encode(text).tolist()


def get_embedding_model() -> EmbeddingModel:
    """Factory: returns FakeEmbeddingModel by default, real model when EMBEDDING_MODEL='real'."""
    from settings import settings
    if settings.EMBEDDING_MODEL == "real":
        return SentenceTransformerEmbeddingModel()
    return FakeEmbeddingModel()
