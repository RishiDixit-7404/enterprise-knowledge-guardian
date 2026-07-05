"""RetrievalAgent orchestrating the 4-arm hybrid search and reranking."""

import uuid
import datetime
from typing import List, Dict, Any, Optional

from sqlalchemy.orm import Session
from sentence_transformers import CrossEncoder

from db.repositories import ChunkRepository
from db.graph import Graph
from ingest.interfaces import get_embedding_model
from retrieval.fusion import reciprocal_rank_fusion

class RetrievalAgent:
    def __init__(self, session: Session, graph: Graph):
        self.session = session
        self.graph = graph
        self.chunk_repo = ChunkRepository(session)
        self.embedding_model = get_embedding_model()
        
        # Lazy load reranker to avoid blocking instantiation if not needed
        self._reranker = None

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
        return self._reranker

    def retrieve(
        self, 
        question: str, 
        top_k: int = 5,
        company: Optional[str] = None,
        doc_type: Optional[str] = None,
        date_from: Optional[datetime.datetime] = None,
        date_to: Optional[datetime.datetime] = None
    ) -> Dict[str, Any]:
        """
        Executes the 4-arm retrieval, fusion, reranking, and section expansion pipeline.
        Returns a dict containing 'expanded_chunks' and 'retrieval_trace'.
        """
        trace = {
            "question": question,
            "filters": {
                "company": company,
                "doc_type": doc_type,
                "date_from": date_from.isoformat() if date_from else None,
                "date_to": date_to.isoformat() if date_to else None
            },
            "stages": {}
        }
        
        # 1. Dense Retrieval
        query_embedding = self.embedding_model.embed_text(question)
        dense_results = self.chunk_repo.search(
            query_embedding, top_k=20, 
            company=company, doc_type=doc_type, date_from=date_from, date_to=date_to
        )
        trace["stages"]["dense"] = [str(c.id) for c, _ in dense_results]

        # 2. BM25 Retrieval
        bm25_results = self.chunk_repo.search_bm25(
            question, top_k=20,
            company=company, doc_type=doc_type, date_from=date_from, date_to=date_to
        )
        trace["stages"]["bm25"] = [str(c.id) for c, _ in bm25_results]

        # 3. Graph Retrieval
        graph_chunk_ids = self.graph.get_evidence_chunks_for_text(question)
        graph_chunks = self.chunk_repo.get_by_ids(graph_chunk_ids)
        # Apply metadata filters to graph results manually since they were fetched by ID
        filtered_graph_chunks = []
        for c in graph_chunks:
            meta = c.chunk_metadata or {}
            if company and meta.get("company") != company:
                continue
            if doc_type and meta.get("doc_type") != doc_type:
                continue
            if date_from or date_to:
                c_date = meta.get("date")
                if not c_date:
                    continue
                try:
                    dt = datetime.datetime.fromisoformat(c_date)
                    if date_from and dt < date_from:
                        continue
                    if date_to and dt > date_to:
                        continue
                except ValueError:
                    pass
            filtered_graph_chunks.append((c, 1.0))
        
        trace["stages"]["graph"] = [str(c.id) for c, _ in filtered_graph_chunks]

        # 4. Fusion
        fused_results = reciprocal_rank_fusion([dense_results, bm25_results, filtered_graph_chunks], k=60)
        
        # Dedup items
        fused_candidates = [chunk for chunk, _ in fused_results]
        trace["stages"]["fusion"] = [str(c.id) for c in fused_candidates]

        # 5. Reranking
        if not fused_candidates:
            return {"expanded_chunks": [], "retrieval_trace": trace}
            
        rerank_inputs = [[question, c.text] for c in fused_candidates]
        scores = self.reranker.predict(rerank_inputs)
        
        # Sort by reranker score descending
        # Secondary sort by chunk.id to guarantee deterministic ordering
        reranked_pairs = sorted(
            zip(fused_candidates, scores), 
            key=lambda x: (x[1], str(x[0].id)), 
            reverse=True
        )
        
        top_reranked = reranked_pairs[:top_k]
        trace["stages"]["reranker"] = [
            {"chunk_id": str(c.id), "score": float(s)} 
            for c, s in top_reranked
        ]

        # 6. Section Expansion
        # For each top chunk, retrieve all chunks in its parent section.
        expanded_chunks_output = []
        seen_sections = set()
        
        for chunk, score in top_reranked:
            section_key = (chunk.document_id, chunk.chunk_metadata.get("section", ""))
            if section_key in seen_sections:
                continue
                
            seen_sections.add(section_key)
            
            section_chunks = self.chunk_repo.get_section_chunks(chunk.document_id, section_key[1])
            if not section_chunks:
                # Fallback if somehow not found
                section_chunks = [chunk]
                
            full_text = " ".join([c.text for c in section_chunks])
            
            expanded_chunks_output.append({
                "document_id": str(chunk.document_id),
                "section": section_key[1],
                "text": full_text,
                "metadata": chunk.chunk_metadata,
                "reranker_score": float(score),
                "constituent_chunk_ids": [str(c.id) for c in section_chunks]
            })

        return {
            "expanded_chunks": expanded_chunks_output,
            "retrieval_trace": trace
        }
