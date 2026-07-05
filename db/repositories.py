from sqlalchemy.orm import Session
from sqlalchemy import select, func, text, DateTime
from typing import List, Optional, Tuple
import uuid
import datetime
from models.database import Document, Chunk, RunLog, EvalRecord

class DocumentRepository:
    def __init__(self, session: Session):
        self.session = session

    def add(self, document: Document) -> Document:
        """Adds a Document record to the database and commits."""
        self.session.add(document)
        self.session.commit()
        self.session.refresh(document)
        return document

    def get_by_id(self, doc_id: uuid.UUID) -> Optional[Document]:
        """Retrieves a Document by its primary UUID."""
        return self.session.execute(
            select(Document).where(Document.id == doc_id)
        ).scalar_one_or_none()

    def get_by_external_id(self, source: str, external_id: str) -> Optional[Document]:
        """Retrieves a Document by its source and external ID."""
        return self.session.execute(
            select(Document).where(Document.source == source, Document.external_id == external_id)
        ).scalar_one_or_none()


class ChunkRepository:
    def __init__(self, session: Session):
        self.session = session

    def add(self, chunk: Chunk) -> Chunk:
        """Adds a Chunk record to the database and commits."""
        self.session.add(chunk)
        self.session.commit()
        self.session.refresh(chunk)
        return chunk

    def get_by_id(self, chunk_id: uuid.UUID) -> Optional[Chunk]:
        """Retrieves a Chunk by its primary UUID."""
        return self.session.execute(
            select(Chunk).where(Chunk.id == chunk_id)
        ).scalar_one_or_none()

    def get_by_document(self, doc_id: uuid.UUID) -> List[Chunk]:
        """Retrieves all Chunks associated with a specific Document ID ordered by ordinal."""
        return list(self.session.execute(
            select(Chunk).where(Chunk.document_id == doc_id).order_by(Chunk.ordinal)
        ).scalars().all())

    def get_by_ids(self, chunk_ids: List[uuid.UUID]) -> List[Chunk]:
        """Retrieves multiple Chunks by their UUIDs."""
        if not chunk_ids:
            return []
        return list(self.session.execute(
            select(Chunk).where(Chunk.id.in_(chunk_ids))
        ).scalars().all())

    def get_section_chunks(self, document_id: uuid.UUID, section: str) -> List[Chunk]:
        """Retrieves all Chunks for a given document_id and section metadata, ordered by ordinal."""
        stmt = (
            select(Chunk)
            .where(Chunk.document_id == document_id)
            .where(Chunk.chunk_metadata["section"].astext == section)
            .order_by(Chunk.ordinal)
        )
        return list(self.session.execute(stmt).scalars().all())

    def _apply_metadata_filters(
        self, stmt, *,
        company: Optional[str] = None,
        doc_type: Optional[str] = None,
        date_from: Optional[datetime.datetime] = None,
        date_to: Optional[datetime.datetime] = None,
    ):
        """Applies metadata filters (via the JSONB chunk_metadata column) to a query statement."""
        if company is not None:
            stmt = stmt.where(
                Chunk.chunk_metadata["company"].astext == company
            )
        if doc_type is not None:
            stmt = stmt.where(
                Chunk.chunk_metadata["doc_type"].astext == doc_type
            )
        if date_from is not None:
            stmt = stmt.where(
                func.cast(Chunk.chunk_metadata["date"].astext, DateTime) >= date_from
            )
        if date_to is not None:
            stmt = stmt.where(
                func.cast(Chunk.chunk_metadata["date"].astext, DateTime) <= date_to
            )
        return stmt

    def search(
        self, query_embedding: List[float], top_k: int = 10, *,
        company: Optional[str] = None,
        doc_type: Optional[str] = None,
        date_from: Optional[datetime.datetime] = None,
        date_to: Optional[datetime.datetime] = None,
    ) -> List[Tuple[Chunk, float]]:
        """Dense retrieval using pgvector cosine distance (<->).
        Returns list of (Chunk, similarity_score) tuples ordered by similarity descending.
        """
        distance = Chunk.embedding.cosine_distance(query_embedding).label("distance")
        stmt = (
            select(Chunk, distance)
            .where(Chunk.embedding.isnot(None))
            .order_by(distance.asc())
            .limit(top_k)
        )
        stmt = self._apply_metadata_filters(
            stmt, company=company, doc_type=doc_type,
            date_from=date_from, date_to=date_to,
        )
        rows = self.session.execute(stmt).all()
        # Convert cosine distance to similarity: similarity = 1 - distance
        return [(row[0], 1.0 - row[1]) for row in rows]

    def search_bm25(
        self, query_text: str, top_k: int = 10, *,
        company: Optional[str] = None,
        doc_type: Optional[str] = None,
        date_from: Optional[datetime.datetime] = None,
        date_to: Optional[datetime.datetime] = None,
    ) -> List[Tuple[Chunk, float]]:
        """Full-text search using PostgreSQL tsvector/tsquery with ts_rank.
        Returns list of (Chunk, rank_score) tuples ordered by rank descending.
        """
        tsquery = func.plainto_tsquery("english", query_text)
        rank = func.ts_rank(Chunk.tsv, tsquery).label("rank")
        stmt = (
            select(Chunk, rank)
            .where(Chunk.tsv.op("@@")(tsquery))
            .order_by(rank.desc())
            .limit(top_k)
        )
        stmt = self._apply_metadata_filters(
            stmt, company=company, doc_type=doc_type,
            date_from=date_from, date_to=date_to,
        )
        rows = self.session.execute(stmt).all()
        return [(row[0], row[1]) for row in rows]



class RunLogRepository:
    def __init__(self, session: Session):
        self.session = session

    def add(self, run_log: RunLog) -> RunLog:
        """Adds a RunLog record to the database and commits."""
        self.session.add(run_log)
        self.session.commit()
        self.session.refresh(run_log)
        return run_log

    def get_by_id(self, log_id: uuid.UUID) -> Optional[RunLog]:
        """Retrieves a RunLog by its primary UUID."""
        return self.session.execute(
            select(RunLog).where(RunLog.id == log_id)
        ).scalar_one_or_none()


class EvalRecordRepository:
    def __init__(self, session: Session):
        self.session = session

    def add(self, record: EvalRecord) -> EvalRecord:
        """Adds an EvalRecord to the database and commits."""
        self.session.add(record)
        self.session.commit()
        self.session.refresh(record)
        return record

    def add_batch(self, records: List[EvalRecord]) -> List[EvalRecord]:
        """Adds multiple EvalRecords in a single transaction."""
        for r in records:
            self.session.add(r)
        self.session.commit()
        for r in records:
            self.session.refresh(r)
        return records

    def get_by_eval_run_id(self, eval_run_id: uuid.UUID) -> List[EvalRecord]:
        """Retrieves all EvalRecords from a single evaluation run."""
        return list(self.session.execute(
            select(EvalRecord)
            .where(EvalRecord.eval_run_id == eval_run_id)
            .order_by(EvalRecord.created_at.desc())
        ).scalars().all())

    def get_recent(self, limit: int = 50) -> List[EvalRecord]:
        """Retrieves the most recent EvalRecords."""
        return list(self.session.execute(
            select(EvalRecord)
            .order_by(EvalRecord.created_at.desc())
            .limit(limit)
        ).scalars().all())

    def get_aggregated_metrics(self, dataset_version: Optional[str] = None) -> dict:
        """Computes average scores per metric across recent evaluation runs.

        Returns a dict of {metric_name: {"avg": float, "count": int, "min": float, "max": float}}.
        """
        stmt = select(
            EvalRecord.metric,
            func.avg(EvalRecord.score).label("avg_score"),
            func.count(EvalRecord.id).label("count"),
            func.min(EvalRecord.score).label("min_score"),
            func.max(EvalRecord.score).label("max_score"),
        ).where(EvalRecord.score.isnot(None)).group_by(EvalRecord.metric)

        if dataset_version:
            stmt = stmt.where(EvalRecord.dataset_version == dataset_version)

        rows = self.session.execute(stmt).all()
        result = {}
        for row in rows:
            result[row[0]] = {
                "avg": float(row[1]) if row[1] is not None else None,
                "count": int(row[2]),
                "min": float(row[3]) if row[3] is not None else None,
                "max": float(row[4]) if row[4] is not None else None,
            }
        return result

    def get_recent_eval_runs(self, limit: int = 10) -> List[dict]:
        """Returns summaries of the most recent evaluation runs."""
        # Get distinct eval_run_ids ordered by most recent
        subq = (
            select(
                EvalRecord.eval_run_id,
                func.max(EvalRecord.created_at).label("run_date"),
                func.max(EvalRecord.dataset_version).label("dataset_version"),
            )
            .group_by(EvalRecord.eval_run_id)
            .order_by(func.max(EvalRecord.created_at).desc())
            .limit(limit)
        )
        runs = self.session.execute(subq).all()

        result = []
        for run_row in runs:
            run_id = run_row[0]
            # Get metrics for this run
            metrics_stmt = (
                select(EvalRecord.metric, EvalRecord.score, EvalRecord.error_reason)
                .where(EvalRecord.eval_run_id == run_id)
            )
            metrics = self.session.execute(metrics_stmt).all()
            metrics_dict = {}
            for m in metrics:
                if m[1] is not None:
                    metrics_dict[m[0]] = float(m[1])
                elif m[2] is not None:
                    metrics_dict[m[0]] = {"error": m[2]}
                else:
                    metrics_dict[m[0]] = None

            result.append({
                "eval_run_id": str(run_id),
                "run_date": run_row[1].isoformat() if run_row[1] else None,
                "dataset_version": run_row[2],
                "metrics": metrics_dict,
            })
        return result
