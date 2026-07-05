from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from settings import settings

engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_session():
    """Dependency for obtaining a database session."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

def init_db():
    """Ensures pgvector extension is enabled, all database tables are created,
    and search indexes exist."""
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()
    Base.metadata.create_all(bind=engine)
    # Create search indexes if they don't already exist
    with engine.connect() as conn:
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_chunks_embedding_hnsw "
            "ON chunks USING hnsw (embedding vector_cosine_ops);"
        ))
        conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_chunks_tsv_gin "
            "ON chunks USING gin (tsv);"
        ))
        conn.commit()
