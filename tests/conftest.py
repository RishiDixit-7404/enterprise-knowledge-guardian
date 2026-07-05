import pytest
from fastapi.testclient import TestClient
from api.main import app

@pytest.fixture(scope="function")
def client():
    """Provides a TestClient for testing the FastAPI application endpoints."""
    from api.main import verify_token
    app.dependency_overrides[verify_token] = lambda: "test-token"
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()

@pytest.fixture(scope="module")
def setup_db_and_graph():
    """Initializes DB schemas for tests."""
    from db.session import init_db, engine, Base
    Base.metadata.drop_all(bind=engine)
    init_db()

@pytest.fixture(scope="module")
def populated_db_and_graph(setup_db_and_graph):
    """Provides a fresh populated database and graph from the ingest worker."""
    import uuid
    from db.session import get_session
    from models.database import IngestJob
    from worker.main import process_job
    from db.graph import Graph
    
    # Clean up before
    graph = Graph()
    with graph.driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    db = next(get_session())
    db.execute(pytest.importorskip("sqlalchemy").text("DELETE FROM chunks"))
    db.execute(pytest.importorskip("sqlalchemy").text("DELETE FROM documents"))
    db.execute(pytest.importorskip("sqlalchemy").text("DELETE FROM ingest_jobs"))
    db.commit()

    # Ingest Edgar mock docs (includes Apple)
    job1 = IngestJob(id=uuid.uuid4(), source="edgar", params={"tickers": ["AAPL"]}, status="Queued")
    db.add(job1)
    db.commit()
    db.refresh(job1)
    process_job(job1, db)

    # Ingest News mock docs (includes Tesla)
    job2 = IngestJob(id=uuid.uuid4(), source="news", params={"tickers": ["TSLA"]}, status="Queued")
    db.add(job2)
    db.commit()
    db.refresh(job2)
    process_job(job2, db)

    yield db, graph

    # Clean up after
    with graph.driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    db.execute(pytest.importorskip("sqlalchemy").text("DELETE FROM chunks"))
    db.execute(pytest.importorskip("sqlalchemy").text("DELETE FROM documents"))
    db.execute(pytest.importorskip("sqlalchemy").text("DELETE FROM ingest_jobs"))
    db.commit()
    graph.close()
