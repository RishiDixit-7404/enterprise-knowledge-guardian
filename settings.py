from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/ekg"
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"
    WORKER_POLL_INTERVAL: float = 5.0
    EMBEDDING_MODEL: str = "fake"  # 'fake' for offline deterministic, 'real' for all-MiniLM-L6-v2
    LLM_MODEL: str = "fake"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    MLFLOW_ENABLED: bool = False
    PORT: int = 8000
    HOST: str = "0.0.0.0"
    API_KEY: str | None = None

    # Pydantic v2 configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
