
import os
from collections.abc import Iterator
from contextlib import contextmanager
from urllib.parse import quote_plus

from dotenv import load_dotenv
from sqlmodel import Session, SQLModel, create_engine

from db.models import Job, TranscriptSegment, TranscriptWord  # Register tables

load_dotenv()

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "127.0.0.1")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "inference")
POSTGRES_USER = os.getenv("POSTGRES_USER", "appuser")
POSTGRES_PASSWORD = os.environ["POSTGRES_PASSWORD"]

DATABASE_URL = (
    f"postgresql+psycopg://{quote_plus(POSTGRES_USER)}:"
    f"{quote_plus(POSTGRES_PASSWORD)}@"
    f"{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
    max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "10")),
    pool_recycle=1800,
)


@contextmanager
def get_session() -> Iterator[Session]:
    with Session(engine) as session:
        yield session


def create_tables() -> None:
    SQLModel.metadata.create_all(engine)
