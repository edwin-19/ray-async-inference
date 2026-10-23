
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    Index,
    Text,
    UniqueConstraint,
)
from sqlmodel import Field, SQLModel


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobStage(str, Enum):
    MAIN = "main"
    VAD = "vad"
    ASR = "asr"
    ALIGN = "align"
    COMPLETED = "completed"


class Job(SQLModel, table=True):
    __tablename__ = "jobs"

    id: UUID = Field(primary_key=True)
    audio_path: str = Field(nullable=False)

    status: JobStatus = Field(default=JobStatus.QUEUED, index=True)
    stage: JobStage = Field(default=JobStage.MAIN)

    duration_seconds: Optional[float] = Field(default=None)
    transcript: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True),
    )
    language: Optional[str] = Field(default=None)
    error: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True),
    )

    created_at: datetime = Field(
        default_factory=utc_now,
        sa_type=DateTime(timezone=True),
        index=True,
    )
    started_at: Optional[datetime] = Field(
        default=None,
        sa_type=DateTime(timezone=True),
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        sa_type=DateTime(timezone=True),
    )
    updated_at: datetime = Field(
        default_factory=utc_now,
        sa_type=DateTime(timezone=True),
    )


class TranscriptSegment(SQLModel, table=True):
    __tablename__ = "transcript_segments"

    __table_args__ = (
        UniqueConstraint(
            "job_id",
            "segment_index",
            name="uq_transcript_segment_job_index",
        ),
        CheckConstraint(
            "segment_index >= 0",
            name="ck_segment_index_nonnegative",
        ),
        CheckConstraint(
            "start_seconds >= 0 AND end_seconds >= start_seconds",
            name="ck_segment_timing_valid",
        ),
        Index(
            "ix_transcript_segments_job_index",
            "job_id",
            "segment_index",
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)

    job_id: UUID = Field(
        foreign_key="jobs.id",
        nullable=False,
        ondelete="CASCADE",
    )
    segment_index: int = Field(nullable=False)

    start_seconds: float = Field(nullable=False)
    end_seconds: float = Field(nullable=False)

    text: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True),
    )
    language: Optional[str] = Field(default=None)

    created_at: datetime = Field(
        default_factory=utc_now,
        sa_type=DateTime(timezone=True),
    )
    alignment_status: str = Field(
        default="pending",
        nullable=False,
        index=True,
    )



class TranscriptWord(SQLModel, table=True):
    __tablename__ = "transcript_words"

    __table_args__ = (
        UniqueConstraint(
            "segment_id",
            "word_index",
            name="uq_transcript_words_segment_index",
        ),
        CheckConstraint(
            "word_index >= 0",
            name="ck_transcript_words_index_nonnegative",
        ),
        CheckConstraint(
            "start_seconds >= 0 AND end_seconds >= start_seconds",
            name="ck_transcript_words_timing_valid",
        ),
        Index(
            "ix_transcript_words_job_segment",
            "job_id",
            "segment_id",
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)

    job_id: UUID = Field(
        foreign_key="jobs.id",
        nullable=False,
        ondelete="CASCADE",
    )

    segment_id: int = Field(
        foreign_key="transcript_segments.id",
        nullable=False,
        ondelete="CASCADE",
    )

    word_index: int = Field(nullable=False)
    text: str = Field(sa_column=Column(Text, nullable=False))
    start_seconds: float = Field(nullable=False)
    end_seconds: float = Field(nullable=False)

    created_at: datetime = Field(
        default_factory=utc_now,
        sa_type=DateTime(timezone=True),
    )
