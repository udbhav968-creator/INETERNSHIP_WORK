# Production PostgreSQL Migration & Concurrency Architecture Plan

> **Author**: DevOps Intern 2  
> **Target**: Klassed NEET Adaptive Engine Backend  
> **Status**: Ready for Review / RFC  
> **Target Production Database**: PostgreSQL 16 (AWS RDS / Supabase / Neon)  
> **Target Concurrency**: 2,500+ writes/second, p99 submission latency < 15ms  

---

## 1. Executive Summary & Problem Statement

The Klassed NEET Adaptive Engine currently persists relational state to a local SQLite database (`backend/klassed.db`). While SQLite is effective for development and deterministic standalone verification, it uses file-level locking for write transactions (`BEGIN IMMEDIATE / EXCLUSIVE`). 

During live NEET mock exams:
1. **Concurrent Attempt Submissions**: Up to 1,000+ students concurrently submit question attempts, update BKT mastery probabilities, and refresh LinUCB bandit feature states.
2. **Lock Contention**: Concurrent write requests trigger `sqlite3.OperationalError: database is locked`, forcing exponential backoff retries and degrading API response times.
3. **Connection Pooling Incompatibility**: SQLite does not support client connection pooling across serverless or containerized replicas (e.g. multiple Uvicorn worker processes or Vercel edge handlers).

This document outlines the end-to-end migration plan to PostgreSQL 16 with PgBouncer connection pooling, Alembic migration scripts, and a zero-downtime cutover strategy.

---

## 2. PostgreSQL 16 Target Schema Definition

### 2.1 Complete DDL Specification

```sql
-- PostgreSQL 16 Production Schema for Klassed Engine

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- 1. Topics Table
CREATE TABLE IF NOT EXISTS topics (
    topic_id VARCHAR(64) PRIMARY KEY,
    subject VARCHAR(32) NOT NULL,
    name VARCHAR(128) NOT NULL,
    class_level INTEGER NOT NULL CHECK (class_level IN (11, 12)),
    importance_weight NUMERIC(6, 4) NOT NULL DEFAULT 1.5000,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_topics_subject ON topics (subject);

-- 2. Questions Table
CREATE TABLE IF NOT EXISTS questions (
    question_id VARCHAR(64) PRIMARY KEY,
    topic_id VARCHAR(64) NOT NULL REFERENCES topics(topic_id) ON DELETE RESTRICT,
    subject VARCHAR(32) NOT NULL,
    difficulty VARCHAR(16) NOT NULL CHECK (difficulty IN ('basic', 'medium', 'advanced')),
    question_text TEXT NOT NULL,
    option_a TEXT NOT NULL,
    option_b TEXT NOT NULL,
    option_c TEXT NOT NULL,
    option_d TEXT NOT NULL,
    correct_option CHAR(1) NOT NULL CHECK (correct_option IN ('a', 'b', 'c', 'd')),
    explanation TEXT NOT NULL DEFAULT '',
    ideal_time_sec INTEGER NOT NULL DEFAULT 90,
    has_diagram BOOLEAN NOT NULL DEFAULT FALSE,
    paper_source VARCHAR(64) NOT NULL DEFAULT 'Official NEET PYQ',
    irt_a NUMERIC(6, 4) NOT NULL DEFAULT 1.0000,
    irt_b NUMERIC(6, 4) NOT NULL DEFAULT 0.0000,
    irt_c NUMERIC(6, 4) NOT NULL DEFAULT 0.2000,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_questions_topic_diff ON questions (topic_id, difficulty);
CREATE INDEX idx_questions_source ON questions (paper_source);

-- 3. Students Table
CREATE TABLE IF NOT EXISTS students (
    student_id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(128) NOT NULL,
    email VARCHAR(128) UNIQUE NOT NULL,
    password_hash VARCHAR(256) NOT NULL,
    global_theta NUMERIC(6, 4) NOT NULL DEFAULT 0.0000,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_students_email ON students (email);

-- 4. Student Topic State Table
CREATE TABLE IF NOT EXISTS student_topic_state (
    student_id VARCHAR(64) NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    topic_id VARCHAR(64) NOT NULL REFERENCES topics(topic_id) ON DELETE CASCADE,
    level VARCHAR(16) NOT NULL DEFAULT 'basic' CHECK (level IN ('basic', 'medium', 'advanced')),
    streak INTEGER NOT NULL DEFAULT 0,
    last_3_results JSONB NOT NULL DEFAULT '[]'::jsonb,
    avg_time_ratio NUMERIC(6, 4) NOT NULL DEFAULT 1.0000,
    pace_flag VARCHAR(16) DEFAULT NULL CHECK (pace_flag IN ('fast', 'slow', NULL)),
    mastery_prob NUMERIC(6, 4) NOT NULL DEFAULT 0.3500,
    irt_theta NUMERIC(6, 4) NOT NULL DEFAULT 0.0000,
    confidence_interval JSONB NOT NULL DEFAULT '[-1.96, 1.96]'::jsonb,
    last_practiced TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (student_id, topic_id)
);

CREATE INDEX idx_sts_student_mastery ON student_topic_state (student_id, mastery_prob ASC);
CREATE INDEX idx_sts_student_level ON student_topic_state (student_id, level);

-- 5. Test Attempts Table
CREATE TABLE IF NOT EXISTS test_attempts (
    attempt_id VARCHAR(64) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    student_id VARCHAR(64) NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    question_id VARCHAR(64) NOT NULL REFERENCES questions(question_id) ON DELETE RESTRICT,
    topic_id VARCHAR(64) NOT NULL REFERENCES topics(topic_id) ON DELETE RESTRICT,
    paper_number INTEGER NOT NULL DEFAULT 1,
    is_correct BOOLEAN NOT NULL,
    time_taken_seconds NUMERIC(8, 2) NOT NULL,
    selected_option CHAR(1) NOT NULL,
    difficulty VARCHAR(16) NOT NULL,
    hesitation_ms NUMERIC(10, 2) NOT NULL DEFAULT 0.0,
    option_switches INTEGER NOT NULL DEFAULT 0,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_attempts_student_time ON test_attempts (student_id, timestamp DESC);
CREATE INDEX idx_attempts_question_correct ON test_attempts (question_id, is_correct);
CREATE INDEX idx_attempts_topic_time ON test_attempts (topic_id, timestamp DESC);
```

---

## 3. Connection Pooling & PgBouncer Configuration

FastAPI with SQLAlchemy uses asynchronous or threaded sessions. Under multi-worker deployments (e.g. 4 Uvicorn workers per container $\times$ 3 replicas = 12 processes), each establishing up to 20 connections would overwhelm PostgreSQL's maximum connection ceiling (default 100).

### 3.1 PgBouncer Settings (`pgbouncer.ini`)

```ini
[databases]
klassed_prod = host=127.0.0.1 port=5432 dbname=klassed_prod auth_user=pgbouncer

[pgbouncer]
logfile = /var/log/postgresql/pgbouncer.log
pidfile = /var/run/postgresql/pgbouncer.pid
listen_addr = *
listen_port = 6432
auth_type = scram-sha-256
auth_file = /etc/pgbouncer/userlist.txt

; Transaction pooling mode provides optimal reuse for short REST queries
pool_mode = transaction

; Connection limits
max_client_conn = 1000
default_pool_size = 30
min_pool_size = 10
reserve_pool_size = 5
reserve_res_sec = 2
max_db_connections = 80

; Timeouts
server_idle_timeout = 600
client_idle_timeout = 120
query_timeout = 10
```

### 3.2 SQLAlchemy 2.0 Engine Configuration

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

POSTGRES_DATABASE_URL = (
    "postgresql+psycopg2://klassed_app:Secr3tPass@127.0.0.1:6432/klassed_prod"
)

engine = create_engine(
    POSTGRES_DATABASE_URL,
    pool_size=20,
    max_overflow=10,
    pool_timeout=15,
    pool_recycle=1800,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
```

---

## 4. Alembic Migration Workflow

### 4.1 Migration Directory Structure

```
backend/
├── alembic.ini
├── alembic/
│   ├── env.py
│   ├── script.py.mako
│   └── versions/
│       ├── 0001_initial_schema.py
│       └── 0002_add_telemetry_columns.py
```

### 4.2 Initial Migration Script (`0001_initial_schema.py`)

Alembic automatically translates SQLAlchemy declarative models into PostgreSQL DDL:
```bash
alembic revision --autogenerate -m "create_initial_klassed_tables"
alembic upgrade head
```

---

## 5. Zero-Downtime Cutover Strategy

To ensure zero exam disruption during production cutover, we employ a 4-phase rollout:

```mermaid
flowchart LR
    A["Phase 1: Dual-Write (Async worker writes to both SQLite & Postgres)"] --> B["Phase 2: Data Backfill & SHA-256 Verification"]
    B --> C["Phase 3: Read Cutover (FastAPI reads Postgres; writes remain dual)"]
    C --> D["Phase 4: Full Cutover (Writes exclusive to Postgres; SQLite archived)"]
```

### Phase 1: Dual-Write Setup
- The primary database remains SQLite.
- Every write transaction in `main.py` dispatches an asynchronous background task that duplicates the write to PostgreSQL.
- Write failures to PostgreSQL are logged to an error queue (`sentry` or dead-letter table) without interrupting the student's request.

### Phase 2: Historical Data Backfill & Integrity Check
- A batch synchronization script reads all rows from `backend/klassed.db` and upserts them to PostgreSQL.
- Checksums and row counts across all 5 tables must match 100%:
  $$\text{Count}(\text{SQLite}) = \text{Count}(\text{PostgreSQL})$$

### Phase 3: Read Cutover
- FastAPI switches all `SELECT` queries to PostgreSQL via PgBouncer.
- Writes continue to both engines.
- Latency and error rates are monitored for 48 hours.

### Phase 4: Full Cutover & SQLite Deprecation
- Write path to SQLite is decommissioned.
- SQLite file is sealed with read-only permissions and archived to S3.
- PostgreSQL becomes the sole system of record.

---

## 6. Disaster Recovery & Rollback SLA

- **Recovery Point Objective (RPO)**: $\le 30\text{ seconds}$ via continuous Write-Ahead Log (WAL) archiving to cloud storage.
- **Recovery Time Objective (RTO)**: $\le 5\text{ minutes}$ via automated multi-AZ standby failover.
- **Rollback Procedure**: If PostgreSQL p99 latency exceeds 50ms during Phase 3, read traffic instantly flips back to SQLite using the `USE_POSTGRES=false` feature flag without downtime.
