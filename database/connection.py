"""
Database Connection Pool
========================

Provides a psycopg connection pool shared across the application.
"""

import logging
from psycopg_pool import ConnectionPool
from config.settings import (
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB,
    POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_SSLMODE,
)

log = logging.getLogger(__name__)

_conninfo = (
    f"host={POSTGRES_HOST} port={POSTGRES_PORT} "
    f"dbname={POSTGRES_DB} user={POSTGRES_USER} "
    f"password={POSTGRES_PASSWORD} "
    f"sslmode={POSTGRES_SSLMODE}"
)

pool = ConnectionPool(
    conninfo=_conninfo,
    min_size=1,
    max_size=10,
    max_idle=300,
    check=ConnectionPool.check_connection,
)


def get_db_connection():
    """Return a pooled connection (use as a context manager)."""
    return pool.connection()
