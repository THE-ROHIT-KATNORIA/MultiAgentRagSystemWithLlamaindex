"""
Database Connection Pool
========================

Provides a psycopg connection pool shared across the application.
"""

import logging
from psycopg_pool import ConnectionPool
from config.settings import (
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB,
    POSTGRES_USER, POSTGRES_PASSWORD,
)

log = logging.getLogger(__name__)

_conninfo = (
    f"host={POSTGRES_HOST} port={POSTGRES_PORT} "
    f"dbname={POSTGRES_DB} user={POSTGRES_USER} "
    f"password={POSTGRES_PASSWORD}"
)

pool = ConnectionPool(conninfo=_conninfo, min_size=2, max_size=10)


def get_db_connection():
    """Return a pooled connection (use as a context manager)."""
    return pool.connection()
