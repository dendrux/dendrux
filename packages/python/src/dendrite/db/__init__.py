"""Database layer — state persistence for agent runs.

Provides SQLAlchemy models, session management, and Alembic migrations
for persisting runs, traces, tool calls, and token usage.

Optional dependency: install with `pip install dendrite[db]`.
"""
