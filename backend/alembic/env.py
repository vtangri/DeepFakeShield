"""
Alembic migrations environment file.
"""
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# Import Base and all models for autogenerate
from app.db.base import Base
from app.models import *  # noqa: Import all models

# this is the Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

from app.core.config import settings
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

# Model's MetaData object for 'autogenerate' support
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        def process_revision_directives(context, revision, directives):
            if config.get_main_option("sqlalchemy.url").startswith("sqlite"):
                for directive in directives:
                    for op in directive.upgrade_ops.ops:
                        if hasattr(op, 'type_') and str(op.type_) == 'UUID':
                             op.type_ = sa.CHAR(32)

        context.configure(
            connection=connection, 
            target_metadata=target_metadata,
            render_as_batch=True,
            # Handle SQLite UUID limitation
            user_module_prefix="sa.",
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
