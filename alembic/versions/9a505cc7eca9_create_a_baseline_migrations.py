"""Create a baseline migrations

Revision ID: 9a505cc7eca9
Revises:
Create Date: 2024-10-11 14:19:19.875656

"""

from typing import Sequence, Union

import pgvector
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

import letta.metadata
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9a505cc7eca9"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "agent_source_mapping",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("agent_id", sa.String(), nullable=False),
        sa.Column("source_id", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("agent_source_mapping_idx_user", "agent_source_mapping", ["user_id", "agent_id", "source_id"], unique=False)
    op.create_table(
        "agents",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.Column("description", sa.String(), nullable=True),
        sa.Column("message_ids", sa.JSON(), nullable=True),
        sa.Column("memory", sa.JSON(), nullable=True),
        sa.Column("system", sa.String(), nullable=True),
        sa.Column("agent_type", sa.String(), nullable=True),
        sa.Column("llm_config", letta.metadata.LLMConfigColumn(), nullable=True),
        sa.Column("embedding_config", letta.metadata.EmbeddingConfigColumn(), nullable=True),
        sa.Column("metadata_", sa.JSON(), nullable=True),
        sa.Column("tools", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("agents_idx_user", "agents", ["user_id"], unique=False)
    op.create_table(
        "block",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("value", sa.String(), nullable=False),
        sa.Column("limit", sa.BIGINT(), nullable=True),
        sa.Column("name", sa.String(), nullable=True),
        sa.Column("template", sa.Boolean(), nullable=True),
        sa.Column("label", sa.String(), nullable=False),
        sa.Column("metadata_", sa.JSON(), nullable=True),
        sa.Column("description", sa.String(), nullable=True),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("block_idx_user", "block", ["user_id"], unique=False)
    op.create_table(
        "files",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("source_id", sa.String(), nullable=False),
        sa.Column("file_name", sa.String(), nullable=True),
        sa.Column("file_path", sa.String(), nullable=True),
        sa.Column("file_type", sa.String(), nullable=True),
        sa.Column("file_size", sa.Integer(), nullable=True),
        sa.Column("file_creation_date", sa.String(), nullable=True),
        sa.Column("file_last_modified_date", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "jobs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metadata_", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "messages",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("agent_id", sa.String(), nullable=False),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column("text", sa.String(), nullable=True),
        sa.Column("model", sa.String(), nullable=True),
        sa.Column("name", sa.String(), nullable=True),
        sa.Column("tool_calls", letta.metadata.ToolCallColumn(), nullable=True),
        sa.Column("tool_call_id", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("message_idx_user", "messages", ["user_id", "agent_id"], unique=False)
    op.create_table(
        "organizations",
        sa.Column("id", sa.VARCHAR(), autoincrement=False, nullable=False),
        sa.Column("name", sa.VARCHAR(), autoincrement=False, nullable=False),
        sa.Column("created_at", postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
        sa.PrimaryKeyConstraint("id", name="organizations_pkey"),
    )
    op.create_table(
        "passages",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("text", sa.String(), nullable=True),
        sa.Column("file_id", sa.String(), nullable=True),
        sa.Column("agent_id", sa.String(), nullable=True),
        sa.Column("source_id", sa.String(), nullable=True),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(dim=4096), nullable=True),
        sa.Column("embedding_config", letta.metadata.EmbeddingConfigColumn(), nullable=True),
        sa.Column("metadata_", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("passage_idx_user", "passages", ["user_id", "agent_id", "file_id"], unique=False)
    op.create_table(
        "sources",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.Column("embedding_config", letta.metadata.EmbeddingConfigColumn(), nullable=True),
        sa.Column("description", sa.String(), nullable=True),
        sa.Column("metadata_", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("sources_idx_user", "sources", ["user_id"], unique=False)
    op.create_table(
        "tokens",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("key", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("tokens_idx_key", "tokens", ["key"], unique=False)
    op.create_index("tokens_idx_user", "tokens", ["user_id"], unique=False)

    op.create_table(
        "users",
        sa.Column("id", sa.VARCHAR(), autoincrement=False, nullable=False),
        sa.Column("org_id", sa.VARCHAR(), autoincrement=False, nullable=True),
        sa.Column("name", sa.VARCHAR(), autoincrement=False, nullable=False),
        sa.Column("created_at", postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
        sa.Column("policies_accepted", sa.BOOLEAN(), autoincrement=False, nullable=False),
        sa.PrimaryKeyConstraint("id", name="users_pkey"),
    )
    op.create_table(
        "tools",
        sa.Column("id", sa.VARCHAR(), autoincrement=False, nullable=False),
        sa.Column("name", sa.VARCHAR(), autoincrement=False, nullable=False),
        sa.Column("user_id", sa.VARCHAR(), autoincrement=False, nullable=True),
        sa.Column("description", sa.VARCHAR(), autoincrement=False, nullable=True),
        sa.Column("source_type", sa.VARCHAR(), autoincrement=False, nullable=True),
        sa.Column("source_code", sa.VARCHAR(), autoincrement=False, nullable=True),
        sa.Column("json_schema", postgresql.JSON(astext_type=sa.Text()), autoincrement=False, nullable=True),
        sa.Column("module", sa.VARCHAR(), autoincrement=False, nullable=True),
        sa.Column("tags", postgresql.JSON(astext_type=sa.Text()), autoincrement=False, nullable=True),
        sa.PrimaryKeyConstraint("id", name="tools_pkey"),
    )


def downgrade() -> None:
    op.drop_table("users")
    op.drop_table("tools")
    op.drop_index("tokens_idx_user", table_name="tokens")
    op.drop_index("tokens_idx_key", table_name="tokens")
    op.drop_table("tokens")
    op.drop_index("sources_idx_user", table_name="sources")
    op.drop_table("sources")
    op.drop_index("passage_idx_user", table_name="passages")
    op.drop_table("passages")
    op.drop_table("organizations")
    op.drop_index("message_idx_user", table_name="messages")
    op.drop_table("messages")
    op.drop_table("jobs")
    op.drop_table("files")
    op.drop_index("block_idx_user", table_name="block")
    op.drop_table("block")
    op.drop_index("agents_idx_user", table_name="agents")
    op.drop_table("agents")
    op.drop_index("agent_source_mapping_idx_user", table_name="agent_source_mapping")
    op.drop_table("agent_source_mapping")
