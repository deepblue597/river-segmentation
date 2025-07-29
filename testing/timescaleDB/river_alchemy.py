from sqlalchemy import (
    create_engine,
    Column,
    String,
    Float,
    Boolean,
    MetaData,
    Table,
    text,
)
from sqlalchemy.dialects.postgresql import TIMESTAMP

engine = create_engine("postgresql+psycopg2://postgres:password@localhost:5432/river")
metadata = MetaData()

river_segmentation = Table(
    "river_segmentation",
    metadata,
    Column("timestamp", TIMESTAMP, primary_key=True),
    Column("model_name", String),
    Column("filename", String),
    Column("water_coverage", Float),
    Column("avg_confidence", Float),
    Column("overflow_detected", Boolean),
    Column("location", String),
)

metadata.create_all(engine)

with engine.connect() as connection:
    connection.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb;"))
    connection.execute(
        text("SELECT create_hypertable('river_segmentation', 'timestamp');")
    )


# Insert
# with engine.connect() as conn:
#     mock_rows = [generate_mock_row() for _ in range(5)]
#     insert_stmt = insert(river_segmentation_results).values(mock_rows)
#     conn.execute(insert_stmt)
#     conn.commit()

# Select
# with engine.connect() as conn:
#     select_stmt = select(river_segmentation_results)
#     results = conn.execute(select_stmt).fetchall()
#     for row in results:
#         print(row)
