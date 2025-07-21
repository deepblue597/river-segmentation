from sqlalchemy import create_engine , Column , Integer , String , Float , Boolean , MetaData , Table, text
from sqlalchemy.dialects.postgresql import TIMESTAMP
from mock_data import generate_mock_row
from sqlalchemy import insert , select

engine = create_engine('postgresql+psycopg2://postgres:password@linux-pc:5432/river')
metadata = MetaData()

river_segmentation_results = Table(
    'river_segmentation_results', 
    metadata,
    Column('timestamp', TIMESTAMP, primary_key=True),
    Column('image_name', String, primary_key=True), 
    Column('water_coverage_pct', Float),
    Column('avg_confidence', Float),
    Column('iou_score', Float),
    Column('dice_score', Float),
    Column('overflow_detected', Boolean),
    Column('processing_time_ms', Integer),
    Column('model_version', String),
    Column('location', String)
)

# metadata.create_all(engine)

# with engine.connect() as connection:
    
#     connection.execute(text('CREATE EXTENSION IF NOT EXISTS timescaledb;'))
#     connection.execute(text("SELECT create_hypertable('river_segmentation_results', 'timestamp');"))
    

# Insert
with engine.connect() as conn:
    mock_rows = [generate_mock_row() for _ in range(5)]
    insert_stmt = insert(river_segmentation_results).values(mock_rows)
    conn.execute(insert_stmt)
    conn.commit()

#Select
# with engine.connect() as conn:
#     select_stmt = select(river_segmentation_results)
#     results = conn.execute(select_stmt).fetchall()
#     for row in results:
#         print(row)