from sqlalchemy import create_engine, text


engine = create_engine(
    "postgresql+psycopg2://postgres:password@localhost:5432/postgres"
)

with engine.connect() as connection:
    # Set autocommit mode for database creation
    connection.execute(text("COMMIT"))
    connection.execute(text("CREATE DATABASE river"))
    print("Database 'river' created successfully!")
