import psycopg2

conn = psycopg2.connect(
    dbname="test",
    user="postgres",
    password="password",
    host="linux-pc",
    port="5432"
)

with conn:
    cursor = conn.cursor()
