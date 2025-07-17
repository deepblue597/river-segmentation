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
    
    # Method 1: Insert single record
    cursor.execute("""
        INSERT INTO stock_data (date, open, high, low, close, volume)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, ('2025-07-17', 0.0601, 0.0608, 0.0595, 0.0605, 1500000))
    
    # Method 2: Insert multiple records at once (more efficient)
    stock_records = [
        ('2025-07-16', 0.0595, 0.0602, 0.0590, 0.0598, 1200000),
        ('2025-07-15', 0.0590, 0.0598, 0.0585, 0.0595, 1100000),
        ('2025-07-14', 0.0585, 0.0595, 0.0580, 0.0590, 1300000),
    ]
    
    cursor.executemany("""
        INSERT INTO stock_data (date, open, high, low, close, volume)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, stock_records)
    
    # Method 3: Insert with ON CONFLICT (upsert)
    cursor.execute("""
        INSERT INTO stock_data (date, open, high, low, close, volume)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (date) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume
    """, ('2025-07-17', 0.0601, 0.0610, 0.0595, 0.0607, 1600000))
    
    # Commit the changes
    conn.commit()
    print("✅ Data inserted successfully!")
    
    # Verify the data
    cursor.execute("SELECT * FROM stock_data ORDER BY date DESC LIMIT 5")
    results = cursor.fetchall()
    print("\n📊 Latest 5 records:")
    for row in results:
        print(f"Date: {row[0]}, Close: {row[4]}, Volume: {row[5]}")

