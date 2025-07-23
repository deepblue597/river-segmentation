
from sqlalchemy import MetaData, Table, create_engine
from connectors import Connector


class TimescaleConnector(Connector):
    
    def __init__(
        self, 
        address, 
        port, 
        target,
        username=None,
        password=None,):
        super().__init__(address, port, target)
        self.username = username
        self.password = password
        self.metadata = MetaData()
    
    def connect(self):
        # Implementation for connecting to TimescaleDB
        print("Connecting to TimescaleDB...")
        try:
            # Add connection logic here
            self.engine = create_engine(
                f'postgresql://{self.username}:{self.password}@{self.address}:{self.port}/{self.target}'
            )
            
            with self.engine.connect() as connection:
                result = connection.execute("SELECT 1")
                if result.fetchone() is not None:
                    print("Connected to TimescaleDB:", self.target)
            
            
        except Exception as e:
            print(f"Failed to connect to TimescaleDB: {str(e)}")
            raise e
    
    def disconnect(self):
        # Implementation for disconnecting from TimescaleDB
        print("Disconnecting from TimescaleDB...")
        try:
            self.engine.dispose()
            print("Disconnected from TimescaleDB.")
        except Exception as e:
            print(f"Failed to disconnect from TimescaleDB: {str(e)}")
            raise e
        
    def is_connected(self):
        # Check if the connection is active
        try:
            with self.engine.connect() as connection:
                return True
        except Exception:
            return False
        
    def get_connection_info(self):
        # Return connection information
        return {
            'address': self.address,
            'port': self.port,
            'target': self.target
        } 
        
    def insert_data(self, table_name, data):
        # Insert data into a TimescaleDB table
        try:
            with self.engine.connect() as connection:
                table_name_connection = Table(
                    table_name, 
                    MetaData(), 
                    autoload_with=self.engine
                )
                
                insert_stmt = table_name_connection.insert().values(data)
                result = connection.execute(insert_stmt)
                connection.commit()
                print(f'result: {result}')
                print(f"Data inserted into {table_name} successfully.")
        except Exception as e:
            print(f"Failed to insert data into {table_name}: {str(e)}")
            raise e
        
    