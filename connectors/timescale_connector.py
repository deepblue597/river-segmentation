
from sqlalchemy import MetaData, Table, create_engine, text
from connectors import Connector


class TimescaleConnector(Connector):
    """
    A connector for TimescaleDB that extends the base Connector class.
    It provides methods to connect, disconnect, check connection status, and retrieve connection information.
    
    Attributes:
        address (str): The TimescaleDB server address.
        port (int): The port number for the TimescaleDB server.
        target (str): The TimescaleDB database name.
        username (str, optional): The username for authentication.
        password (str, optional): The password for authentication.
    
    Methods:
        connect(): Connects to the TimescaleDB server.
        disconnect(): Disconnects from the TimescaleDB server.
        is_connected(): Checks if the connection to the TimescaleDB server is active.
        get_connection_info(): Returns a dictionary with connection information.
        insert_data(table_name, data): Inserts data into a TimescaleDB table.
        
    """
    
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
                result = connection.execute(text("SELECT 1"))
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
                connection.execute(text("SELECT 1"))
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
                # Load table metadata
                table_obj = Table(
                    table_name, 
                    self.metadata, 
                    autoload_with=self.engine
                )
                
                insert_stmt = table_obj.insert().values(data)
                result = connection.execute(insert_stmt)
                connection.commit()
                print(f"Data inserted into {table_name} successfully.")
        except Exception as e:
            print(f"Failed to insert data into {table_name}: {str(e)}")
            raise e
        
    