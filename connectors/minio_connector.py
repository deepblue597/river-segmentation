
from connectors import Connector
import boto3

class MinIOConnector(Connector):
    
    def __init__(
        self, 
        address, 
        port, 
        target, 
        access_key=None, 
        secret_key=None, 
        region_name='eu-west-1'
        ):
        super().__init__(address, port, target)
        self.access_key = access_key
        self.secret_key = secret_key
        self.region_name = region_name
        
    def connect(self):
        # Implementation for connecting to MinIO
        print("Connecting to MinIO...") 
        try:

            
            # Add connection logic here
            self.s3_client = boto3.client(
                's3',
                endpoint_url=f'http://{self.address}:{self.port}',
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name=self.region_name
            )
            
            print(f"Connected to MinIO: {self.address}:{self.port}" )
        
        except Exception as e:
            print(f"Failed to connect to MinIO: {str(e)}")
            raise e    
                
    
    def disconnect(self):
        # Implementation for disconnecting from MinIO
        print("Disconnecting from MinIO...")
        self.s3_client = None
        print("Disconnected from MinIO.")
        
    def is_connected(self):
        # Check if the connection is active
        return self.s3_client is not None
    
    def get_connection_info(self):
        # Return connection information
        return {
            'address': self.address,
            'port': self.port,
            'target': self.target,
            'region_name': self.region_name
        }
    
    def insert_object(self, object_name, data , content_type='image/png'):
        # Insert an object into the MinIO bucket
        try:
            self.s3_client.put_object(
                Bucket=self.target,
                Key=object_name,
                Body=data, 
                ContentType=content_type
            )
            print(f"Object '{object_name}' inserted into bucket '{self.target}'.")
        except Exception as e:
            print(f"Failed to insert object: {str(e)}")
            raise e
    
    def get_object(self, object_name):
        # Retrieve an object from the MinIO bucket
        try:
            response = self.s3_client.get_object(
                Bucket=self.target,
                Key=object_name
            )
            return response['Body'].read()
        except Exception as e:
            print(f"Failed to get object: {str(e)}")
            raise e