
from river_segmentation.connectors import Connector
from confluent_kafka import Producer


class KafkaProducerConnector(Connector): 
    """A connector for Kafka Producer that extends the base Connector class.
    It provides methods to connect, disconnect, check connection status, and retrieve connection information.
    
    Attributes:
        address (str): The Kafka broker address.
        port (int): The port number for the Kafka broker.
        topic (str): The Kafka topic to produce messages to.
        security_protocol (str): The security protocol to use, default is 'plaintext'.
        username (str, optional): The username for authentication, if required.
        password (str, optional): The password for authentication, if required.  
        sasl_mechanism (str, optional): The SASL mechanism to use, if required.
        acks (str): The acknowledgment level for message delivery, default is 'all'.
        batch_size (int): The size of the batch for message production, default is 163
        linger_ms (int): The time to wait before sending a batch, default is 0.
        compression_type (str): The compression type for messages, default is 'snappy'.
        message_max_bytes (int): The maximum size of a message, default is 50MB.      
    
    Methods:
        connect(): Connects to the Kafka broker and initializes the producer.
        disconnect(): Disconnects from the Kafka broker.
        is_connected(): Checks if the connection to the Kafka broker is active.
        get_connection_info(): Returns a dictionary with connection information.
        produce(key, value): Produces a message to the Kafka topic.
        delivery_callback(err, msg): Callback function for message delivery status.
    """
    
    def __init__(
        self, 
        address, 
        port, 
        topic, 
        security_protocol='PLAINTEXT', 
        username=None, 
        password=None, 
        sasl_mechanism=None,
        acks='all' ,
        batch_size=16384,
        linger_ms=0,
        compression_type='snappy',
        message_max_bytes=52428800
            
    ):
        super().__init__(address, port, topic)
        self.security_protocol = security_protocol
        self.username = username
        self.password = password
        self.acks = acks
        self.batch_size = batch_size
        self.linger_ms = linger_ms
        self.compression_type = compression_type
        self.message_max_bytes = message_max_bytes
        self.sasl_mechanism = sasl_mechanism
    
    
    def connect(self):
        # Implementation for connecting to Kafka Producer
        print("Connecting to Kafka Producer...")
        
        config = {
                'bootstrap.servers': f'{self.address}:{self.port}',
                'security.protocol': self.security_protocol,
                'acks': self.acks,
                'batch.size': self.batch_size,
                'linger.ms': self.linger_ms,
                'compression.type': self.compression_type,
                'message.max.bytes': self.message_max_bytes
            }
        if self.sasl_mechanism is not None:
            config.update({
                'security.protocol': 'SASL_PLAINTEXT',
                'sasl.mechanisms': 'PLAIN',
                'sasl.username': self.username,
                'sasl.password': self.password
        })
            
        try:
            
            self.producer = Producer(config)
            print(f"Connected to Kafka Producer: {self.address}:{self.port}")
        
        except Exception as e:
            print(f"Failed to connect to Kafka Producer: {str(e)}")
            raise e
    
    def disconnect(self):
        # Implementation for disconnecting from Kafka Producer
        print("Disconnecting from Kafka Producer...")
        try:
            if hasattr(self, 'producer') and self.producer is not None:
                self.producer.flush()
                self.producer = None
            print("Disconnected from Kafka Producer.")
        except Exception as e:
            print(f"Failed to disconnect from Kafka Producer: {str(e)}")
            raise e
        
    def is_connected(self):
        # Check if the connection is active
        try:
            if self.producer is None:
                return False
            # Try a simple operation to verify connection
            self.producer.list_topics()
            return True
        except Exception:
            return False
    
    def get_connection_info(self):
        # Return connection information
        return {
            'address': self.address,
            'port': self.port,
            'topic': self.target,
            'security_protocol': self.security_protocol,
            'username': self.username,
            'sasl_mechanism': self.sasl_mechanism,
            'acks': self.acks,
            'batch_size': self.batch_size,
            'linger_ms': self.linger_ms,
            'compression_type': self.compression_type,
            'message_max_bytes': self.message_max_bytes
        }
    
    def delivery_callback(self, err, msg):
        # executed when a record is successfully sent or an exception is thrown
        if err:
            print(f'ERROR: Message failed delivery: {err}')
        else:
            print(
                f"Produced event to topic {msg.topic()}: partition {msg.partition()}: value = {msg.value().decode('utf-8')}")
            pass
        
    
    def produce(self, key, value):
        """Produce a message to the Kafka topic.
        
        Args:
            key (str): The key for the message.
            value (str): The value for the message.
        """
        try:
            self.producer.produce(
                topic=self.target,
                key=key,
                value=value,
                callback=self.delivery_callback
            )
            self.producer.poll(0)  # Poll to trigger delivery callback
            self.producer.flush()

        except Exception as e:
            print(f"Failed to produce message: {str(e)}")
    
    