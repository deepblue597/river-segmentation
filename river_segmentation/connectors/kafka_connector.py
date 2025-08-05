from river_segmentation.connectors import Connector
from quixstreams import Application
from quixstreams.kafka import ConnectionConfig


class KafkaConnector(Connector):
    """A connector for Kafka that extends the base Connector class.
    It provides methods to connect, disconnect, check connection status, and retrieve connection information.

    Attributes:
        address (str): The Kafka broker address.
        port (int): The port number for the Kafka broker.
        topic (str): The Kafka topic to connect to.
        consumer_group (str): The consumer group for the Kafka connection.
        auto_offset_reset (str): The offset reset policy, default is 'earliest'.
        security_protocol (str): The security protocol to use, default is 'plaintext'.
        username (str, optional): The username for authentication, if required.
        password (str, optional): The password for authentication, if required.

    Methods:
        connect(): Connects to the Kafka broker and initializes the application and topic.
        disconnect(): Disconnects from the Kafka broker.
        is_connected(): Checks if the connection to the Kafka broker is active.
        get_connection_info(): Returns a dictionary with connection information.
    """

    def __init__(
        self,
        address,
        port,
        target,
        consumer_group,
        auto_offset_reset="earliest",
        security_protocol="plaintext",
        username=None,
        password=None,
    ):
        super().__init__(address, port, target)
        self.consumer_group = consumer_group
        self.auto_offset_reset = auto_offset_reset
        self.security_protocol = security_protocol
        self.username = username
        self.password = password

    def connect(self):
        # Implementation for connecting to Kafka
        print("Connecting to Kafka...")
        try:
            # Add connection logic here
            self.connectionConfig = ConnectionConfig(
                bootstrap_servers=f"{self.address}:{self.port}",
                security_protocol=self.security_protocol,
                sasl_username=self.username,
                sasl_password=self.password,
            )

            self.app = Application(
                broker_address=self.connectionConfig,
                consumer_group=self.consumer_group,
                auto_offset_reset=self.auto_offset_reset,
            )

            self.topic_obj = self.app.topic(self.target, value_deserializer="json")
            self.sdf_stream = self.app.dataframe(topic=self.topic_obj)
            print("Connected to Kafka topic:", self.target)
        except Exception as e:
            print(f"Failed to connect to Kafka: {str(e)}")
            raise e

    def disconnect(self):
        # Implementation for disconnecting from Kafka
        print("Disconnecting from Kafka...")
        try:
            # Add disconnection logic here
            if hasattr(self, "app") and self.app is not None:
                self.app.stop()
            self.app = None
            self.topic_obj = None
            self.sdf_stream = None
            print("Disconnected from Kafka.")
        except Exception as e:
            print(f"Failed to disconnect from Kafka: {str(e)}")
            raise e

    def is_connected(self):
        # Check if the connection is active
        return (
            hasattr(self, "app")
            and self.app is not None
            and hasattr(self, "topic_obj")
            and self.topic_obj is not None
        )

    def get_connection_info(self):
        # Return connection information
        return {
            "address": self.address,
            "port": self.port,
            "topic": self.target,
            "consumer_group": self.consumer_group,
            "auto_offset_reset": self.auto_offset_reset,
            "security_protocol": self.security_protocol,
            "username": self.username,
        }
