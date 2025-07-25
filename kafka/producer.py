import json
import argparse

from confluent_kafka import Producer
from sseclient import SSEClient as EventSource
# from kafka import RoundRobinPartitioner


# https://docs.confluent.io/platform/current/clients/producer.html
# https://docs.confluent.io/platform/current/installation/configuration/producer-configs.html
def create_kafka_producer(bootstrap_server, acks, linger_ms=0, batch_size=16 * 1024, compression_type=None, use_sasl=False, sasl_username=None, sasl_password=None):
    config = {
        # User-specific properties that you must set
        'bootstrap.servers': bootstrap_server,
        # 'partitioner': RoundRobinPartitioner,
        # TODO: find out how to change number of partitions -> through kafka admin
        'acks': acks,  # 0 1 all|-1
        # 'value.serializer': lambda x: json.dumps(x).encode('utf-8')
        'batch.size': batch_size,  # default 16Kb
        # ' delivery.timeout.ms' : 120000 ( default 2 mins )
        # 'enable.idempotence': True (default)
        'linger.ms': linger_ms,  # Wait up to x ms for the batch to fill before sending default 0
        'compression.type': compression_type,  # None ( default )
        'message.max.bytes': 52428800,      # 50MB
        
        
    }
    
    # Add SASL configuration if required
    if use_sasl:
        config.update({
            'security.protocol': 'SASL_PLAINTEXT',
            'sasl.mechanisms': 'PLAIN',
            'sasl.username': sasl_username,
            'sasl.password': sasl_password
        })

    # not working
    # partitioner = RoundRobinPartitioner(partitions=3)  # Assume we have 3 partitions in the topic

    # Create Producer instance
    producer = Producer(config)
    return producer


def construct_id(event_data):
    event_data = {'id': event_data['id']}

    return json.dumps({"id": event_data['id']}).encode('utf-8')


def delivery_callback(err, msg):
    # executed when a record is successfully sent or an exception is thrown
    if err:
        print(f'ERROR: Message failed delivery: {err}')
    else:
        print(
            f"Produced event to topic {msg.topic()}: partition {msg.partition()}: value = {msg.value().decode('utf-8')}")
