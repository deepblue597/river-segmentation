
from models.model import RiverSegmentationModel


model = RiverSegmentationModel(
    model_path="best_model.pth.tar",
    model_name="unetplusplus_efficientnet-b3",
    overflow_threshold=80,
)

model.load_model()

model.minIOConnection(
    address='linux-pc', 
    port=9000, 
    access_key='minio',
    secret_key='minio123',
    region_name='eu-west-1'    
)

model.timescaleConnection(
    address='linux-pc', 
    port=5432,
    database='river',
    username='postgres',
    password='password',
    table_name='river_seg'
)

model.kafkaConnection(
    address='linux-pc',
    port=39092,
    target='River',
    consumer_group= 'model-prediction-04',
    auto_offset_reset='earliest',
    security_protocol='plaintext',
)

model.__str__()

if __name__ == "__main__":
    # Start the model prediction application
    model.start_streaming()   