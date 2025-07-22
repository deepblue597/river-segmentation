from quixstreams import Application

from predict_single_image import predict_and_save



app = Application(
    broker_address='linux-pc:39092',
    consumer_group='model-prediction-04', 
    auto_offset_reset='earliest',
    # Add debug logging
    #loglevel='DEBUG'
)
topic = app.topic('River')

sdf = app.dataframe(topic=topic)

sdf = sdf.apply(predict_and_save)

if __name__ == "__main__":
    app.run()
    