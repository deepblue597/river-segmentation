# %%
import json
import pandas as pd
from river import datasets, preprocessing
#import kagglehub
from river.datasets import synth
import random
from producer import create_kafka_producer, delivery_callback
from server_funcs import parse_command_line_arguments
import os
import base64
# %%

def convert_keys_to_underscores(data):
    """
    Recursively converts all dictionary keys by replacing spaces with underscores.
    """
    if isinstance(data, dict):
        return {
            key.replace(" ", "_").lower(): convert_keys_to_underscores(value)
            for key, value in data.items()
        }
    elif isinstance(data, list):
        return [convert_keys_to_underscores(item) for item in data]
    else:
        return data

if __name__ == "__main__":

    args = parse_command_line_arguments()

    # init producer - using PLAINTEXT (no authentication)
    producer = create_kafka_producer(
        bootstrap_server=args.bootstrap_server, 
        acks='all', 
        compression_type='snappy'
    )
# %%
    # path = kagglehub.dataset_download("fedesoriano/the-boston-houseprice-data")
    dataset = datasets.Phishing()
    #dataset = datasets.TrumpApproval()
    #dataset = datasets.ImageSegments()
    #dataset = datasets.CreditCard()
    # print("Path to dataset files:", path)
    #gen = synth.Agrawal(classification_function=0, seed=42)
    # gen = synth.ConceptDriftStream(stream=synth.SEA(seed=42, variant=0),
    #                                drift_stream=synth.SEA(seed=42, variant=1),
    #                                seed=1, position=500, width=50)
    #dataset = iter(gen.take(1000))
    # dataset = synth.ConceptDriftStream(
    #    seed=42,
    #    position=500,
    #    width=40).take(1000)
    #dataset = datasets.Bananas().take(500)
    # dataset = [
    #     [1, 2],
    #     [1, 4],
    #     [1, 0],
    #     [-4, 2],
    #     [-4, 4],
    #     [-4, 0],
    #     [5, 0],
    #     [5, 2],
    #     [5, 4]
    # ]
    #dataset = datasets.AirlinePassengers()
    
    #rng = random.Random(12345)
    #dataset = rng.choices([0, 1], k=1000) + rng.choices(range(4, 8), k=1000)
    #dataset = synth.Logical(seed=42, n_tiles=100)
#     dataset = [
#     ("Chinese Beijing Chinese", "yes"),
#     ("Chinese Chinese Shanghai", "yes"),
#     ("Chinese Macao", "yes"),
#     ("Tokyo Japan Chinese", "no")
# ]
    # dataset = (
    #     ({'user': 'Alice', 'item': 'Superman', 'time': .12}, True),
    #     ({'user': 'Alice', 'item': 'Terminator', 'time': .13}, True),
    #     ({'user': 'Alice', 'item': 'Star Wars', 'time': .14}, True),
    #     ({'user': 'Alice', 'item': 'Notting Hill', 'time': .15}, False),
    #     ({'user': 'Alice', 'item': 'Harry Potter ', 'time': .16}, True),
    #     ({'user': 'Bob', 'item': 'Superman', 'time': .13}, True),
    #     ({'user': 'Bob', 'item': 'Terminator', 'time': .12}, True),
    #     ({'user': 'Bob', 'item': 'Star Wars', 'time': .16}, True),
    #     ({'user': 'Bob', 'item': 'Notting Hill', 'time': .10}, False)
    # )
    #path = kagglehub.dataset_download("fedesoriano/heart-failure-prediction")
    #path = kagglehub.dataset_download("unsdsn/world-happiness")
    #print("Path to dataset files:", path)
    
    #dataset = datasets.Elec2()
    #path = kagglehub.dataset_download("vstacknocopyright/electricity")
# %%
    #path

# %%
    # csv_path = os.path.join(path, "boston.csv")
    # Trump approval , Airline , Phising
    df = pd.read_csv(dataset.path)
    #df = pd.read_csv(path+'/2019.csv')
    #df = pd.read_csv(path+'/electricity-normalized.csv')
    # Bananas, Clustering , CreditCard
    #df = pd.DataFrame(dataset)
# %%
    df
    
#%% 

# for idx, row in df.iterrows():
#     json_message = row.to_json()
#     #sample_dict = { str(idx): int(row[0])}
#     #json_message = json.dumps({str(idx): row[0]})
#     #sample_dict = {**row[0].isoformat(), 'passengers': row[1]}
#     #sample_dictimport kagglehub

# Download latest version
#path = kagglehub.dataset_download("fedesoriano/heart-failure-prediction")

#print("Path to dataset files:", path) = {'data': **row[0], 'class': row[1]}
#     sample_dict = {'data' : json_message}
#     print(sample_dict)
# %%
    print('Messages are being published to Kafka topic')
    messages_count = 0
    
    folder_path = './images'  # change to your folder
    

    #for idx, row in df.iterrows():

        #drift 
        #sample_dict = { str(idx): int(row[0])}
        # bananas , CreditCard , facto , Agrawal
        #sample_dict = {'x' : {**row[0]}, 'class': row[1]}
        # convert to json format
        #TRump , airline, List dataset 
        #json_message = row.to_json()
        #sample_dict = {'data': json.loads(json_message)}
        #Chinese Beijing
        #sample_dict = {'value' : row[0], 'class': row[1]}
        #json_message = convert_keys_to_underscores(json_message)
        #json_message = json.dumps({str(idx): int(row[0])})
        #json_message = json.dumps(sample_dict)
        #print(json_message)
        #row_dict = row.to_dict()
        #row_dict_converted = convert_keys_to_underscores(row_dict)
        #json_message = json.dumps(row_dict_converted)
        # Produce the message to kafka
        # producer.produce(
        #     args.topic_name, value=json_message, key=str(idx), callback=delivery_callback)

        # # Polling to handle responses
        # producer.poll(0)

        # messages_count += 1
    for image_filename  in os.listdir(folder_path):
        file_path = os.path.join(folder_path, image_filename)  # join folder + filename
        with open(file_path, 'rb') as image_file:
            image_data = image_file.read()
            
            image_base_64 = base64.b64encode(image_data).decode('utf-8')
            
        message = {
            'filename':image_filename,
            'image': image_base_64,
            'date'  : '2023-10-01', 
        }
        
        producer.produce(
            args.topic_name, 
            value=json.dumps(message), 
            key=str(messages_count), 
            callback=delivery_callback
        )
        messages_count += 1

    # Flush to ensure all messages are sent before exit
    producer.flush()