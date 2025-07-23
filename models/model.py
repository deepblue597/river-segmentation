from datetime import datetime
import cv2
import numpy as np
import torch
import os
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

from connectors.kafka_connector import KafkaConnector
from connectors.minio_connector import MinIOConnector
from connectors.timescale_connector import TimescaleConnector
import base64


class RiverSegmentationModel: 
    """
    A class for river segmentation model that handles connections to Kafka, MinIO, and TimescaleDB.
    It provides methods for loading the model, making predictions, and saving results to the respective services
    
    Attributes:
        model_path (str): Path to the trained model file.
        model_name (str): Name of the model.
        input_size (tuple): Input size for the model.
        mean (tuple): Mean values for normalization.
        std (tuple): Standard deviation values for normalization.
        max_pixel_value (float): Maximum pixel value for normalization.
        overflow_threshold (float): Threshold for overflow detection.
        
    Methods:
        load_model(): Loads the trained model from the specified path.
        get_prediction_transform(): Returns the transformation to be applied to input images before prediction.
        minIOConnection(address, port, target, access_key, secret_key): Returns a MinIO connector instance.
        timescaleConnection(address, port, target, username, password): Returns a TimescaleDB connector instance.
        kafkaConnection(address, port, topic, consumer_group, auto_offset_reset='earliest', security_protocol='plaintext', username=None, password=None): Returns a Kafka connector instance.
        predict(X): Makes a prediction on the input data and saves results to MinIO and TimescaleDB.
        start_streaming(): Starts the Kafka streaming application and applies the prediction function to each message.
        __str__(): Returns a string representation of the model.
        
    """
    
    def __init__(
        self, 
        model_path="best_model.pth.tar",
        model_name="RiverSegmentationModel",
        input_size=(512, 512),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225), 
        max_pixel_value=255.0, 
        overflow_threshold=80,
        model_architecture="unetplusplus",
        encoder_name="efficientnet-b3",
        encoder_weights="imagenet"
        
        ):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.model_name = model_name
        self.mean = mean
        self.std = std
        self.max_pixel_value = max_pixel_value
        self.transform = self.get_prediction_transform()
        self.overflow_threshold = overflow_threshold
        self.model_architecture = model_architecture
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        
        # Initialize the model architecture based on parameters
        self.model = self._create_model()
        
    def _create_model(self):
        """Create model based on architecture parameter"""
        if self.model_architecture.lower() == "unetplusplus":
            return smp.UnetPlusPlus(
                encoder_name=self.encoder_name,
                encoder_weights=self.encoder_weights,
                in_channels=3,
                classes=1,
            )
        elif self.model_architecture.lower() == "unet":
            return smp.Unet(
                encoder_name=self.encoder_name,
                encoder_weights=self.encoder_weights,
                in_channels=3,
                classes=1,
            )
        elif self.model_architecture.lower() == "deeplabv3plus":
            return smp.DeepLabV3Plus(
                encoder_name=self.encoder_name,
                encoder_weights=self.encoder_weights,
                in_channels=3,
                classes=1,
            )
        elif self.model_architecture.lower() == "fpn":
            return smp.FPN(
                encoder_name=self.encoder_name,
                encoder_weights=self.encoder_weights,
                in_channels=3,
                classes=1,
            )
        else:
            raise ValueError(f"Unsupported model architecture: {self.model_architecture}")
        
        
    def load_model(self):
        
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            self.model.to(self.device)
            self.model.eval()
            print(f"✓ {self.model_name} loaded successfully!")
            
        else:
            print(f"❌ Model file '{self.model_path}' not found!")
            raise FileNotFoundError(f"Model file '{self.model_path}' does not exist.")
        
    
    def get_prediction_transform(self):
        """
        Returns the transformation to be applied to input images before prediction.
        """
        return A.Compose([
            A.Resize(height=self.input_size[0], width=self.input_size[1]),
            A.Normalize(
                mean=self.mean,
                std=self.std,
                max_pixel_value=self.max_pixel_value
            ),
            ToTensorV2(),
        ])
        
    def minIOConnection(self, address, port, target, access_key, secret_key):
        """
        Returns a MinIO connector instance.
        This is a placeholder for the actual implementation.
        """
        # Assuming you have a MinIOConnector class defined elsewhere
        self.MinIOconnector = MinIOConnector(
            address=address,
            port=port,
            target=target,
            access_key=access_key,
            secret_key=secret_key
        )
        
        self.MinIOconnector.connect()
        
    
    def timescaleConnection(self, address, port, target, username, password , table_name='predictions'):
        """
        Returns a TimescaleDB connector instance.
        This is a placeholder for the actual implementation.
        """
        # Assuming you have a TimescaleDBConnector class defined elsewhere
        self.TimescaleDBconnector = TimescaleConnector(
            address=address,
            port=port,
            target=target,
            username=username,
            password=password
        )
        
        self.table = table_name
        self.TimescaleDBconnector.connect()
    
    
    def kafkaConnection(self, address, port, topic, consumer_group, auto_offset_reset='earliest', security_protocol='plaintext', username=None, password=None):
        """
        Returns a Kafka connector instance.
        This is a placeholder for the actual implementation.
        """
        # Assuming you have a KafkaConnector class defined elsewhere
        self.KafkaConnector = KafkaConnector(
            address=address,  # Replace with actual Kafka address
            port=port,           # Replace with actual Kafka port
            topic=topic,  # Replace with actual topic name    
            consumer_group=consumer_group,  # Replace with actual consumer group
            auto_offset_reset=auto_offset_reset,
            security_protocol=security_protocol,
            username=username,  # Replace with actual username if needed
            password=password   # Replace with actual password if needed
        )
        
        self.KafkaConnector.connect()
        
    
    def predict(self, X):
        
        image_data = X['image']
        image_bytes = base64.b64decode(image_data)
        
        #Decode the image bytes to a numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        
        
        if image is None:
            print("Error: Could not decode image from bytes")
            return None

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        original_image = cv2.resize(image, self.input_size)
        #Apply transformations
        transform = self.transform(image=image)
        input_tensor = transform['image'].unsqueeze(0).to(self.device)
        
        # Make prediction
        try: 
            with torch.no_grad():
                prediction = self.model(input_tensor)
                prediction_probs = torch.sigmoid(prediction)
                confidence_map = prediction_probs.cpu().squeeze().numpy()
                prediction_mask = (confidence_map > 0.5).astype(np.uint8)
                print("Prediction completed successfully.")
                
            # Calucalte Statistics
            total_pixels = prediction_mask.size
            water_pixels = np.sum(prediction_mask)
            water_percentage = (water_pixels / total_pixels) * 100
            avg_confidence = np.mean(confidence_map)
            overflow_flag = water_percentage > self.overflow_threshold
            
            print(f"Water coverage: {water_percentage:.1f}% of image")
            print(f"Average confidence: {avg_confidence:.2f}")
            
            # Connect to MinIO and save the images
            # Encode prediction mask as PNG bytes
            encoded_mask = cv2.imencode('.png', prediction_mask * 255)[1].tobytes()
            self.MinIOconnector.insert_object(
                object_name=f'predictions/{X["filename"]}',
                data=encoded_mask,
                content_type='image/png'
            )
            
            # Encode confidence map as PNG bytes
            encoded_confidence = cv2.imencode('.png', (confidence_map * 255).astype(np.uint8))[1].tobytes()
            self.MinIOconnector.insert_object(
                object_name=f'confidence_maps/{X["filename"]}',
                data=encoded_confidence,
                content_type='image/png'
            )
            
            # Create overlay image
            overlay = original_image.copy()  
            overlay[prediction_mask == 1] = [0, 255, 0]  # Green for water
            blended_image = cv2.addWeighted(original_image, 0.7, overlay, 0.3, 0)  

            encoded_image = cv2.imencode('.png', blended_image)[1].tobytes()

            self.MinIOconnector.insert_object(
                object_name=f'overlay/{X["filename"]}',
                data=encoded_image,
                content_type='image/png'
            )
            
            self.TimescaleDBconnector.insert_data(
                table_name=self.table,
                data={
                    'timestamp': datetime.now(),
                    'model_name': self.model_name,
                    'filename': X['filename'],
                    'water_coverage': float(water_percentage),
                    'avg_confidence': float(avg_confidence), 
                    'overflow_flag': overflow_flag,
                    'location': X.get('location', 'unknown'),  # Use .get() to safely access location
                }
            )
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None
        return {
            'filename': X['filename'],
            'water_coverage': water_percentage,
            'avg_confidence': avg_confidence,
            'overflow_flag': overflow_flag,
            'confidence_map': confidence_map,
            'prediction_mask': prediction_mask,
            'overlay_image': encoded_image
        }
        

    def __str__(self):
        return f"RiverSegmentationModel(name={self.model_name}, input_size={self.input_size})"
    
    
    def start_streaming(self): 
        
        self.KafkaConnector.sdf_stream = self.KafkaConnector.sdf_stream.apply(self.predict)
        # Start the streaming application
        self.KafkaConnector.app.run() 