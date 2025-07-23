from abc import ABC, abstractmethod

class Connector(ABC):
    
    def __init__(self, address , port , target):
        self.address = address
        self.port = port
        self.target = target
        
    
    @abstractmethod
    def connect(self):
        pass
    
    @abstractmethod
    def disconnect(self):
        pass
    
    @abstractmethod
    def is_connected(self):
        pass
    
    @abstractmethod
    def get_connection_info(self):
        pass
    
    
