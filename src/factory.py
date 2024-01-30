from abc import ABC, abstractmethod
import math
from keras.models import Sequential
from keras.layers import Dense, Input
from dvc.api import params_show

class ANN(ABC):
    """Defines methods to create model"""
    @abstractmethod
    def build(self, n_layers, units, activation):
        """build sequential model with given parameters"""
        pass

class FullyConnectedANN(ANN):
    """Fully connected ANN"""
    def __init__(self):
        self.model = Sequential()

    def build(self, n_layers, units, activation):
        """build FC architecture with given parameters"""
        self.model.add(Input(shape=(10,)))

        for _ in range(n_layers):
            self.model.add(Dense(units=units,
                                 activation=activation))
            
        self.model.add(Dense(units=1,
                             activation='linear'))
        return self.model
    
class PyramidalANN(ANN):
    """Pyramidal architecture of ANN"""
    def __init__(self):
        self.model = Sequential()

    def build(self, n_layers, units, activation):
        """build pyramidal architecture with given parameters"""
        self.model.add(Input(shape=(10,)))

        while(units != 1):
            for _ in range(n_layers):
                self.model.add(Dense(units=units,
                                    activation=activation))

            units = math.floor(math.sqrt(units))

        self.model.add(Dense(units=1,
                             activation='linear'))
        return self.model
    
class ANNFactory(ABC):
    """Defines the factory method to create model"""
    @abstractmethod
    def initialize_model(self):
        """Create a ANN instance"""
        pass

class FullyConnectedANNFactory(ANNFactory):
    """Creates FullyConnectedANN instances"""
    def initialize_model(self):
        """factory method implementation for creating FullyConnectedANN"""
        return FullyConnectedANN()
    
class PyramidalANNFactory(ANNFactory):
    """Creates PyramidalANN instances"""
    def initialize_model(self):
        """factory method implementation for creating PyramidalANN"""
        return PyramidalANN()
    
def create_model_initializator(kind='FC'):
    models = {'FC': FullyConnectedANN(),
              'P': PyramidalANN()}
    return models[kind]

def main():
    params = params_show()

    kind = params['model']['kind']
    n_layers = params['model']['n_layers']
    units = params['model']['units']
    activation = params['model']['activation']

    loss = params['train']['loss']
    lr = params['train']['lr']
    epochs = params['train']['epochs']


    model_init = create_model_initializator(kind)
    model = model_init.build(n_layers=n_layers,
                             units=units,
                             activation=activation)
    
    print(model.summary())

if __name__ == '__main__':
    main()