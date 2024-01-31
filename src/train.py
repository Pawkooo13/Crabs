from factory import create_model_initializator
from dvc.api import params_show
from keras.optimizers import Adam
import os 
import config as cfg
import pandas as pd
import numpy as np

def main():
    params = params_show()

    train = pd.read_csv(os.path.join(cfg.DATA_DIR, 'processed_train.csv'), dtype=np.float32) #to avoid boolean type
    valid = pd.read_csv(os.path.join(cfg.DATA_DIR, 'processed_valid.csv'), dtype=np.float32) #to avoid boolean type

    Y_train = train['Age']
    X_train = train.drop(columns=['Age'], 
                         axis=1)

    Y_valid = valid['Age']
    X_valid = valid.drop(columns=['Age'], 
                         axis=1)

    kind = params['model']['kind']
    n_layers = params['model']['n_layers']
    units = params['model']['units']
    activation = params['model']['activation']

    loss = params['train']['loss']
    lr = params['train']['lr']
    epochs = params['train']['epochs']
    batch_size = params['train']['batch_size']

    model_init = create_model_initializator(kind)
    model = model_init.build(n_layers=n_layers,
                             units=units,
                             activation=activation)
    
    print(model.summary())

    model.compile(loss=loss,
                  optimizer=Adam(learning_rate=lr))
    
    history = model.fit(x=X_train,
                        y=Y_train,
                        validation_data=(X_valid, Y_valid),
                        epochs=epochs,
                        batch_size=batch_size)

    model.save(os.path.join(cfg.MODELS_DIR, 'ANN.keras'))
    pd.DataFrame(history.history).to_csv(os.path.join(cfg.MODELS_DIR, 'history.csv'))

if __name__ == '__main__':
    main()