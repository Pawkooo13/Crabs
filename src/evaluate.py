import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import config as cfg
import tensorflow
from prepare import DataProcessor
import json


def eval_scores(y_true, y_pred):
    """
    Print the evaluation scores for the model
    Parameters:
        y_true (1d array): true labels array
        y_pred (1d array): predicted values array

    Returns:
        0
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    metrics = {'MSE': mse,
               'RMSE': rmse,
               'MAE': mae,
               'R2': r2}

    with open(cfg.METRICS_PATH, 'w') as outfile:
        json.dump(metrics, outfile)


if __name__ == '__main__':
    # load test data from file
    df = pd.read_csv(cfg.TEST_PATH)

    model = tensorflow.keras.saving.load_model(cfg.MODELS_PATH)

    DP = DataProcessor()
    test = DP.process(df=df,
                      train=False)

    Y_test = test['Age']
    X_test = test.drop(columns=['Age'],
                       axis=1)

    Y_pred = model.predict(X_test)

    eval_scores(y_true=Y_test,
                y_pred=Y_pred)
