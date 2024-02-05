import config as cfg
import pandas as pd
from sklearn.model_selection import train_test_split
from dvc.api import params_show
import numpy as np
import os


class DataProcessor:
    @staticmethod
    def change_units(df):
        """change units to meters and kilograms"""
        df['Length'] = df['Length'] * 0.3048  # m
        df['Diameter'] = df['Diameter'] * 0.3048  # m
        df['Height'] = df['Height'] * 0.3048  # m
        df['Weight'] = df['Weight'] * 0.02835  # kg
        df['Shucked Weight'] = df['Shucked Weight'] * 0.02835  # kg
        df['Viscera Weight'] = df['Viscera Weight'] * 0.02835  # kg
        df['Shell Weight'] = df['Shell Weight'] * 0.02835  # kg
        return df

    @staticmethod
    def remove_outliers(df):
        """remove outliers from column Height after changing units"""
        return df.query('`Height` > 0 and `Height` <= 0.25')

    @staticmethod
    def check_sum_of_weights(df):
        """
        check if sum of weights features are lower than whole crab`s weight
        """
        return df.query('`Weight` > `Shucked Weight` + `Viscera Weight` + `Shell Weight`')

    @staticmethod
    def change_weight_features_to_ratio(df):
        """change values in columns connected with weight to ratio"""
        df['Shell Weight'] = df['Shell Weight'] / df['Weight']
        df['Shucked Weight'] = df['Shucked Weight'] / df['Weight']
        df['Viscera Weight'] = df['Viscera Weight'] / df['Weight']
        return df

    # noinspection PyMethodMayBeStatic
    def process(self, df, train=True):
        """
        run data pipeline and return processed dataframe with one hot encoded Sex column
        """
        processed_df = (df.pipe(DataProcessor.change_units)
                        .pipe(DataProcessor.check_sum_of_weights)
                        .pipe(DataProcessor.change_weight_features_to_ratio))
        if train:
            processed_df = processed_df.pipe(DataProcessor.remove_outliers)

        processed_df = pd.get_dummies(processed_df, dtype=np.float32)  # to avoid boolean
        return processed_df


def main():
    data = pd.read_csv(cfg.DATA_PATH, delimiter=',')
    params = params_show()
    print(data.head())

    test_size = params['test_size']

    DP = DataProcessor()
    processed_data = DP.process(data)

    train, valid = train_test_split(processed_data, test_size=test_size, random_state=42)

    train.to_csv(os.path.join(cfg.DATA_DIR, 'processed_train.csv'), index=False)
    valid.to_csv(os.path.join(cfg.DATA_DIR, 'processed_valid.csv'), index=False)


if __name__ == '__main__':
    main()
