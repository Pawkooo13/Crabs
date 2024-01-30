import config as cfg
import pandas as pd
from sklearn.model_selection import train_test_split
import os

class DataProcessor:
    @staticmethod
    def change_units(df):   
        '''change units to meters and kilograms'''
        df['Length'] = df['Length'] * 0.3048 #m
        df['Diameter'] = df['Diameter'] * 0.3048 #m
        df['Height'] = df['Height'] * 0.3048 #m
        df['Weight'] = df['Weight'] * 0.02835 #kg
        df['Shucked Weight'] = df['Shucked Weight'] * 0.02835 #kg
        df['Viscera Weight'] = df['Viscera Weight'] * 0.02835 #kg
        df['Shell Weight'] = df['Shell Weight'] * 0.02835 #kg
        return df

    @staticmethod
    def remove_outliers(df):
        '''remove outliers from column Height after changing units'''
        return df.query('`Height` > 0 and `Height` <= 0.25')

    @staticmethod
    def check_sum_of_weights(df):
       '''check if sum of weights features are lower than whole crab`s weight'''
       return df.query('`Weight` > `Shucked Weight` + `Viscera Weight` + `Shell Weight`')

    @staticmethod
    def change_weight_features_to_ratio(df):
        '''change values in columns connected with weight to ratio'''
        df['Shell Weight'] = df['Shell Weight']/df['Weight']
        df['Shucked Weight'] = df['Shucked Weight']/df['Weight']
        df['Viscera Weight'] = df['Viscera Weight']/df['Weight']
        return df

    def process(self, df):
        '''run data pipeline and return processed dataframe with one hot encoded Sex column'''
        processed_df = (df.pipe(DataProcessor.change_units)
                        .pipe(DataProcessor.remove_outliers)
                        .pipe(DataProcessor.check_sum_of_weights)
                        .pipe(DataProcessor.change_weight_features_to_ratio))
        processed_df = pd.get_dummies(processed_df)
        return processed_df

def main():
    data = pd.read_csv(cfg.DATA_PATH, delimiter=',')
    
    DP = DataProcessor()
    processed_data = DP.process(data)

    train, valid = train_test_split(processed_data, test_size=0.3, random_state=42)

    train.to_csv(os.path.join(cfg.DATA_DIR, 'processed_train.csv'), index=False)
    valid.to_csv(os.path.join(cfg.DATA_DIR, 'processed_valid.csv'), index=False)

if __name__ == '__main__':
    main()
