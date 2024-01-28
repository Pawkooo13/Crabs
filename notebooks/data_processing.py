# %%
import pandas as pd
import os
from seaborn import kdeplot, heatmap, lmplot

# %%
train_path = os.path.join(os.getcwd()[:-9], 'data\\train.csv')

# %%
train_data = pd.read_csv(train_path, delimiter=',')

# %%
train_data.head()

# %% [markdown]
# * Sex - Gender of the Crab - Male, Female and Indeterminate.
# * Length - Length of the Crab (in Feet; 1 foot = 30.48 cms)
# * Diameter - Diameter of the Crab (in Feet; 1 foot = 30.48 cms)
# * Height - Height of the Crab (in Feet; 1 foot = 30.48 cms)
# * Weight - Weight of the Crab (in ounces; 1 Pound = 16 ounces)
# * Shucked Weight - Weight without the shell (in ounces; 1 Pound = 16 ounces)
# * Viscera Weight - Weight that wraps around your abdominal organs deep inside  body (in ounces; 1 Pound = 16 ounces)
# * Shell Weight - Weight of the Shell (in ounces; 1 Pound = 16 ounces)
# * Age - Age of the Crab (in months)

# %%
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

# %%
train_data = train_data.apply(change_units, axis=1)

# %%
train_data.head()

# %%
train_data.isnull().sum()

# %%
train_data.isna().sum()

# %%
train_data.describe()

# %%
train_data = train_data.query('Height > 0')

# %% [markdown]
# check if there are crabs with Weight lower than sum of Shucked, Shell and Viscera Weight

# %%
train_data.query('`Weight` < `Shucked Weight` + `Viscera Weight` + `Shell Weight`')

# %%
train_data = train_data.query('`Weight` > `Shucked Weight` + `Viscera Weight` + `Shell Weight`')

# %%
train_data.describe()

# %%
kdeplot(data = train_data, x='Height', hue='Sex', fill=True)

# %%
kdeplot(data = train_data, x='Weight', hue='Sex', fill=True)

# %%
kdeplot(data = train_data, x='Diameter', hue='Sex', fill=True)

# %%
kdeplot(data = train_data, x='Shucked Weight', hue='Sex', fill=True)

# %%
kdeplot(data = train_data, x='Shell Weight', hue='Sex', fill=True)

# %%
kdeplot(data = train_data, x='Viscera Weight', hue='Sex', fill=True)

# %% [markdown]
# change shell, viscera and shucked weight to ratio

# %%
train_data['Shell Weight'] = train_data['Shell Weight']/train_data['Weight']
train_data['Shucked Weight'] = train_data['Shucked Weight']/train_data['Weight']
train_data['Viscera Weight'] = train_data['Viscera Weight']/train_data['Weight']

# %%
train_data.describe()

# %% [markdown]
# one hot encoding Sex

# %%
def one_hot_encode_sex(df):
    one_hot_sex = pd.get_dummies(df.Sex, dtype = float, prefix = 'Sex')
    new_df = pd.concat([df, one_hot_sex], axis = 'columns')
    
    col_names = list(df.columns.values) + list(one_hot_sex.columns.values)
    new_df.columns = col_names
    
    new_df = new_df.drop('Sex', axis = 'columns')
    
    return new_df

# %%
train_data = one_hot_encode_sex(train_data)

# %%
train_data.head()

# %%
heatmap(train_data.corr(), cmap='Blues', annot=True)

# %%
import matplotlib.pyplot as plt

for col in train_data.columns[:7]:
    plt.show(lmplot(data=train_data, x=col, y='Age'))

# %% [markdown]
# remove outlier 

# %%
outliers_idx = train_data.query('`Height` > 0.25').index

# %%
train_data = train_data.drop(outliers_idx)

# %%
train_data.describe()

# %%
lmplot(data=train_data, x='Height', y='Age')

# %%


# %%



