# Crab Age Prediction

The main object of this project is estimating the age of a crab based on it's phisical attributes.

## About the dataset

* Sex - Gender of the Crab - Male, Female and Indeterminate.
* Length - Length of the Crab (in Feet; 1 foot = 30.48 cms)
* Diameter - Diameter of the Crab (in Feet; 1 foot = 30.48 cms)
* Height - Height of the Crab (in Feet; 1 foot = 30.48 cms)
* Weight - Weight of the Crab (in ounces; 1 Pound = 16 ounces)
* Shucked Weight - Weight without the shell (in ounces; 1 Pound = 16 ounces)
* Viscera Weight - Weight that wraps around your abdominal organs deep inside  body (in ounces; 1 Pound = 16 ounces)
* Shell Weight - Weight of the Shell (in ounces; 1 Pound = 16 ounces)
* Age - Age of the Crab (in months)

## How to run this project

After cloning the repository set up virtual environment and install requirements

```
python -m venv venv 
source venv/bin/activate
pip install -r requirements.txt
```

## Running the project

In this project we use DVC (Data Version Control) for running the data processing pipeline.
First, pull data from dvc using dvc-ssh method.

Then, run the 'prepare' stage to preparade raw data

```
dvc repro prepare
```

Run the 'evaluate' stage to evalute models
```
dvc repro evaluate
```
