import pandas as pd
import pickle
import logging
from sklearn.model_selection import train_test_split
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

from flaml import AutoML
from flaml.automl.ml import sklearn_metric_loss_score

from flaml.automl.data import get_output_from_log
import numpy as np

# Create a list of different indicator combinations and train them one by one
FLAML_name = ['FLAML00','FLAML01','FLAML02','FLAML03','FLAML04','FLAML05','FLAML06','FLAML07','FLAML08',
              'FLAML10','FLAML11','FLAML12','FLAML13','FLAML14','FLAML15','FLAML16','FLAML17','FLAML18']
radiation = [['PAR','T_flux'],['PAR_modis','T_era5']]
index = ['EVI','NDVI','LAI']
water = ['LSWI','PDSI','EF']
FLAML = []
for r in radiation:
    for i in index:
        for w in water:
            output_list = [r[0],r[1],i,w,'VegetationTypes','Season','DOY','elevation']
            FLAML.append(output_list)

# Set up the site as the training data set
teststation1 = 'JZA'
teststation2 = 'YCA'
# Training datasets for different ecosystem types
Integration_data = pd.read_excel('../Merged_CropData_8days')

# Extract training data sets and test data sets
trainStation = Integration_data[~((Integration_data['station_name'] == teststation1) & (Integration_data['Date'].dt.year >= 2010) & (Integration_data['Date'].dt.year <= 2014))]
trainStation = trainStation[~((trainStation['station_name'] == teststation2) & (trainStation['Date'].dt.year >= 2008) & (trainStation['Date'].dt.year <= 2010))].dropna()
JZA_data = Integration_data[(Integration_data['station_name'] == teststation1) & (Integration_data['Date'].dt.year >= 2010) & (Integration_data['Date'].dt.year <= 2014)]
YCA_data = Integration_data[(Integration_data['station_name'] == teststation2) & (Integration_data['Date'].dt.year >= 2008) & (Integration_data['Date'].dt.year <= 2010)]
testStation = pd.concat([JZA_data, YCA_data]).dropna()

for i in range(0,18): 
    print('Current indicator selection:',FLAML_name[i])  
    automl_settings = {
        "time_budget": 120,  # in seconds
        "metric": 'r2',
        "task": 'regression',
        "log_file_name":f'../Crop_{FLAML_name[i]}.log' , # None Example Set the path for storing historical records when the model is running
        "seed" : 42    # Set the random seed to a specific value
    }
    automl = AutoML(**automl_settings)

    new_list = FLAML[i] + ['TowerGPP']

    train = trainStation[new_list]
    X_train = train[FLAML[i]]  # Influence factor
    y_train = train['TowerGPP']  # Target variable

    test = testStation[new_list]
    X_test = test[FLAML[i]]  
    y_test = test['TowerGPP']

    # Train with labeled input data
    automl.fit(X_train=X_train, y_train=y_train)
    # save the model 
    filename = f'../Crop_{FLAML_name[i]}.sav'
    pickle.dump(automl, open(filename, 'wb'))
    # Export the best model
    best_model = automl.model
    print('Best model:',best_model)
    print('Best ML leaner:', automl.best_estimator)
    print('Best hyperparmeter config:', automl.best_config)
    print('Best accuracy on validation data: {0:.4g}'.format(1-automl.best_loss))
    print('Training duration of best run: {0:.4g} s'.format(automl.best_config_train_time))

