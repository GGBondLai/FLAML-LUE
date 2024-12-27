# -*- coding: utf-8 -*-
'''
@File    :   FLAML-LUE-Crop.py
@Time    :   2024/12/23 16:33:14
@Author  :   GGLai 
@Version :   1.0
@Desc    :   None
'''
# %%
import pandas as pd
import pickle
import logging
import seaborn as sns
import numpy as np
import warnings
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from flaml import AutoML
from flaml.automl.ml import sklearn_metric_loss_score
from flaml.automl.data import get_output_from_log
# Set Flaml's logger level to WARNING
logging.getLogger('flaml.automl').setLevel(logging.WARNING)
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore", category=FutureWarning)

# %%
timescale = ['daily','8days','16days','monthly']
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

type = 'Crop'
teststation1 = 'JZA'
teststation2 = 'YCA'

for time in timescale:
    print('The files involved in the calculation:','Merged_'+type+'Data_'+time+'.xlsx')
    name = f'{type}_{time}'
    file_name = 'Merged_'+type+'Data_'+time+'.xlsx'
    Integration_data = pd.read_excel('../Input/'+file_name)

    # Divide training data and validation data
    trainStation = Integration_data[~((Integration_data['station_name'] == teststation1) & (Integration_data['Date'].dt.year >= 2010) & (Integration_data['Date'].dt.year <= 2014))]
    trainStation = trainStation[~((trainStation['station_name'] == teststation2) & (trainStation['Date'].dt.year >= 2008) & (trainStation['Date'].dt.year <= 2010))].dropna()
    
    JZA_data = Integration_data[(Integration_data['station_name'] == teststation1) & (Integration_data['Date'].dt.year >= 2010) & (Integration_data['Date'].dt.year <= 2014)]
    YCA_data = Integration_data[(Integration_data['station_name'] == teststation2) & (Integration_data['Date'].dt.year >= 2008) & (Integration_data['Date'].dt.year <= 2010)]

    testStation = pd.concat([JZA_data, YCA_data]).dropna()

    for i in range(0,18): 
        print('Current indicator combination:',FLAML_name[i])  
        automl_settings = {
            "time_budget": 120,  # in seconds
            "metric": 'r2',
            "task": 'regression',
            "log_file_name":f'../Output/{type}Log/{name}_{FLAML_name[i]}.log' ,
            "seed" : 42,   # Set the random seed to a specific value
        }
        automl = AutoML(**automl_settings)
        new_list = FLAML[i] + ['TowerGPP']

        train = trainStation[new_list]
        # train data
        X_train = train[FLAML[i]]
        y_train = train['TowerGPP']
        # test data
        test = testStation[new_list]
        X_test = test[FLAML[i]]
        y_test = test['TowerGPP']
        
        # Train with labeled input data
        automl.fit(X_train=X_train, y_train=y_train)
        # save the model 
        filename = f'../Model/{type}Model/{name}_{FLAML_name[i]}.sav'
        pickle.dump(automl, open(filename, 'wb'))

        # Export the best model
        best_model = automl.model
        print('Best model:',best_model)
        print('Best ML leaner:', automl.best_estimator)
        print('Best hyperparmeter config:', automl.best_config)
        print('Best accuracy on validation data: {0:.4g}'.format(1-automl.best_loss))
        print('Training duration of best run: {0:.4g} s'.format(automl.best_config_train_time))

        # Plot learning curve
        plt.rcParams['font.family'] = 'Times New Roman'
        fig, ax = plt.subplots()
        fig.set_figwidth(10)
        fig.set_figheight(10)
        plt.subplots_adjust(left=0.1,right=0.99)
        time_history, best_valid_loss_history, valid_loss_history, config_history, metric_history =get_output_from_log(filename=automl_settings['log_file_name'], time_budget=60)
        plt.title('Learning Curve_'+name+'_'+FLAML_name[i],fontsize = 26)
        plt.xlabel('Wall Clock Time (s)',fontsize = 26)
        plt.ylabel('Validation r2',fontsize = 26)
        ax.tick_params(axis='x', labelsize=26)
        ax.tick_params(axis='y', labelsize=26)
        plt.step(time_history, 1 - np.array(best_valid_loss_history), where='post')
        plt.savefig(f'../Figure/{type}LearningCurve/LC_{name}_{FLAML_name[i]}.jpg',dpi=1000)
        plt.close()
        # Get feature importance score
        feature_names = automl.feature_names_in_
        feature_importances = automl.feature_importances_
        feature_importance_pairs = list(zip(feature_names, feature_importances))
        # Sort the pairs by importance (descending order)
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
        # Print the sorted feature importances
        for feature, importance in feature_importance_pairs:
            print(f'{feature}: {importance}')
        # Plot feature importance score
        plt.rcParams['font.family'] = 'Times New Roman'
        fig, ax = plt.subplots()
        fig.set_figwidth(14)
        fig.set_figheight(6)
        plt.subplots_adjust(left=0.17,right=0.99,top=0.9,bottom=0.1)
        plt.barh(feature_names, feature_importances)
        for s, v in enumerate(feature_importances):
            plt.text(v, s, f'{v:.2f}', color='black', va='center',fontsize = 26)
        print(f'FIP_{name}_{FLAML_name[i]}')
        plt.title('FIP_'+name+'_'+FLAML_name[i],fontsize = 26)
        ax.tick_params(axis='x', labelsize=24)
        ax.tick_params(axis='y', labelsize=24)
        plt.savefig(f'../Figure/{type}FPI/FIP_{name}_{FLAML_name[i]}.jpg',dpi=1000)
        plt.close()
        # Validation based on the test dataset
        if X_test.empty:
            continue 
        # Simulation based on automatic machine learning model selection
        test_pred_AutoML = automl.predict(X_test)
        test_data = test.copy()
        test_data['PredGPP'] = test_pred_AutoML
        testOutput = testStation.copy()
        testOutput['PredGPP'] = test_pred_AutoML
        JZA_testdata = testOutput[(testOutput['station_name'] == teststation1)]
        YCA_testdata = testOutput[(testOutput['station_name'] == teststation2)]
        # Save simulation results based on different sites
        JZA_testdata.to_excel(f'../Output/{type}Test/Pred{name}_{teststation1}_{FLAML_name[i]}.xlsx', index=False)
        YCA_testdata.to_excel(f'../Output/{type}Test/Pred{name}_{teststation2}_{FLAML_name[i]}.xlsx', index=False)

        # Scatter chart JZA
        test1_y = JZA_testdata['TowerGPP']
        test1_x = JZA_testdata['PredGPP']
        test1_r2 = 1-sklearn_metric_loss_score('r2', test1_x, test1_y)
        test1_mse = sklearn_metric_loss_score('mse', test1_x, test1_y)
        test1_mae = sklearn_metric_loss_score('mae', test1_x, test1_y)
        print(f'{teststation1}_r2:',test1_r2)
        plt.rcParams['font.family'] = 'Times New Roman'
        fig, ax = plt.subplots()
        fig.set_figwidth(10)
        fig.set_figheight(10)
        plt.scatter(test1_x, test1_y)
        max_x = max(test1_x)
        plt.plot([-3, max_x+3], [-3, max_x+3], color='darkblue', linestyle='--', linewidth=1.5, label='y = x')
        # Fit a linear regression line with intercept set to zero
        model = LinearRegression(fit_intercept=False)
        model.fit(test1_x.values.reshape(-1, 1), test1_y.values.reshape(-1, 1))
        a = model.coef_[0][0]
        # Plot the linear regression line
        plt.plot(test1_x, a * test1_x, color='red', linewidth=1.5, label=f'y = {a:.2f}x')
        plt.axis('equal')
        ax.text(0.1, 0.95, f'r2 = {test1_r2:.4f}', fontsize=26, color='black', transform=ax.transAxes)
        ax.text(0.1, 0.9, f'mse = {test1_mse:.4f}', fontsize=26, color='black', transform=ax.transAxes)
        ax.text(0.1, 0.85, f'mae = {test1_mae:.4f}', fontsize=26, color='black', transform=ax.transAxes)
        ax.text(0.1, 0.8, f'Best model: {automl.best_estimator}', fontsize=26, color='black', transform=ax.transAxes)
        plt.title(f'GPP Comparison_{teststation1}_{FLAML_name[i]}', fontsize=26)
        plt.xlabel('PredGPP', fontsize=26)
        plt.ylabel('TowerGPP', fontsize=26)
        plt.legend(loc='lower right',fontsize=26)
        ax.tick_params(axis='x', labelsize=26)
        ax.tick_params(axis='y', labelsize=26)
        max_value = max(max(test1_x), max(test1_y))
        plt.xlim(-1, max_value+3)
        plt.ylim(-1, max_value+3)
        plt.savefig(f'../Figure/{type}Test/Scatter{name}_{teststation1}_{FLAML_name[i]}.jpg', dpi=1000)
        plt.close()
        # Scatter chart YCA
        test2_y = YCA_testdata['TowerGPP']
        test2_x = YCA_testdata['PredGPP']
        test2_r2 = 1-sklearn_metric_loss_score('r2', test2_x, test2_y)
        test2_mse = sklearn_metric_loss_score('mse', test2_x, test2_y)
        test2_mae = sklearn_metric_loss_score('mae', test2_x, test2_y)
        print(f'{teststation2}_r2:',test2_r2)
        plt.rcParams['font.family'] = 'Times New Roman'
        fig, ax = plt.subplots()
        fig.set_figwidth(10)
        fig.set_figheight(10)
        plt.scatter(test2_x, test2_y)
        max_x = max(test2_x)
        plt.plot([-3, max_x+3], [-3, max_x+3], color='darkblue', linestyle='--', linewidth=1.5, label='y = x')
        # Fit a linear regression line with intercept set to zero
        model = LinearRegression(fit_intercept=False)
        model.fit(test2_x.values.reshape(-1, 1), test2_y.values.reshape(-1, 1))
        a = model.coef_[0][0]
        # Plot the linear regression line
        plt.plot(test2_x, a * test2_x, color='red', linewidth=1.5, label=f'y = {a:.2f}x')
        plt.axis('equal')
        ax.text(0.1, 0.95, f'r2 = {test2_r2:.4f}', fontsize=26, color='black', transform=ax.transAxes)
        ax.text(0.1, 0.9, f'mse = {test2_mse:.4f}', fontsize=26, color='black', transform=ax.transAxes)
        ax.text(0.1, 0.85, f'mae = {test2_mae:.4f}', fontsize=26, color='black', transform=ax.transAxes)
        ax.text(0.1, 0.8, f'Best model: {automl.best_estimator}', fontsize=26, color='black', transform=ax.transAxes)
        plt.title(f'GPP Comparison_{teststation2}_{FLAML_name[i]}', fontsize=26)
        plt.xlabel('PredGPP', fontsize=26)
        plt.ylabel('TowerGPP', fontsize=26)
        plt.legend(loc='lower right',fontsize=26)
        ax.tick_params(axis='x', labelsize=26)
        ax.tick_params(axis='y', labelsize=26)
        max_value = max(max(test2_x), max(test2_y))
        plt.xlim(-1, max_value+3)
        plt.ylim(-1, max_value+3)
        plt.savefig(f'../Figure/{type}Test/Scatter{name}_{teststation2}_{FLAML_name[i]}.jpg', dpi=1000)
        plt.close()

# %%
