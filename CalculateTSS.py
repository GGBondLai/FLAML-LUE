# -*- coding: utf-8 -*-
'''
@File    :   CalculateTSS.py
@Time    :   2024/12/24 09:16:27
@Author  :   GGLai 
@Version :   1.0
@Desc    :   None
'''
# %%
import os
import pandas as pd
import numpy as np
import warnings
from math import sqrt
from sklearn.metrics import r2_score, mean_squared_error
warnings.filterwarnings("ignore", category=FutureWarning)
# %% CalculateTSS based on different stations
timescale = ['daily','8days','16days','monthly']
FLAML_name = ['FLAML00','FLAML01','FLAML02','FLAML03','FLAML04','FLAML05','FLAML06','FLAML07','FLAML08',
              'FLAML10','FLAML11','FLAML12','FLAML13','FLAML14','FLAML15','FLAML16','FLAML17','FLAML18']
typelist = ['Forest','Grass','Crop']
test_station = ['ALF','CBF','QYF','DLG','DXG','HBG_S01','JZA','YCA']
df_new = pd.DataFrame(columns=['station', 'time','FLAML','R2','R','Pred_SD','Tower_SD','SD','nuRMSE','RMSE','TSS'])
for type in typelist:
    for time in timescale:
        for station in test_station:
            for i in range(0,18): 
                file = f'../Output/{type}Test/Pred{type}_{time}_{station}_{FLAML_name[i]}.xlsx'
                if not os.path.exists(file):
                    continue
                else:
                    df = pd.read_excel(file,header=[0])
                    x = df['PredGPP']
                    y = df['TowerGPP']
                    r2 = r2_score(y, x)
                    mse = mean_squared_error(y, x)
                    x_mean = np.mean(x)
                    y_mean = np.mean(y)
                    # R
                    sigma_f = np.sqrt(np.mean((x - x_mean)**2))
                    sigma_o = np.sqrt(np.mean((y - y_mean)**2))
                    sigma_hat_f = sigma_f / sigma_o
                    R = np.sum((x - x_mean) * (y - y_mean)) / (sigma_f * sigma_o * len(x))
                    # SD
                    x_SD = np.std(x)
                    y_SD = np.std(y)
                    nuRMSE = sqrt(mse)/y_SD
                    SD = x_SD/y_SD
                    # TSS
                    R_0 = 1
                    S = 4 * (1 + R) / ((sigma_hat_f + 1/sigma_hat_f)**2 * (1 + R_0))
                    new_row = {'station': station, 'time': time,'FLAML':FLAML_name[i],'R2':r2,'R':R,
                               'Pred_SD':x_SD,'Tower_SD':y_SD,'SD':SD,'nuRMSE':nuRMSE,'RMSE':sqrt(mse),'TSS':S}
                    df_new = pd.concat([df_new, pd.DataFrame([new_row])], ignore_index=True)

    df_new.to_excel('../Output/EvaluationIndex_station.xlsx', index=False)
# %% CalculateTSS based on different ecosystems
typelist = ['Forest','Grass','Crop']
timescale = ['daily','8days','16days','monthly']
FLAML_name = ['FLAML00','FLAML01','FLAML02','FLAML03','FLAML04','FLAML05','FLAML06','FLAML07','FLAML08',
              'FLAML10','FLAML11','FLAML12','FLAML13','FLAML14','FLAML15','FLAML16','FLAML17','FLAML18']
dict = {'Forest':['ALF','CBF','QYF'],'Grass':['DLG','DXG','HBG_S01'],'Crop':['JZA','YCA']}
df_new = pd.DataFrame(columns=['station', 'time','FLAML','R2','R','Pred_SD','Tower_SD','SD','nuRMSE','RMSE','TSS'])
keys = dict.keys()
for key in keys:
    df_new = pd.DataFrame()
    for time in timescale:
        name = f'{key}_{time}'
        for i in range(0,18):
            df_station=pd.DataFrame()
            for station in dict[key]:
                df_path = f'../Output/{key}Test/Pred{name}_{station}_{FLAML_name[i]}.xlsx'
                df = pd.read_excel(df_path,header=[0])
                df_station = pd.concat([df_station, df])
            x = df_station['PredGPP']
            y = df_station['TowerGPP']
            r2 = r2_score(y, x)
            mse = mean_squared_error(y, x)
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            # R
            sigma_f = np.sqrt(np.mean((x - x_mean)**2))
            sigma_o = np.sqrt(np.mean((y - y_mean)**2))
            sigma_hat_f = sigma_f / sigma_o
            R = np.sum((x - x_mean) * (y - y_mean)) / (sigma_f * sigma_o * len(x))
            # SD
            x_SD = np.std(x)
            y_SD = np.std(y)
            nuRMSE = sqrt(mse)/y_SD
            SD = x_SD/y_SD
            # TSS
            R_0 = 1
            S = 4 * (1 + R) / ((sigma_hat_f + 1/sigma_hat_f)**2 * (1 + R_0))
            new_row = {'station': key, 'time': time,'FLAML':FLAML_name[i],'R2':r2,'R':R,
                        'Pred_SD':x_SD,'Tower_SD':y_SD,'SD':SD,'nuRMSE':nuRMSE,'RMSE':sqrt(mse),'TSS':S}
            df_new = pd.concat([df_new, pd.DataFrame([new_row])], ignore_index=True)

    df_new.to_excel(f'../Output/EvaluationIndex_{key}.xlsx', index=False)