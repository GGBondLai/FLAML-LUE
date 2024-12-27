# -*- coding: utf-8 -*-
'''
@File    :   ImportanceRankingScore.py
@Time    :   2024/12/24 10:31:55
@Author  :   GGLai 
@Version :   1.0
@Desc    :   None
'''

# %%
import numpy as np
import pandas as pd
import pickle

df = pd.DataFrame(columns=['Type','time','FLAML','BestML','PAR','T_flux','PAR_modis',
                           'T_era5','EVI','NDVI','LAI','LSWI','PDSI','EF',
                           'VegetationTypes','Season','DOY','elevation'])
feature_columns = ['PAR', 'T_flux', 'PAR_modis', 'T_era5', 'EVI', 'NDVI', 'LAI',
                   'LSWI', 'PDSI', 'EF','VegetationTypes', 'Season', 'DOY', 'elevation']
ecosystemList = ['Forest','Grass','Crop']
timescale = ['daily','8days','16days','monthly']
FLAML_name = ['FLAML00','FLAML01','FLAML02','FLAML03','FLAML04','FLAML05','FLAML06','FLAML07','FLAML08',
              'FLAML10','FLAML11','FLAML12','FLAML13','FLAML14','FLAML15','FLAML16','FLAML17','FLAML18']
for ecosystem in ecosystemList:
    for time in timescale:
        for flaml in FLAML_name:
            filename = rf'../Model/{ecosystem}Model/{ecosystem}_{time}_{flaml}.sav'
            with open(filename, 'rb') as file:
                automl = pickle.load(file)
            bestML = automl.best_estimator
            feature_names = automl.feature_names_in_
            feature_importances = automl.feature_importances_
            feature_importance_pairs = list(zip(feature_names, feature_importances))
            # Sort the pairs by importance (ascending order, so highest importance gets the last rank)
            feature_importance_pairs.sort(key=lambda x: x[1], reverse=False)
            ranked_features = {name: rank + 1 for rank, (name, importance) in enumerate(feature_importance_pairs)}
            row = {
                'Type': ecosystem,
                'time': time,
                'FLAML': flaml,
                'BestML': bestML
            }
            for feature in feature_columns:
                row[feature] = ranked_features.get(feature, np.nan)
            row_df = pd.DataFrame([row])
            df = pd.concat([df, row_df], ignore_index=True)
df['T'] = df['T_flux'].fillna(0) + df['T_era5'].fillna(0)
df['PAR'] = df['PAR'].fillna(0) + df['PAR_modis'].fillna(0)
df['VT'] = df['VegetationTypes']
select_columns = ['Type','time','FLAML','BestML','PAR','T','EVI','NDVI','LAI','LSWI','PDSI','EF',
                  'VT','Season','DOY','elevation']
df.to_excel('../Output/ImportanceRanking.xlsx', index=False,columns=select_columns)

# %%
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_excel('../Output/ImportanceRanking.xlsx')
for ecosystem in ecosystemList:
    data = df[df['Type']==f'{ecosystem}'].iloc[:, 4:]
    features = list(data.keys())
    values = np.array([data[f] for f in features])
    means = np.nanmean(values, axis=1)
    std_devs = np.nanstd(values, axis=1)

    x = np.arange(len(features))
    width = 0.7
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, ax = plt.subplots()
    fig.set_figwidth(16)
    fig.set_figheight(10)
    plt.subplots_adjust(left=0.07,right=0.97,top=0.93,bottom=0.05)
    ax.bar(x, means, width, yerr=std_devs, capsize=5, color='skyblue', edgecolor='skyblue')

    ax.set_xticks(x)
    ax.set_xticklabels(features, fontweight='bold')
    ax.set_ylim(0, 9)
    ax.tick_params(axis='x', labelsize=26)
    ax.tick_params(axis='y', labelsize=26)
    plt.title(f'Importance analysis of variables in {ecosystem} cosystem',fontsize = 28,fontweight='bold')
    # plt.xlabel('Variables',fontsize = 26, fontweight='bold')
    plt.ylabel('RankingScore',fontsize = 26, fontweight='bold',labelpad=20)

    plt.savefig(f'../Figure/ImportanceRanking{ecosystem}.jpg',dpi=1000)
    plt.show()
    
# %%
