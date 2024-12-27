# -*- coding: utf-8 -*-
'''
@File    :   CompareOtherGPPs.py
@Time    :   2024/12/26 20:09:45
@Author  :   GGLai 
@Version :   1.0
@Desc    :   None
'''

# %% Calulate R2 and plot
import os
import pandas as pd
import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline
from matplotlib.dates import MonthLocator, DateFormatter
from matplotlib.dates import DateFormatter, YearLocator
from matplotlib.font_manager import FontProperties
import matplotlib.dates as mdates
# %%
rcParams["font.family"] = "Times New Roman"
ecosystems = ['Forest','Grass','Crop']
all_station = {'Forest':['ALF','CBF','QYF'],
            'Grass':['DLG','DXG','HBG_S01'],
            'Crop':['JZA','YCA']}
indx = ['a','b','c','d','e','f','g','h']
fig, axes = plt.subplots(nrows = 3, ncols = 3,figsize=(18, 12), dpi = 1000)
print(axes.shape)
z = -1
for i in range(3):
    ty = ecosystems[i]
    df = pd.read_excel(f'../Output/PredStations/Pred_{ty}_8days_best.xlsx')
    stations = all_station[ty]
    for j in range(3):
        if i <3 and j < 3:
            ax = axes[i,j]
            if i == 2 and j == 2:
                ax.axis('off')
                ax.plot([], [], color='orange', label='FlamlGPP', linestyle='-', linewidth=1)  # 用空数据添加图例
                ax.plot([], [], color='gray', label='MODGPP', linestyle='--', linewidth=1)
                ax.plot([], [], color='red', label='PMLGPP', linestyle='--', linewidth=1)
                ax.scatter([], [], color='black', s=50, label='TowerGPP', marker='*')
                ax.legend(loc='center', fontsize=16, bbox_to_anchor=(0.5, 0.5))
            else:
                z += 1
                station = stations[j]
                print(station)
                df_pred = df[df['station_name']==station]
                df_other = pd.read_csv(f'../Input/OtherGPPs/extractedGPP_{station}_8days.csv')
                df_pred.loc[:, 'Date'] = pd.to_datetime(df_pred['Date'])
                df_other['Date'] = pd.to_datetime(df_other['Date'])
                df_merge = pd.merge(df_pred, df_other, on='Date', how='left')
                print(df_merge.shape)
                ax.plot(df_merge['Date'], df_merge['PredGPP'], color='orange', label='FlamlGPP', linestyle='-', linewidth=1)
                ax.plot(df_merge['Date'], df_merge['GPP_MOD'], color='gray', label='MODGPP', linestyle='--', linewidth=1)
                ax.plot(df_merge['Date'], df_merge['GPP_PML'], color='red', label='PMLGPP', linestyle='--', linewidth=1)
                ax.scatter(df_merge['Date'], df_merge['TowerGPP'], color='black', s=60, label='TowerGPP', marker='*')
                ax.tick_params(axis='x', labelsize=14,rotation=0)
                ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[6])) 
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                xticklabels = ax.get_xticklabels()
                if len(xticklabels) > 8:
                    for x, label in enumerate(xticklabels):
                        if x % 2 == 0:
                            label.set_fontweight('bold')
                        else:
                            label.set_visible(False)
                ax.set_ylabel('GPP (gC $m^{-2} 8d^{-1}$)', fontsize=16, fontweight='bold',labelpad=20)
                ax.yaxis.set_label_coords(-0.12, 0.5)
                # ax.set_ylim(-10,160)
                ax.tick_params(axis='y', labelsize=14)
                for label in ax.get_yticklabels():
                    label.set_fontweight('bold')
                ax.text(0.03, 0.98, f"({indx[z]}) {station}", ha='left', va='top', fontsize=16, fontweight='bold', transform=ax.transAxes)
plt.tight_layout()
plt.savefig(f'../Figure/CompareOtherGPP.jpg', dpi=1000)
plt.show()


# %% Calculate R2
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress
import numpy as np
from matplotlib import rcParams
from flaml.automl.ml import sklearn_metric_loss_score
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

df_new = pd.DataFrame(columns=['station', 'Pred_R2','MOD_R2','PML_R2'])
ecosystems = ['Forest','Grass','Crop']
all_station = {'Forest':['ALF','CBF','QYF'],
            'Grass':['DLG','DXG','HBG_S01'],
            'Crop':['JZA','YCA']}
for ty in ecosystems:
    df = pd.read_excel(f'../Output/PredStations/Pred_{ty}_8days_best.xlsx')
    stations = all_station[ty]
    for station in stations:
        df_pred = df[df['station_name']==station]
        df_other = pd.read_csv(f'../Input/OtherGPPs/extractedGPP_{station}_8days.csv')
        df_pred.loc[:, 'Date'] = pd.to_datetime(df_pred['Date'])
        df_other['Date'] = pd.to_datetime(df_other['Date'])
        df_merge = pd.merge(df_pred, df_other, on='Date', how='inner')
        print(df_merge.shape)
        r_squared1 = np.corrcoef(df_merge['TowerGPP'],df_merge['PredGPP'])[0, 1]**2
        r_squared2 = np.corrcoef(df_merge['TowerGPP'],df_merge['GPP_MOD'])[0, 1]**2
        r_squared3 = np.corrcoef(df_merge['TowerGPP'],df_merge['GPP_PML'])[0, 1]**2
        new_row = {'station': station, 'Pred_R2':r_squared1,
                   'MOD_R2':r_squared2,'PML_R2':r_squared3}
        df_new = pd.concat([df_new, pd.DataFrame([new_row])], ignore_index=True)
print(df_new)
df_new.to_excel('../Output/CompareR2.xlsx', index=False)
# %%
