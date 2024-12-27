# -*- coding: utf-8 -*-
'''
@File    :   ScatterChart.py
@Time    :   2024/12/25 20:02:11
@Author  :   GGLai 
@Version :   1.0
@Desc    :   None
'''

# %%
import os
from matplotlib.font_manager import FontProperties
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from flaml.automl.ml import sklearn_metric_loss_score
import pickle
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# %% Simulation of GPP for all time scales at all sites based on the optimal combination of metrics
timescale = ['daily','8days','16days','monthly']
ecosystems = ['Forest','Grass','Crop']
# Forest:FLAML17;Grass:FLAML08;Crop:FLAML00
BestML = [16,8,0]

radiation = [['PAR','T_flux'],['PAR_modis','T_era5']]
index = ['EVI','NDVI','LAI']
water = ['LSWI','PDSI','EF']
FLAML = []
for r in radiation:
    for i in index:
        for w in water:
            output_list = [r[0],r[1],i,w,'VegetationTypes','Season','DOY','elevation']
            FLAML.append(output_list)
FLAML_name = ['FLAML00','FLAML01','FLAML02','FLAML03','FLAML04','FLAML05','FLAML06','FLAML07','FLAML08',
              'FLAML10','FLAML11','FLAML12','FLAML13','FLAML14','FLAML15','FLAML16','FLAML17','FLAML18']
for i in range(3):
    j = BestML[i]
    flaml = FLAML_name[j]
    ecosystem = ecosystems[i]
    print(ecosystem)
    print(flaml)
    for time in timescale:
        filename = f'../Input/Merged_{ecosystem}Data_{time}.xlsx'
        model = f'../Model/{ecosystem}Model/{ecosystem}_{time}_{flaml}.sav'
        with open(model, 'rb') as file:
                automl = pickle.load(file)
        data = pd.read_excel(filename)
        X_data = data[FLAML[j]]
        data['PredGPP'] = automl.predict(X_data)
        data.to_excel(f'../Output/PredStations/Pred_{ecosystem}_{time}_best.xlsx')

# %% ScatterChart seaborn
import os
from matplotlib.font_manager import FontProperties
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
def calculate_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot
def calculate_metrics(y, x):
    r2 = r2_score(y, x)
    mse = mean_squared_error(y, x)
    mae = mean_absolute_error(y, x)
    residuals = y - x
    std_residuals = np.std(residuals)
    return r2, mse, mae, std_residuals
ecosystems = ['Forest','Grass','Crop']
# Set the axis range
# forest:25,Grass:15,Crop:40
y_range = [25,15,40]
for i in range(3):
    ecosystem = ecosystems[i]
    y_max = y_range[i]
    file_path = f'../Output/PredStations/Pred_{ecosystem}_daily_best.xlsx'
    data = pd.read_excel(file_path)

    file_name = os.path.splitext(os.path.basename(file_path))[0]
    print(file_name)

    # Scatter chart
    x = data['PredGPP']
    y = data['TowerGPP']

    station_name = data['station_name']
    r2 = 1 - sklearn_metric_loss_score('r2', y, x)
    # r2 = calculate_r2(y,x)
    residuals = y - x
    std = np.std(residuals)
    mse = sklearn_metric_loss_score('mse', y, x)
    mae = sklearn_metric_loss_score('mae', y, x)

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.figure(figsize=(10, 10))

    # Use seaborn's scatterplot function to draw a scatterplot and set different colors based on station_index.
    ax = sns.scatterplot(x=x, y=y, hue=station_name, palette='Set1')
    font_1 = FontProperties(family='Times New Roman', weight='bold', size=20)
    font_2 = FontProperties(family='Times New Roman', weight='bold', size=18)

    # Fit a linear regression line with intercept set to zero
    model = LinearRegression(fit_intercept=False)
    model.fit(x.values.reshape(-1, 1), y.values.reshape(-1, 1))
    a = model.coef_[0][0]
    # Plot the linear regression line
    plt.plot(x, a * x, color='red',linestyle='--', linewidth=2)
    # Drawing 1:1 diagonal lines
    plt.plot([0, 39], [0, 39], color='darkblue', linestyle='--', linewidth=2,)
    equation_text = f'y = {a:.2f}x'
    # Add text, use relative position
    ax.text(0.8, 0.95, 'y = x', fontproperties=font_1, color='black', transform=ax.transAxes)
    ax.text(0.8, 0.7, f'y = {a:.4f}x', fontproperties=font_1, color='black', transform=ax.transAxes)
    ax.text(0.05, 0.95, f'r² = {r2:.4f}', fontproperties=font_1, color='black', transform=ax.transAxes)
    ax.text(0.05, 0.90, f'MSE = {mse:.4f}', fontproperties=font_1, color='black', transform=ax.transAxes)
    ax.text(0.05, 0.85, f'STD = {std:.4f}', fontproperties=font_1, color='black', transform=ax.transAxes)

    # plt.title('GPP Comparison')
    plt.xticks(fontproperties=font_2)
    plt.yticks(fontproperties=font_2)
    # plt.xlabel('PredGPP (gC·m\u207B\u00B2d\u207B\u00B9)', fontproperties=font_1)
    # plt.ylabel('TowerGPP(gC·m\u207B\u00B2d\u207B\u00B9)',fontproperties=font_1)

    # Subscripting with MathText
    plt.xlabel(r'GPP$_{pred}$ (gC·m$^{-2}$d$^{-1}$)', fontproperties=font_1)
    plt.ylabel(r'GPP$_{tower}$ (gC·m$^{-2}$d$^{-1}$)', fontproperties=font_1)
    # Set the legend position and show only the legend for points, adjust the legend font size
    legend_font = FontProperties(family='Times New Roman', size=16, weight='bold')
    plt.legend(loc='lower right', prop=legend_font)
    
    plt.xticks(np.arange(0, y_max+1, 5))
    plt.yticks(np.arange(0, y_max+1, 5))
    plt.xlim(0, y_max)
    plt.ylim(0, y_max)
    plt.savefig(f'../Figure/ScatterSeaborn_{ecosystem}.jpg', dpi=1000)

# %% Bias Chart--Month
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.ticker import FuncFormatter
group = {'Forest':{'Forest':['ALF','BNF','CBF','DHF','MEF','HZF','QYF'],
                   'NF':['HZF','QYF'],'MF':['CBF','DHF'],'EBF':['ALF','BNF'],'DBF':['MEF']},
         'Grass':{'Grass':['XLG','NMG','DLG','DMG','HBG_S01','HBG_G01','DXG'],
                  'Grassland':['XLG','NMG','DLG','DMG'],'Shrub':['HBG_S01'],'Meadow':['HBG_G01','DXG']},
         'Crop':{'Crop':['JZA','SYA','GCA','LCA','YCA','JRA'],
                   'SC':['JZA','SYA'],'DC':['GCA','LCA','YCA','JRA']}}
# group = {'Forest':{'Forest':['ALF','BNF','CBF','DHF','MEF','HZF','QYF'],
#                    'NF':['HZF','QYF'],'MF':['CBF','DHF'],'EBF':['ALF','BNF'],'DBF':['MEF']}}
plt.rcParams['font.family'] = 'Times New Roman'
def month_to_name(x, pos):
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    return months[int(x)-1]
x_range = [2.5,1.2,0.5]
for i,ecosystem in enumerate(group.keys()):
    filename = f'../Output/PredStations/Pred_{ecosystem}_daily_best.xlsx'
    df = pd.read_excel(filename)
    df['Bias'] = df['PredGPP']-df['TowerGPP']
    TypeList = group[ecosystem]
    ax_num = len(TypeList.keys())
    cols = math.ceil(ax_num / 2)
    fig, axes = plt.subplots(nrows = 2, ncols = cols, 
                             figsize=(5*cols, 10), dpi = 1000)
    axes_flat = axes.flatten()
    for index, (ty, ax) in enumerate(zip(TypeList.keys(), axes_flat)):
        stations = TypeList[ty]
        mask = df['station_name'].isin(stations)
        subset_data = df.loc[mask].copy()
        subset_data['Month'] = subset_data['Date'].dt.month
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        boxplot_data = subset_data.groupby('Month')['Bias'].apply(lambda x: list(x)).tolist()
        # Remove outliers by setting flierprops to an empty dictionary
        boxprops = dict(color='blue', facecolor='skyblue')
        flierprops = dict(marker='', color='white')  # Hide outliers completely
        boxprops = dict(color='blue', facecolor='skyblue')
        ax.boxplot(boxplot_data, vert=False, patch_artist=True, boxprops=boxprops, flierprops=flierprops)
        ax.yaxis.set_major_formatter(FuncFormatter(month_to_name))
        ax.set_ylabel(f'{ty}',fontsize = 20,fontweight = 'bold')
        ax.set_xlabel('Bias(gC m⁻² d⁻¹)',fontsize = 20,fontweight = 'bold')
        ax.set_xlim(-x_range[i], x_range[i])
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        xticklabels = ax.get_xticklabels()
        yticklabels = ax.get_yticklabels()
        for xlabel in ax.get_xticklabels():
            xlabel.set_fontweight('bold')
        for ylabel in ax.get_yticklabels():
            ylabel.set_fontweight('bold')
    for ax in axes_flat[ax_num:]:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'../Figure/Bias_{ecosystem}.jpg', dpi=1000)
    plt.show()
        

# %% LineChart - monthly
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import matplotlib.dates as mdates
import math
group = {'Forest':['ALF','BNF','CBF','DHF','MEF','HZF','QYF'],
         'Grass':['XLG','NMG','DLG','DMG','HBG_S01','HBG_G01','DXG'],
         'Crop':['JZA','SYA','GCA','LCA','YCA','JRA']}
# group = {'Forest':['ALF','BNF','CBF']}
indx = ['a','b','c','d','f','g','h']
plt.rcParams['font.family'] = 'Times New Roman'
for ecosystem in group.keys():
    df_8days = pd.read_excel(f'../Output/PredStations/Pred_{ecosystem}_8days_best.xlsx')
    df_16days = pd.read_excel(f'../Output/PredStations/Pred_{ecosystem}_16days_best.xlsx')
    df_monthly = pd.read_excel(f'../Output/PredStations/Pred_{ecosystem}_monthly_best.xlsx')
    stations = group[ecosystem]
    cols = math.ceil(len(stations) / 2)
    fig, axes = plt.subplots(nrows = 2, ncols = cols, 
                             figsize=(6*cols, 16), dpi = 1000)
    axes_flat = axes.flatten()
    for i, station in enumerate(stations):
        data_8days = df_8days[df_8days['station_name']==station]
        data_16days = df_16days[df_16days['station_name']==station]
        data_monthly = df_monthly[df_monthly['station_name']==station]
        dataList = [data_8days,data_16days,data_monthly]
        ax = axes_flat[i]
        ax.axis('off')
        gs = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=ax.get_subplotspec(), hspace=0)
        for j in range(3):
            data = dataList[j]
            y_max = data['TowerGPP'].max()
            sub_ax = fig.add_subplot(gs[j])
            sub_ax.plot(data['Date'], data['PredGPP'], color='orange', label='Simulation')
            sub_ax.scatter(data['Date'], data['TowerGPP'], color='black', s=20, label='Observation',marker='^')   
            if j == 0:
                sub_ax.set_xticklabels([])
                sub_ax.set_ylabel('GPP (gC $m^{-2} 8d^{-1}$)', fontsize=16, fontweight='bold',labelpad=20)
                num = indx[i]
                sub_ax.text(0.02, 0.98, f"({num}) {station}", transform=ax.transAxes, fontsize=20,fontweight='bold', ha="left", va="top")
            elif j==1:
                sub_ax.set_xticklabels([])
                sub_ax.set_ylabel('GPP (gC $m^{-2} 16d^{-1}$)', fontsize=16, fontweight='bold',labelpad=20)
            else:
                sub_ax.tick_params(axis='x', labelsize=18,rotation=0)
                sub_ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[6])) 
                sub_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                xticklabels = sub_ax.get_xticklabels()
                if len(xticklabels) > 8:
                    for i, label in enumerate(xticklabels):
                        if i % 2 == 0:
                            label.set_fontweight('bold')
                        else:
                            label.set_visible(False)
                else:
                    for label in sub_ax.get_xticklabels():
                        label.set_fontweight('bold')
            sub_ax.set_ylabel('GPP (gC $m^{-2} mon^{-1}$)', fontsize=16, fontweight='bold',labelpad=20)
            sub_ax.yaxis.set_label_coords(-0.12, 0.5)
            sub_ax.set_ylim(-5,y_max+10)
            sub_ax.tick_params(axis='y', labelsize=18)
            for label in sub_ax.get_yticklabels():
                label.set_fontweight('bold')
        for k in range(len(stations), len(axes_flat)):
            axes_flat[k].axis('off')
    plt.tight_layout()
    plt.savefig(f'../Figure/LineChart_{ecosystem}.jpg', dpi=1000)
    plt.show()

