# -*- coding: utf-8 -*-
'''
@File    :   BoxDiagram.py
@Time    :   2024/12/24 19:42:48
@Author  :   GGLai 
@Version :   1.0
@Desc    :   None
'''

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
# %%

fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize=(15, 15), dpi = 1000)
axes_flat = axes.flatten()
dict = {'Forest':['Forest','ALF','CBF','QYF'],
        'Grass':['Grass','DLG','DXG','HBG_S01'],
        'Crop':['Crop','JZA','YCA']}
idx = ['a','b','c','d']
keys = dict.keys()
for key in keys:
    fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize=(15, 15), dpi = 1000)
    axes_flat = axes.flatten()
    variables = dict[key]
    for i in range(0, 4, 1):
        if i < len(variables):
            name = variables[i]
            if i == 0:
                df = pd.read_excel(f'../Output/EvaluationIndex_{key}.xlsx')
                df = df.iloc[:, :12]
                df['station']=df['ecosystem']
            else:
                df = pd.read_excel('../Output/EvaluationIndex_station.xlsx')
                df = df.iloc[:, :11]
                df[df['station']==name]
            df_daily = df[df['time']=='daily']
            df_8days = df[df['time']=='8days']
            df_16days = df[df['time']=='16days']
            df_monthly = df[df['time']=='monthly']
            for f in [df_daily, df_8days, df_16days,df_monthly]:
                ax = axes_flat[i]
                pvalue=stats.mannwhitneyu(df['new'],df['old'],alternative='two-sided')[1]
                plt.boxplot([f,f],labels=['old','new'],widths=0.3,
                            patch_artist=True,
                            boxprops={'color':'black','facecolor':'lightgrey'},
                            flierprops={'marker':'+','markerfacecolor':'black','color':'lightgrey'},
                            medianprops={"linestyle":'-','color':'black'})


            
        else:
            axes_flat[i].axis('off')
            continue

# %%
import matplotlib.pyplot as plt
import pandas as pd
import vedo
from vedo import np, settings, Axes, Brace, Line, Ribbon, show
from vedo.pyplot import whisker
from vedo import Plotter
def get_p_value_label(p_value):
    # 根据 p 值判断标记
    if p_value < 0.0001:
        label = '****'
    elif p_value < 0.001:
        label = '***'
    elif p_value < 0.01:
        label = '**'
    elif p_value <= 0.05:
        label = '*'
    else:
        label = 'ns'
    text = {'comment': label, 's': 0.7, 'style': '['}
    return text

fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize=(15, 15), dpi = 1000)
axes_flat = axes.flatten()
station = ['ALF','CBF','QYF']
plotter = vedo.Plotter()
for i in range(0, 4, 1):
    if i==3:
        axes_flat[i].axis('off')
        continue
    else:
        ax = axes_flat[i]
        df = pd.read_excel('../Output/EvaluationIndex_station.xlsx')
        data = df[df['station']==station[i]]
        df_daily = data[data['time']=='daily']['R2']
        df_8days = data[data['time']=='8days']['R2']
        df_16days = data[data['time']=='16days']['R2']
        df_monthly = data[data['time']=='monthly']['R2']
        data_to_plot = [df_daily, df_8days, df_16days, df_monthly]

        boxprops = {'facecolor': 'skyblue', 'edgecolor': 'black'}
        flierprops = {'markerfacecolor': 'red', 'marker': 'o', 'markersize': 5}
        medianprops = {'color': 'black', 'linewidth': 2}
        
        bp = ax.boxplot(data_to_plot, patch_artist=True,
                        boxprops=boxprops, 
                        flierprops=flierprops,
                        medianprops=medianprops)
        ax.set_title(f"Station {station[i]}", fontsize=14)
        ax.set_xticklabels(['Daily', '8 Days', '16 Days', 'Monthly'], fontsize=12)
        ax.set_ylabel('R2 Value', fontsize=12)

        t_stat, p_value = stats.ttest_ind(df_daily, df_8days)
        if p_value < 0.0001:
            ax.text(1.5, np.max(data_to_plot) + 0.05, "****", ha='center', fontsize=14, color='black')
        elif p_value < 0.001:
            ax.text(1.5, np.max(data_to_plot) + 0.05, "***", ha='center', fontsize=14, color='black')
        elif p_value < 0.01:
            ax.text(1.5, np.max(data_to_plot) + 0.05, "**", ha='center', fontsize=14, color='black')
        elif p_value <= 0.05:
            ax.text(1.5, np.max(data_to_plot) + 0.05, "*", ha='center', fontsize=14, color='black')
        else:
            ax.text(1.5, np.max(data_to_plot) + 0.05, "ns", ha='center', fontsize=14, color='black')
plt.tight_layout()
plt.show()
# %%
from vedo import np, settings, Axes, Brace, Line, Ribbon, show
from vedo.pyplot import whisker

settings.default_font = "Theemim"
timescale = ['daily','8days','16days','monthly']
dict = {'Forest':['Forest','ALF','CBF','QYF'],
        'Grass':['Grass','DLG','DXG','HBG_S01'],
        'Crop':['Crop','JZA','YCA']}
df = pd.read_excel('../Output/EvaluationIndex_station.xlsx')
keys = dict.keys()
for key in keys:
    for i in range(0, 4, 1):
        if i < len(variables):
            name = variables[i]
            if i == 0:
                df = pd.read_excel(f'../Output/EvaluationIndex_{key}.xlsx')
                df = df.iloc[:, :12]
                df['station']=df['ecosystem']
            else:
                df = pd.read_excel('../Output/EvaluationIndex_station.xlsx')
                df = df.iloc[:, :11]
                df[df['station']==name]
            ws = []
            for j in range(4):
                xval = j*2
                time = timescale[j]
                data = df[df['time']==time]['R2']
                w = whisker(data, bc=i, s=0.5).x(xval)
                ws.append(w)
            bra1 = Brace([0, 3],[2, 3], comment='*~*', s=0.7, style='[')
            bra2 = Brace([4,-1],[8,-1], comment='ET (mmd^-1 ) > mean', s=0.4)
            axes = Axes(xrange=[-1,9],
                        yrange=[-3,5],
                        htitle=':beta_c  expression: change in time',
                        xtitle=' ',
                        ytitle='Level of :ET protein in \muM/l',
                        x_values_and_labels=[(0,'ET^A\n at t=1h'),
                                            (4,'ET^B\n at t=2h'),
                                            (8,'ET^C\n at t=4h'),
                                            ],
                        xlabel_size=0.02,
                        xygrid=False,
                        )
            show(ws, bra1, bra2, line, band, __doc__, axes, zoom=1.3)
        else:
            continue
# %%
import matplotlib.pyplot as plt
from matplotlib import rcParams
from vedo import np, settings, Axes, Brace, Line, Ribbon, show
from vedo.pyplot import whisker
rcParams['font.family'] = 'Times New Roman'
settings.default_font = "Theemim"
timescale = ['daily','8days','16days','monthly']
df = pd.read_excel('../Output/EvaluationIndex_station.xlsx')
df = df.iloc[:, :11]
df[df['station']=='CBF']
ws = []
def com(p_value):
    if p_value < 0.0001:
        label = '****'
    elif p_value < 0.001:
        label = '***'
    elif p_value < 0.01:
        label = '*~*'
    elif p_value <= 0.05:
        label = '*'
    else:
        label = 'ns'
    return label
for j in range(4):
    xval = j*0.2
    time = timescale[j]
    data = df[df['time']==time]['R2']
    w = whisker(data, bc=i, s=0.1).x(xval)
    ws.append(w)
bra1 = Brace([0, 0.97],[0.2, 0.97], comment='*~***', s=0.7, style='[')
axes = Axes(xrange=[-0.2,0.8],
            yrange=[0,1.1],
            htitle='(c) CBF',
            xtitle=' ',
            ytitle=rf"R^2",
            x_values_and_labels=[(0,'daily'),
                                (0.2,'8days'),
                                (0.4,'16days'),
                                (0.6,'monthly')],
            xlabel_size=0.02,
            xygrid=False,
            )
show(ws, bra1, axes, zoom=1.3)
plt.savefig('BoxDisgram_CBF.jpg', dpi=1000, bbox_inches='tight')
# %%
