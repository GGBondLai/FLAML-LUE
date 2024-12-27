# -*- coding: utf-8 -*-
'''
@File    :   TayorDisgram.py
@Time    :   2024/12/24 15:54:20
@Author  :   GGLai 
@Version :   1.0
@Desc    :   None
'''
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import skill_metrics as sm
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

def custom_agg(group):
    result = pd.Series()
    result['station'] = group['station'].iloc[0]
    result['R2'] = group['R2'].dropna().mean()
    result['R'] = group['R'].dropna().mean()
    result['SD'] = group['SD'].dropna().mean()
    result['nuRMSE'] = group['nuRMSE'].dropna().mean()
    return result
# %%
# specify some styles for the standard deviation
COLS_STD = {
    'grid': "#C0C0C0",
    'tick_labels': '#000000',
    'ticks': '#DDDDDD',
    'title': '#000000'
}
COLS_COR = {
    'grid': '#DDDDDD',
    'tick_labels': '#000000',
    'title': '#000000'
}
# specify some styles for the root mean square deviation
STYLES_RMS = {
    'color': '#AAAADD',
    'linestyle': '--'
}
FONT_FAMILY = 'Times New Roman'
FONT_SIZE = 22
# update figures global properties
plt.rcParams.update({'font.size': FONT_SIZE, 'font.family': FONT_FAMILY})
marksize = 14
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
            data = df.groupby('FLAML',as_index=False).apply(custom_agg).reset_index(drop=True)
            dict_data = {}
            for index, row in data.iterrows():
                scheme_name = row['FLAML']
                values = (row['SD'], row['nuRMSE'], row['R2'])
                dict_data[scheme_name] = values
            MARKERS = {
                "Observed": {
                    "marker": "^",
                    "color_edge": "#000000",
                    "color_face": "#000000",
                    "markersize": marksize
                },
                "FLAML00": {
                    "marker": "+",
                    "color_edge": "#000000",
                    "color_face": "#000000",
                    "markersize": marksize
                },
                "FLAML01": {
                    "marker": "+",
                    "color_edge": "#000000",
                    "color_face": "#00FF00",
                    "markersize": marksize
                },
                "FLAML02": {
                    "marker": "+",
                    "color_edge": "#AA0000",
                    "color_face": "#0000FF",
                    "markersize": marksize
                },
                "FLAML03": {
                    "marker": "+",
                    "color_edge": "#00AA00",
                    "color_face": "#FFFF00",
                    "markersize": marksize
                },
                "FLAML04": {
                    "marker": "+",
                    "color_edge": "#0000AA",
                    "color_face": "#00FFFF",
                    "markersize": marksize
                },
                "FLAML05": {
                    "marker": "+",
                    "color_edge": "#FF00FF",
                    "color_face": "#FF00FF",
                    "markersize": marksize
                },
                "FLAML06": {
                    "marker": "+",
                    "color_edge": "#800000",
                    "color_face": "#800000",
                    "markersize": marksize
                },
                "FLAML07": {
                    "marker": "+",
                    "color_edge": "#008000",
                    "color_face": "#008000",
                    "markersize": marksize
                },
                "FLAML08": {
                    "marker": "+",
                    "color_edge": "#000080",
                    "color_face": "#000080",
                    "markersize": marksize
                },
                "FLAML10": {
                    "marker": "*",
                    "color_edge": "#808000",
                    "color_face": "#808000",
                    "markersize": marksize
                },
                "FLAML11": {
                    "marker": "*",
                    "color_edge": "#800080",
                    "color_face": "#800080",
                    "markersize": marksize
                },
                "FLAML12": {
                    "marker": "*",
                    "color_edge": "#008080",
                    "color_face": "#008080",
                    "markersize": marksize
                },
                "FLAML13": {
                    "marker": "*",
                    "color_edge": "#C0C0C0",
                    "color_face": "#C0C0C0",
                    "markersize": marksize
                },
                "FLAML14": {
                    "marker": "*",
                    "color_edge": "#808080",
                    "color_face": "#808080",
                    "markersize": marksize
                },
                "FLAML15": {
                    "marker": "*",
                    "color_edge": "#FF0000",
                    "color_face": "#FF0000",
                    "markersize": marksize
                },
                "FLAML16": {
                    "marker": "*",
                    "color_edge": "#D4AF37",
                    "color_face": "#FFD700",
                    "markersize": marksize
                },
                "FLAML17": {
                    "marker": "*",
                    "color_edge": "#FFA07A",
                    "color_face": "#FFA07A",
                    "markersize": marksize
                },
                "FLAML18": {
                    "marker": "*",
                    "color_edge": "#A52A2A",
                    "color_face": "#A52A2A",
                    "markersize": marksize
                }
            }
            subplot_data = {
                "title": f"({idx[i]}) {name}",
                "y_label": True,
                "x_label": True,
                "observed": (1.00, 0, 1.00),
                "modeled": dict_data
            }   
            # get subplot object and ensure it will be a square
            # y-axis labels will only appear on leftmost subplot
            ax = axes_flat[i]
            ax.set(adjustable='box', aspect='equal')
            # create the plot with the observed data
            stdev, crmsd, ccoef = subplot_data["observed"]
            sm.taylor_diagram(ax,
                            np.asarray((stdev, stdev)), 
                            np.asarray((crmsd, crmsd)), 
                            np.asarray((ccoef, ccoef)),
                            markercolors = {
                                "face": MARKERS["Observed"]["color_edge"],
                                "edge": MARKERS["Observed"]["color_face"]
                                },
                            markersize = MARKERS["Observed"]["markersize"],
                            markersymbol = MARKERS["Observed"]["marker"],
                            styleOBS = ':',
                            colOBS = MARKERS["Observed"]["color_edge"],
                            alpha = 1.0,
                            axismax = 1.15,
                            titleCOR='off',
                            titleSTD = 'on',
                            tickSTD = [0.5],
                            titleRMS = 'off',
                            showlabelsRMS = 'on',
                            tickRMS = [0, 0.25, 0.5,0.75,1],
                            colRMS = STYLES_RMS['color'],
                            tickRMSangle = 135,
                            styleRMS = STYLES_RMS['linestyle'],
                            colscor = COLS_COR,
                            colsstd = COLS_STD,
                            styleCOR='-',
                            styleSTD='-',
                            colframe='#DDDDDD',
                            labelweight="bold",
                            titlecorshape='linear')
            ax.text(stdev+0.01, 0.08, "Obsv.",
                    verticalalignment="top", horizontalalignment="left",
                    fontsize=FONT_SIZE+2, fontweight="bold")
            ax.text(1.04, 0.8, r'R$^{2}$', color='black', 
                    ha='center', va='center', rotation= 45,
                    fontsize=FONT_SIZE+2, fontweight="bold")
            ax.text(0.03, 1.08, f'({idx[i]}) {name}', color='black', 
                    ha='left', va='center',
                    fontsize=FONT_SIZE+2, fontweight="bold")
            # get rid of variables not to be used anymore
            del stdev, crmsd, ccoef
            # create one overlay for each model marker
            for model_id, (stdev, crmsd, ccoef) in subplot_data["modeled"].items():
                marker = MARKERS[model_id]
                sm.taylor_diagram(ax,
                                np.asarray((stdev, stdev)),
                                np.asarray((crmsd, crmsd)),
                                np.asarray((ccoef, ccoef)),
                                markersymbol = marker["marker"],
                                markercolors = {
                                    "face": marker["color_face"],
                                    "edge": marker["color_edge"]
                                },
                                markersize = marker["markersize"],
                                alpha = 1.0,
                                overlay = 'on',
                                styleCOR = '-',
                                styleSTD = '-',)

                # get rid of variables not to be used anymore
                del model_id, stdev, crmsd, ccoef, marker

            # # set titles (upper, left, bottom)
            # ax.set_title(subplot_data["title"], loc="left", y=1.1)

            # add y label
            if subplot_data["y_label"]:
                ax.set_ylabel("Standard Deviation", fontfamily=FONT_FAMILY,
                            fontsize=FONT_SIZE+2)
            # # add xlabel or hide xticklabels
            # if subplot_data["x_label"]:
            #     ax.set_xlabel("Standard Deviation", fontfamily=FONT_FAMILY,
            #                 fontsize=FONT_SIZE+2)
            # else:
            #     ax.set_xticklabels(ax.get_xticklabels(), color=ax.get_facecolor())

            # just for the peace of mind...
            ax.set_xticklabels(ax.get_xticks(), fontweight='bold', fontsize=FONT_SIZE)
            ax.set_yticklabels(ax.get_yticks(), fontweight='bold', fontsize=FONT_SIZE)

            del subplot_data, ax      
        else:
            axes_flat[i].axis('off')
            ax = axes_flat[i]
            # create legend in the last subplot
            # build legend handles    
            legend_handles = []
            legend_handles.append(mlines.Line2D([], [],
                                color=STYLES_RMS['color'],
                                linestyle=STYLES_RMS['linestyle'],
                                label="RMSE"))

            # for marker_label, marker_desc in list(MARKERS.items())[1:]:
            for marker_label, marker_desc in MARKERS.items():
                marker = mlines.Line2D([], [], 
                                    marker=marker_desc["marker"],
                                    markersize=marker_desc["markersize"],
                                    markerfacecolor=marker_desc["color_face"],
                                    markeredgecolor=marker_desc["color_edge"],
                                    linestyle='None',
                                    label=marker_label)
                legend_handles.append(marker)
                del marker_label, marker_desc, marker
            # create legend and free memory
            ax.legend(
                handles=legend_handles,
                loc="center",
                ncol=2,
                fontsize=20,
                markerscale=2,
                handlelength=2.5,
                columnspacing=1.2,
                labelspacing=1.2,
                frameon=True,
                borderpad=1.2,
                handletextpad=1
            )

            del ax, legend_handles

    # avoid some overlapping
    plt.tight_layout()
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.03, top=0.95, hspace=0.07)
    plt.savefig(f'../Figure/Tayor{key}.jpg', dpi=1000, facecolor='w')
    # # Show plot and close it
    # None if args.no_show else plt.show()
    plt.show()
