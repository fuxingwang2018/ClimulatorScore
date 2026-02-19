#!/usr/bin/env python

__version__ = "Time-stamp: <2018-12-06 11:55:22 ycopin>"
__author__ = "Yannick Copin <yannick.copin@laposte.net>"

"""
Example of use of TaylorDiagram. Illustration dataset courtesy of Michael
Rawlins.

Rawlins, M. A., R. S. Bradley, H. F. Diaz, 2012. Assessment of regional climate
model simulation estimates over the Northeast United States, Journal of
Geophysical Research (2012JGRD..11723112R).
"""

from taylorDiagram import TaylorDiagram
import numpy as NP
import matplotlib.pyplot as PLT
import sys

outpath = '/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/statistic_figs/TaylorDiagram/'
variable = "tas"  #tas
outfile = 'TaylorDiagram_ECE_wt_worog_' + str(variable) + '.png'

if variable == 'tas':
    fig_suptitle = '2-m air temperature'
    # Reference std
    stdrefs = dict(ECEHI2HI=6.71,
               ECEMC2MC=6.89,
               ECEHM2HI=6.71,
               ECEHM2MC=6.89,
               ECEHI2MC=6.89,
               ECEMC2HI=6.71,
              )

    # Sample std,rho: Be sure to check order and that correct numbers are placed!
    # LNOISE0.1 WT WOROG
    #samples = dict(ECEHI2HI=[[6.82, 0.98, "HCLIM12"],
    #                   [6.50, 0.97, "CNN"],
    #                   [6.02, 0.95, "SRGAN"]],
    #           ECEMC2MC=[[6.84, 0.98, "HCLIM12"],
    #                   [6.50, 0.97, "CNN"],
    #                   [6.15, 0.95, "SRGAN"]],
    #           ECEHM2HI=[[6.82, 0.98, "HCLIM12"],
    #                   [6.50, 0.97, "CNN"],
    #                   [5.93, 0.95, "SRGAN"]],
    #           ECEHM2MC=[[6.82, 0.98, "HCLIM12"],
    #                   [6.59, 0.97, "CNN"],
    #                   [5.39, 0.95, "SRGAN"]],
    #           )
    # WT WOROG
    samples = dict(ECEHI2HI=[[6.82, 0.98, "HCLIM12"],
                       [6.50, 0.97, "CNN"],
                       [6.56, 0.99, "SRGAN"]],
               ECEMC2MC=[[6.84, 0.98, "HCLIM12"],
                       [6.50, 0.97, "CNN"],
                       [6.71, 0.99, "SRGAN"]],
               ECEHM2HI=[[6.82, 0.98, "HCLIM12"],
                       [6.50, 0.97, "CNN"],
                       [6.29, 0.98, "SRGAN"]],
               ECEHM2MC=[[6.84, 0.98, "HCLIM12"],
                       [6.59, 0.97, "CNN"],
                       [6.25, 0.99, "SRGAN"]],
               ECEHI2MC=[[6.84, 0.98, "HCLIM12"],
                       [0., 0., "CNN"],
                       [6.00, 0.90, "SRGAN"]],
               ECEMC2HI=[[6.82, 0.98, "HCLIM12"],
                       [0.0, 0.0, "CNN"],
                       [6.84, 0.84, "SRGAN"]],
               )
if variable == 'pr':
    fig_suptitle = 'precipitation'
    # Reference std
    stdrefs = dict(ECEHI2HI=9.02,
               ECEMC2MC=9.88,
               ECEHM2HI=9.02,
               ECEHM2MC=9.88,
               ECEHI2MC=9.88,
               ECEMC2HI=9.02,
              )

    # Sample std,rho: Be sure to check order and that correct numbers are placed!
    # LNOISE_0.1 WP WOROG
    #samples = dict(ECEHI2HI=[[7.36, 0.53, "HCLIM12"],
    #                   [0.0, 0.0, "CNN"],
    #                   [5.47, 0.45, "SRGAN"]],
    #           ECEMC2MC=[[8.41, 0.54, "HCLIM12"],
    #                   [0.0, 0.0, "CNN"],
    #                   [7.17, 0.54, "SRGAN"]],
    #           ECEHM2HI=[[7.36, 0.45, "HCLIM12"],
    #                   [0.0, 0.0, "CNN"],
    #                   [5.97, 0.47, "SRGAN"]],
    #           ECEHM2MC=[[8.41, 0.54, "HCLIM12"],
    #                   [0.0, 0.0, "CNN"],
    #                   [7.06, 0.47, "SRGAN"]],
    #           )
    # LNOISE1.0 (default) WP WOROG
    samples = dict(ECEHI2HI=[[7.36, 0.53, "HCLIM12"],
                       [0.0, 0.0, "CNN"],
                       [6.29, 0.56, "SRGAN"]],
               ECEMC2MC=[[8.41, 0.54, "HCLIM12"],
                       [0.0, 0.0, "CNN"],
                       [8.23, 0.54, "SRGAN"]],
               ECEHM2HI=[[7.36, 0.45, "HCLIM12"],
                       [0.0, 0.0, "CNN"],
                       [5.35, 0.58, "SRGAN"]],
               ECEHM2MC=[[8.41, 0.54, "HCLIM12"],
                       [0.0, 0.0, "CNN"],
                       [5.75, 0.59, "SRGAN"]],
               ECEHI2MC=[[8.41, 0.54, "HCLIM12"],
                       [0.0, 0.0, "CNN"],
                       [6.72, 0.57, "SRGAN"]],
               ECEMC2HI=[[7.36, 0.45, "HCLIM12"],
                       [0.0, 0.0, "CNN"],
                       [7.09, 0.55, "SRGAN"]],
               )

#samples = {'ECEHist':  {'CNN': {'MBE': 1.2, 'RMSE': 1.56, 'CORR': 0.97}},
#                       {'SRGAN': {'MBE': -0.21, 'RMSE': 2.22, 'CORR': 0.97}},
#           'ECEFutMC': {'CNN': {'MBE': 1.2, 'RMSE': 1.56, 'CORR': 0.97}},
#                       {'SRGAN': {'MBE': -0.3, 'RMSE': 2.45, 'CORR': 0.97}},
#           'ECEHistFutMC': {'CNN': {'MBE': 1.22, 'RMSE': 1.57, 'CORR': 0.97}},
#                           {'SRGAN': {'MBE': -0.04, 'RMSE': 2.4, 'CORR': 0.97}},
#           }

print('samples', samples)
print('samples.keys()', samples.keys())

# Colormap (see http://www.scipy.org/Cookbook/Matplotlib/Show_colormaps)
colors = PLT.matplotlib.cm.Set1(NP.linspace(0,1,len(samples['ECEHI2HI'])))

# Here set placement of the points marking 95th and 99th significance
# levels. For more than 102 samples (degrees freedom > 100), critical
# correlation levels are 0.195 and 0.254 for 95th and 99th
# significance levels respectively. Set these by eyeball using the
# standard deviation x and y axis.


x95 = [0.05, 13.9] # [0.01, 0.68]For Prcp, this is for 95th level (r = 0.195)
y95 = [0.0, 71.0]  # [0.0, 3.45]
x99 = [0.05, 19.0] # [0.01, 0.95] For Prcp, this is for 99th level (r = 0.254)
y99 = [0.0, 70.0]  # [0.0, 3.45]

#rects = dict(Stockholm_EV1_PGW1=231,
#             Stockholm_EV2_PGW1=232,
#             Stockholm_EV3_PGW1=233,
#             NorrLink_EV1_PGW1=234,
#             NorrLink_EV2_PGW1=235,
#             NorrLink_EV3_PGW1=236)

rects = {'ECEHI2HI':231,
         'ECEMC2MC':232,
         'ECEHM2HI':233,
         'ECEHM2MC':234,
         'ECEHI2MC':235,
         'ECEMC2HI':236,
        }

fig = PLT.figure(figsize=(27,20))
fig.suptitle(str(fig_suptitle), fontsize = 24 ) #size='x-large')
symbols = ['^', 'o', 's', 'D', 'v', '*', 'p']

i = 0
for exp in rects.keys():
    i+=1
    print('exp', exp)
    dia = TaylorDiagram(stdrefs[exp], fig=fig, rect=rects[exp],
                        label='Ref-HCLIM3')

    dia.ax.plot(x95,y95,color='k')
    dia.ax.plot(x99,y99,color='k')

    # Add samples to Taylor diagram
    for i,(stddev,corrcoef,name) in enumerate(samples[exp]):
        dia.add_sample(stddev, corrcoef,
                       #marker='$%d$' % (i+1), ms=16, ls='',
                       #mfc='k', mec='k', # B&W
                       marker=symbols[i % len(symbols)],
                       ms=16, ls='',
                       mfc=colors[i], mec=colors[i], # Colors
                       label='a-' + name)

    # Add RMS contours, and label them
    contours = dia.add_contours(levels=5, colors='0.5') # 5 levels
    dia.ax.clabel(contours, inline=1, fontsize=18, fmt='%.1f')
    # Tricky: ax is the polar ax (used for plots), _ax is the
    # container (used for layout)
    #dia._ax.set_title(exp.capitalize())
    dia._ax.set_title(exp, fontsize = 24)

# Add a figure legend and title. For loc option, place x,y tuple inside [ ].
# Can also use special options here:
# http://matplotlib.sourceforge.net/users/legend_guide.html
# Remove the line from each sample point
for p in dia.samplePoints:
    p.set_linestyle('') # This hides the line but keeps the marker

fig.legend(dia.samplePoints,
           [ p.get_label() for p in dia.samplePoints ],
           numpoints=1,  loc='upper right', fontsize=24)
           #numpoints=1, prop=dict(size='small'), loc='center')

fig.tight_layout()

PLT.savefig(outpath + '/' + outfile)
PLT.show()

"""
from file_reader import FileReader

#file_in = "/home/sm_fuxwa/Figures/BRIGHT/eval_obs_grid/temperature_domain_average_time_series_original_NGCD_EOBS25_mixed.statis.yaml"
file_in = '/home/sm_fuxwa/Figures/BRIGHT/eval_obs_grid/temperature_itownonly_domain_average_time_series_original_EOBS27_regrid_NGCD_regrid_mixed.statis.yaml'
filereader = FileReader(file_in)
data = filereader.Read_Yaml()
#print('data', data)

samples={}
stdrefs={}

sample_statistics = ['STD', 'CORR', 'MODEL']
stdref_statistics = 'STD'

for key, values in data.items():
    statis = []
    #print('key', key)
    for idata in data[key][1:]:
        #print('idata', idata)
        istatis = []
        for statis_name in sample_statistics:
            istatis.append((idata[statis_name]))
        #print('istatis', istatis)
        statis.append(istatis)
    samples[key] = statis

    stdrefs[key] = data[key][0][stdref_statistics]
"""

"""
for key, values in d.items():
    for idf_merged in d[key]:
        statis.append(list(idf_merged.values()))
    samples[key] = statis
"""
