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

outpath = '/nobackup/rossby26/users/sm_fuxwa/AI/Emilia_Romagna/statistic_figs/TaylorDiagram/'
outfile = 'TaylorDiagram_ECE.png'

# Reference std
stdrefs = dict(ECEHist=6.71,
               ECEFutMC=6.89,
               ECEHistFutMC=6.71,
              )

# Sample std,rho: Be sure to check order and that correct numbers are placed!
samples = dict(ECEHist=[[6.82, 0.98, "HCLIM12"],
                       [6.50, 0.97, "CNN"],
                       [6.07, 0.95, "SRGAN"]],
               ECEFutMC=[[6.84, 0.98, "HCLIM12"],
                       [6.50, 0.97, "CNN"],
                       [6.05, 0.95, "SRGAN"]],
               ECEHistFutMC=[[6.82, 0.98, "HCLIM12"],
                       [6.50, 0.97, "CNN"],
                       [5.93, 0.95, "SRGAN"]],
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

#sys.exit()
# Colormap (see http://www.scipy.org/Cookbook/Matplotlib/Show_colormaps)
colors = PLT.matplotlib.cm.Set1(NP.linspace(0,1,len(samples['ECEHist'])))

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

rects = {'ECEHist':131,
         'ECEFutMC':132,
         'ECEHistFutMC':133,
        }

fig = PLT.figure(figsize=(11,8))
fig.suptitle("Precipitations", size='x-large')

for exp in rects.keys():

    print('exp', exp)
    dia = TaylorDiagram(stdrefs[exp], fig=fig, rect=rects[exp],
                        label='Reference')

    dia.ax.plot(x95,y95,color='k')
    dia.ax.plot(x99,y99,color='k')

    # Add samples to Taylor diagram
    for i,(stddev,corrcoef,name) in enumerate(samples[exp]):
        dia.add_sample(stddev, corrcoef,
                       marker='$%d$' % (i+1), ms=10, ls='',
                       #mfc='k', mec='k', # B&W
                       mfc=colors[i], mec=colors[i], # Colors
                       label=name)

    # Add RMS contours, and label them
    contours = dia.add_contours(levels=5, colors='0.5') # 5 levels
    dia.ax.clabel(contours, inline=1, fontsize=10, fmt='%.1f')
    # Tricky: ax is the polar ax (used for plots), _ax is the
    # container (used for layout)
    dia._ax.set_title(exp.capitalize())

# Add a figure legend and title. For loc option, place x,y tuple inside [ ].
# Can also use special options here:
# http://matplotlib.sourceforge.net/users/legend_guide.html

fig.legend(dia.samplePoints,
           [ p.get_label() for p in dia.samplePoints ],
           numpoints=1, prop=dict(size='small'), loc='center')

fig.tight_layout()

PLT.savefig(outpath + '/' + outfile)
PLT.show()

