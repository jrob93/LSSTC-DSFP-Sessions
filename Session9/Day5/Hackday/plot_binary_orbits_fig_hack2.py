'''
Plot to investigate trends between inclination and eccentricity
'''

from matplotlib import rc
rc('font',**{'size':9})

import pandas as pd
import numpy
import py_func as pf
import matplotlib.pyplot as pyplot
import matplotlib.gridspec as gridspec
import os
import matplotlib.ticker as ticker
import astropy.stats.bayesian_blocks as bb
from scipy import stats

from sklearn.neighbors import KernelDensity

def kde_sklearn(data, grid, bandwidth = 1.0, **kwargs):
    kde_skl = KernelDensity(bandwidth = bandwidth, **kwargs)
    kde_skl.fit(data[:, numpy.newaxis])
    log_pdf = kde_skl.score_samples(grid[:, numpy.newaxis]) # sklearn returns log(density)

    return numpy.exp(log_pdf)

numpy.random.seed(0)

path="/Users/jrobinson/xq1_grav_cloud/binary_stability/orbit_results/orbit_results_plots"
fname_df="df_plot_100_all_stable.txt"

markers=['^','s','o']

df=pd.read_csv("{}/{}".format(path,fname_df),sep="\t",index_col=0) # orbits post selection

# only plot the simple binary systems
df=df[df['N_sys']==2]

# # Only include f values 3,10,30
# df=df[(df['f']<100.0) & (df['f']>1.0)]

fig = pyplot.figure()

pc_tex=0.16605 # latex pc in inches
text_width=39.0*pc_tex
column_sep=2.0*pc_tex
column_width=(text_width-column_sep)/2.0
s_x=1.0
s_y=1.0
x_len=((text_width/2.0)+(1.5*pc_tex))*s_x
y_len=(x_len)*s_y
print "size: {}x{} inches".format(x_len,y_len)
fig.set_size_inches(x_len,y_len)

gs = gridspec.GridSpec(2,1,hspace=0.0)
ax1 = pyplot.subplot(gs[1,0])
ax2 = pyplot.subplot(gs[0,0])

ax2.tick_params(labelbottom=False)

print len(df)
print len(df[~numpy.isnan(df['m1(kg)'])])
df=df[~numpy.isnan(df['m1(kg)'])] # drop entries without binaries
print list(df)

df['m2/m1']=numpy.array(df['m2(kg)']).astype(float)/numpy.array(df['m1(kg)']).astype(float) # move this to a separate script

int_list=numpy.array(df['M_tot(kg)'])
for i,M_tot in enumerate(numpy.unique(numpy.array(df['M_tot(kg)']).astype(float))):
    int_mask=[int_list==M_tot]
    int_list[int_mask]=[i]*len(int_mask)
print int_list
int_list=int_list.astype(int)
col_list=[]
marker_list=[]
for i in int_list:
    col_list.append(pf.pyplot_colours[i])
    marker_list.append(markers[i])
print col_list
print marker_list

a_bin=numpy.array(df['a(m)'])
e_bin=numpy.array(df['e'])
I_bin=numpy.array(df['I(rad)'])

m1=numpy.array(df['m1(kg)'])
m2=numpy.array(df['m2(kg)'])
m2m1=numpy.array(df['m2/m1'])
Mtot=numpy.array(df['M_tot(kg)'])

# x_data_1=e_bin
# y_data_1=numpy.degrees(I_bin)
x_data_1=(m1+m2)/Mtot
# x_data_1=numpy.log10((m1+m2)/Mtot)
y_data_1=numpy.degrees(I_bin)

x_data_2=(m1+m2)/Mtot
# x_data_2=numpy.log10((m1+m2)/Mtot)
y_data_2=e_bin

ax1.set_xlabel("(m1+m2)/Mc")
ax1.set_ylabel("I(degrees)")
# ax2.set_xlabel("(m1+m2)/Mc")
ax2.set_ylabel("e")

# Use marker size to represent mass ratio
marker_size_factor=50.0
size_min=10
marker_size=(m2m1*marker_size_factor)+size_min
print marker_size

for i in range(len(x_data_1)):
    # label="M_c={:.2e}kg".format(Mtot[i])
    label=None

    ax2.scatter(x_data_2[i],y_data_2[i],
    edgecolors=col_list[i],facecolors='none',
    marker=marker_list[i],
    s=marker_size[i],
    alpha=1.0)

    ax1.scatter(x_data_1[i],y_data_1[i],
    edgecolors=col_list[i],facecolors='none',
    marker=marker_list[i],
    s=marker_size[i],
    alpha=1.0,
    label=label)

    # ax2.scatter(x_data_2[i],y_data_2[i],
    # edgecolors=col_list[i],facecolors='none',
    # marker=marker_list[i],
    # alpha=1.0)
    #
    # ax1.scatter(x_data_1[i],y_data_1[i],
    # edgecolors=col_list[i],facecolors='none',
    # marker=marker_list[i],
    # alpha=1.0,
    # label=label)

ax1.axhline(90,color="k",alpha=0.2,zorder=0)

padding_y=0.1
padding_x=padding_y
# ax1.set_xlim(numpy.amin(x_data)-padding_x,0.0+padding_x)
# # ax1.set_ylim(numpy.amin(y_data)-padding_y,0.0+padding_y)
# # ax4.set_xlim(0.0-padding_x,1.0+padding_x)
# ax4.set_ylim(0.0-padding_y,numpy.amax(10**y_data)+padding_y)

# Add points showing marker size for the legend
# marker_size_leg=numpy.array([0.05,0.25,1.0])*marker_size_factor
leg_vals=numpy.array([0.01,0.5,1.0])
marker_size_leg=(leg_vals*marker_size_factor)+size_min
temp_points=[]
for i,s in enumerate(marker_size_leg):
    # ax1.scatter(0,0,s=s,color=pf.pyplot_colours[0],label="a/R_hill = {}".format(s/marker_size_factor))
    temp_points.append(ax2.scatter(0,0,s=s,c="None",edgecolors="k",label="m2/m1={:.2f}".format(leg_vals[i])))
# make legend with these points representing size

# Remove duplicates from legend
handles, labels = ax2.get_legend_handles_labels()
from collections import OrderedDict
by_label = OrderedDict(zip(labels, handles))
ax2.legend(by_label.values(), by_label.keys(),loc='upper right', bbox_to_anchor=(0.9, 1.0),prop={'size': 6})

# after legend is made remove the points
for tp in temp_points:
    tp.remove()

pyplot.tight_layout()

#save the figure
script_name=os.path.basename(__file__).split('.')[0]
picname="{}_{}.png".format(script_name,fname_df.split("/")[-1].split(".")[0])
print "save {}".format(picname)
pyplot.savefig(picname,bbox_inches='tight',pad_inches=0.0)

picname="{}_{}.pdf".format(script_name,fname_df.split("/")[-1].split(".")[0])
print "save {}".format(picname)
pyplot.savefig(picname,bbox_inches='tight',pad_inches=0.0)

pyplot.show()
# pyplot.close()
