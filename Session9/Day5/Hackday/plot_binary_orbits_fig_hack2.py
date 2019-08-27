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

# use LaTeX fonts in the plot
pyplot.rc('text', usetex=True)
pyplot.rc('font', family='serif')

def kde_sklearn(data, grid, bandwidth = 1.0, **kwargs):
    kde_skl = KernelDensity(bandwidth = bandwidth, **kwargs)
    kde_skl.fit(data[:, numpy.newaxis])
    log_pdf = kde_skl.score_samples(grid[:, numpy.newaxis]) # sklearn returns log(density)

    return numpy.exp(log_pdf)

numpy.random.seed(0)

path="/Users/jrobinson/xq1_grav_cloud/binary_stability/orbit_results/orbit_results_plots"
fname_df="df_plot_100_all_stable.txt"
embolden=0

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

ax1.set_xlabel("$(m_1+m_2)/M_\\mathrm{{c}}$")
ax1.set_ylabel("$i_{{\\mathrm{{bin}}}}~(\\mathrm{{degrees}})$")
# ax2.set_xlabel("(m1+m2)/Mc")
ax2.set_ylabel("$e_{{\\mathrm{{bin}}}}$")

print len(df)
print len(df[~numpy.isnan(df['m1(kg)'])])
df=df[~numpy.isnan(df['m1(kg)'])] # drop entries without binaries
print list(df)

df['m2/m1']=numpy.array(df['m2(kg)']).astype(float)/numpy.array(df['m1(kg)']).astype(float) # move this to a separate script

if embolden==0:
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

else:
    print "class iii systems"
    df_iii=df[(df['m2/m1']>0.1) & ((df['m1(kg)']+df['m2(kg)'])/df['M_tot(kg)']>0.1)]

    int_list_iii=numpy.array(df_iii['M_tot(kg)'])
    for i,M_tot in enumerate(numpy.unique(numpy.array(df_iii['M_tot(kg)']).astype(float))):
        int_mask=[int_list_iii==M_tot]
        int_list_iii[int_mask]=[i]*len(int_mask)
    print int_list_iii
    int_list_iii=int_list_iii.astype(int)
    col_list_iii=[]
    marker_list_iii=[]
    for i in int_list_iii:
        col_list_iii.append(pf.pyplot_colours[i])
        marker_list_iii.append(markers[i])
    print col_list_iii
    print marker_list_iii

    a_bin_iii=numpy.array(df_iii['a(m)'])
    e_bin_iii=numpy.array(df_iii['e'])
    I_bin_iii=numpy.array(df_iii['I(rad)'])

    m1_iii=numpy.array(df_iii['m1(kg)'])
    m2_iii=numpy.array(df_iii['m2(kg)'])
    m2m1_iii=numpy.array(df_iii['m2/m1'])
    Mtot_iii=numpy.array(df_iii['M_tot(kg)'])

    x_data_1_iii=(m1_iii+m2_iii)/Mtot_iii
    y_data_1_iii=numpy.degrees(I_bin_iii)

    x_data_2_iii=(m1_iii+m2_iii)/Mtot_iii
    y_data_2_iii=e_bin_iii

    # Use marker size to represent mass ratio
    marker_size_factor=50.0
    size_min=10
    marker_size=(m2m1_iii*marker_size_factor)+size_min
    print marker_size

    for i in range(len(x_data_1_iii)):
        # label="M_c={:.2e}kg".format(Mtot[i])
        label=None

        ax2.scatter(x_data_2_iii[i],y_data_2_iii[i],
        edgecolors=col_list_iii[i],facecolors='none',
        marker=marker_list_iii[i],
        s=marker_size[i],
        alpha=1.0)

        ax1.scatter(x_data_1_iii[i],y_data_1_iii[i],
        edgecolors=col_list_iii[i],facecolors='none',
        marker=marker_list_iii[i],
        s=marker_size[i],
        alpha=1.0,
        label=label)

    df_other=df[~((df['m2/m1']>0.1) & ((df['m1(kg)']+df['m2(kg)'])/df['M_tot(kg)']>0.1))]
    print "other systems = {}".format(len(df_other))

    int_list_other=numpy.array(df_other['M_tot(kg)'])
    for i,M_tot in enumerate(numpy.unique(numpy.array(df_other['M_tot(kg)']).astype(float))):
        int_mask=[int_list_other==M_tot]
        int_list_other[int_mask]=[i]*len(int_mask)
    print int_list_other
    int_list_other=int_list_other.astype(int)
    col_list_other=[]
    marker_list_other=[]
    for i in int_list_other:
        col_list_other.append(pf.pyplot_colours[i])
        marker_list_other.append(markers[i])
    print col_list_other
    print marker_list_other

    a_bin_other=numpy.array(df_other['a(m)'])
    e_bin_other=numpy.array(df_other['e'])
    I_bin_other=numpy.array(df_other['I(rad)'])

    m1_other=numpy.array(df_other['m1(kg)'])
    m2_other=numpy.array(df_other['m2(kg)'])
    m2m1_other=numpy.array(df_other['m2/m1'])
    Mtot_other=numpy.array(df_other['M_tot(kg)'])

    x_data_1_other=(m1_other+m2_other)/Mtot_other
    y_data_1_other=numpy.degrees(I_bin_other)

    x_data_2_other=(m1_other+m2_other)/Mtot_other
    y_data_2_other=e_bin_other

    # Use marker size to represent mass ratio
    marker_size_factor=50.0
    size_min=10
    marker_size=(m2m1_other*marker_size_factor)+size_min
    print marker_size

    for i in range(len(x_data_1_other)):
        # label="M_c={:.2e}kg".format(Mtot[i])
        label=None

        ax2.scatter(x_data_2_other[i],y_data_2_other[i],
        edgecolors=col_list_other[i],facecolors='none',
        marker=marker_list_other[i],
        s=marker_size[i],
        alpha=0.25)

        ax1.scatter(x_data_1_other[i],y_data_1_other[i],
        edgecolors=col_list_other[i],facecolors='none',
        marker=marker_list_other[i],
        s=marker_size[i],
        alpha=0.25,
        label=label)

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
    temp_points.append(ax2.scatter(0,0,s=s,c="None",edgecolors="k",label="$\\frac{{m_2}}{{m_1}}={:.2f}$".format(leg_vals[i])))
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

# pyplot.show()
pyplot.close()
