'''
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

path="/Users/jrobinson/xq1_grav_cloud/binary_stability/orbit_results/orbit_results_plots"
fname_dfs=["df_plot_100_all_stable.txt"]
fname_df=fname_dfs[0]

df=pd.read_csv("{}/{}".format(path,fname_df),sep="\t",index_col=0) # orbits post selection

markers=['^','s','o']

fig = pyplot.figure()

# pc_tex=0.16605 # latex pc in inches
# text_width=39.0*pc_tex
# column_sep=2.0*pc_tex
# column_width=(text_width-column_sep)/2.0
# x_len=text_width+(column_sep)+(2.2*pc_tex) # add an additional couple points to fit
# y_len=x_len/1.5
# print "size: {}x{} inches".format(x_len,y_len)
# fig.set_size_inches(x_len,y_len)

gs = gridspec.GridSpec(1,1)
ax1 = pyplot.subplot(gs[0,0])

print len(df)
print len(df[~numpy.isnan(df['m1(kg)'])])
df=df[~numpy.isnan(df['m1(kg)'])] # drop entries without binaries
print list(df)

# only plot the simple binary systems
df=df[df['N_sys']==2]
print len(df)

# for i,M_tot in enumerate(numpy.unique(numpy.array(df['M_tot(kg)']).astype(float))):
#     df2=df[df['M_tot(kg)']==M_tot]
#     print "mass = {}kg, number of binaries = {}".format(M_tot,len(df2))
#     rho=numpy.array(df2['rho(kgm-3)']).astype(float)
#     f=numpy.array(df2['f']).astype(float)
#     R1=((3.0*numpy.array(df2['m1(kg)']).astype(float))/(4.0*numpy.pi*rho))**(1.0/3.0)
#     R2=((3.0*numpy.array(df2['m2(kg)']).astype(float))/(4.0*numpy.pi*rho))**(1.0/3.0)
#     print R1
#     print f
#     R1=R1*f
#     print R1
#     R2=R2*f
#     R2R1=R2/R1
#     a_bin=numpy.array(df2['a(m)']).astype(float)
#     R_hill=numpy.array(df2['R_hill(m)']).astype(float)
#
#     size_1=a_bin/(R1+R2)
#     print size_1
#     print len(size_1[size_1<10.0])
#     print numpy.median(size_1),numpy.mean(size_1)
#
#     ax1.hist(size_1,bins=100,alpha=0.5,color=pf.pyplot_colours[i],zorder=1,edgecolor="k",linewidth=1)

for i,f in enumerate(numpy.unique(numpy.array(df['f']).astype(float))):
    df2=df[df['f']==f]
    print "f = {}, number of binaries = {}".format(f,len(df2))
    rho=numpy.array(df2['rho(kgm-3)']).astype(float)
    R1=((3.0*numpy.array(df2['m1(kg)']).astype(float))/(4.0*numpy.pi*rho))**(1.0/3.0)
    R2=((3.0*numpy.array(df2['m2(kg)']).astype(float))/(4.0*numpy.pi*rho))**(1.0/3.0)
    R1=R1*f
    R2=R2*f
    R2R1=R2/R1
    a_bin=numpy.array(df2['a(m)']).astype(float)
    R_hill=numpy.array(df2['R_hill(m)']).astype(float)

    size_1=a_bin/(R1+R2)
    print size_1
    print len(size_1[size_1<10.0])
    print numpy.median(size_1),numpy.mean(size_1)

    ax1.hist(size_1,bins=100,
    alpha=0.5,color=pf.pyplot_colours[i],
    zorder=1,edgecolor="k",linewidth=1,label="f={}".format(f))

# Set axis labels
ax1.set_ylabel('N')
ax1.set_xlabel("a_bin/(R_1+R_2)")

ax1.legend()

pyplot.tight_layout()

# #save the figure
# script_name=os.path.basename(__file__).split('.')[0]
# picname="{}_{}.png".format(script_name,fname_df.split("/")[-1].split(".")[0])
# print "save {}".format(picname)
# pyplot.savefig(picname,bbox_inches='tight',pad_inches=0.0)
#
# picname="{}_{}.pdf".format(script_name,fname_df.split("/")[-1].split(".")[0])
# print "save {}".format(picname)
# pyplot.savefig(picname,bbox_inches='tight',pad_inches=0.0)

pyplot.show()
# pyplot.close()
