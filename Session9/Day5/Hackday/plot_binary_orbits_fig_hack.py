'''
plot the aei for all our simualted binary orbits.
Compare the observed tnbs, that we have full orbital solutions for.
Do as polar plot?
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
ax2 = pyplot.subplot(gs[0,0],sharex=ax1)

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

# a_bin=numpy.array(df['a(m)'])
# e_bin=numpy.array(df['e'])
# I_bin=numpy.array(df['I(rad)'])
#
# m1=numpy.array(df['m1(kg)'])
# m2=numpy.array(df['m2(kg)'])
# m2m1=numpy.array(df['m2/m1'])
# Mtot=numpy.array(df['M_tot(kg)'])
#
# # x_data=a_bin
# x_data=numpy.log10(a_bin)
# y_data_2=e_bin
# y_data_1=numpy.degrees(I_bin)
#
# for i in range(len(x_data)):
#     label="$M_\\mathrm{{c}}={:.2e}~\\mathrm{{kg}}$".format(Mtot[i])
#     ax2.scatter(x_data[i],y_data_2[i],
#     edgecolors=col_list[i],facecolors='none',
#     marker=marker_list[i],
#     s=20,
#     alpha=1.0)
#
#     ax1.scatter(x_data[i],y_data_1[i],
#     edgecolors=col_list[i],facecolors='none',
#     marker=marker_list[i],
#     s=20,
#     alpha=1.0,
#     label=label)

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

# x_data=a_bin
x_data_iii=numpy.log10(a_bin_iii)
y_data_2_iii=e_bin_iii
y_data_1_iii=numpy.degrees(I_bin_iii)

for i in range(len(x_data_iii)):
    label="$M_\\mathrm{{c}}={:.2e}~\\mathrm{{kg}}$".format(Mtot_iii[i])
    ax2.scatter(x_data_iii[i],y_data_2_iii[i],
    edgecolors=col_list_iii[i],facecolors='none',
    marker=marker_list_iii[i],
    s=50,
    alpha=1.0)

    ax1.scatter(x_data_iii[i],y_data_1_iii[i],
    edgecolors=col_list_iii[i],facecolors='none',
    marker=marker_list_iii[i],
    s=50,
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

# x_data=a_bin
x_data_other=numpy.log10(a_bin_other)
y_data_2_other=e_bin_other
y_data_1_other=numpy.degrees(I_bin_other)

for i in range(len(x_data_other)):
    label="$M_\\mathrm{{c}}={:.2e}~\\mathrm{{kg}}$".format(Mtot_other[i])
    ax2.scatter(x_data_other[i],y_data_2_other[i],
    edgecolors=col_list_other[i],facecolors='none',
    marker=marker_list_other[i],
    s=20,
    alpha=0.15)

    ax1.scatter(x_data_other[i],y_data_1_other[i],
    edgecolors=col_list_other[i],facecolors='none',
    marker=marker_list_other[i],
    s=20,
    alpha=0.15,
    label=label)

ax1.axhline(90,color="k",alpha=0.2,zorder=0)

# # include observed objects
# # load real binaries
# df_tot=pd.read_csv("/Users/jrobinson/grav_cloud/python_stuff/acquire_binaries/df_tnb_tot_deets_18_06_2019.txt",sep="\t",index_col=0)
# print df_tot
# print list(df_tot)
#
# # Find anything that is a special case on Grundy's webpage, or a dwarf planet at: https://en.wikipedia.org/wiki/Dwarf_planet
# DP_names=["ceres","pluto","haumea","makemake","eris","orcus","salacia","quaoar","sedna"]
# DP_DES=["2002 ms4","2007 or10"]
# df_weird=pd.DataFrame()
# df_norm=pd.DataFrame()
# for i in range(len(df_tot)):
#     name_check=0
#     name = df_tot.iloc[i]['Object']
#     # print name
#     DES = df_tot.iloc[i]['DES']
#     print i,name,DES
#
#     if str(df_tot.iloc[i]['Orbit Status']).lower()=="special case":
#         print "weird"
#         # if df_tot.iloc[i]['Deltamag']==0.0:
#         #     continue
#         # elif "lempo" in name.lower():
#         #     continue
#         # else:
#         #     df_weird=df_weird.append(df_tot.iloc[i])
#         #     continue
#         df_weird=df_weird.append(df_tot.iloc[i])
#         continue
#
#     elif (DES.lower() in DP_DES):
#         print "weird"
#         df_weird=df_weird.append(df_tot.iloc[i])
#         continue
#     else:
#         for n in (name.lower().split(" ")):
#             if n in DP_names:
#                 print "weird"
#                 df_weird=df_weird.append(df_tot.iloc[i])
#                 name_check+=1
#                 break
#
#     if name_check==0:
#         print "append"
#         df_norm=df_norm.append(df_tot.iloc[i])
#
# print len(df_tot)
# print len(df_norm)
# print len(df_weird)
#
# # df_tot=df_norm
# # exit()
#
# a_bin_norm=numpy.array(df_norm['a(km)'])*1e3
# e_bin_norm=numpy.array(df_norm['e'])
# I_bin_norm=numpy.array(df_norm['i(deg)'])
#
# a_bin_weird=numpy.array(df_weird['a(km)'])*1e3
# e_bin_weird=numpy.array(df_weird['e'])
# I_bin_weird=numpy.array(df_weird['i(deg)'])
#
# x_data_norm=numpy.log10(a_bin_norm)
# y_data_norm_1=I_bin_norm # lol
# y_data_norm_2=e_bin_norm
#
# x_data_weird=numpy.log10(a_bin_weird)
# y_data_weird_1=I_bin_weird
# y_data_weird_2=e_bin_weird
#
# alph_norm=0.5
# ax1.scatter(x_data_norm,y_data_norm_1,
# marker='x',
# color="k",
# s=15,
# alpha=alph_norm,
# label="observed binaries")
#
# ax2.scatter(x_data_norm,y_data_norm_2,
# marker='x',
# color="k",
# s=15,
# alpha=alph_norm)
#
# ax1.scatter(x_data_weird,y_data_weird_1,
# marker="+",
# color="r",
# s=15,
# alpha=1.0,
# label="Special/D.P.")
#
# ax2.scatter(x_data_weird,y_data_weird_2,
# marker="+",
# color="r",
# s=15,
# alpha=1.0)

# load real binaries
df_tot=pd.read_csv("/Users/jrobinson/grav_cloud/python_stuff/acquire_binaries/Grundy2019_table_19.csv",sep=",",encoding='utf-8')
df_tot=df_tot.replace(',',"",regex=True)
print list(df_tot)

# Load binaries with inclination
df_complete=df_tot[~df_tot['incl(deg)'].isna()]

a_bin_real=numpy.array(df_complete['a_bin(km)']).astype(float)*1e3
# e_bin_real=numpy.array(df_complete['e_hel']).astype(float)
e_bin_real=numpy.array(df_complete['e']).astype(float)
I_bin_real=numpy.array(df_complete['incl(deg)']).astype(float)

x_data_real=numpy.log10(a_bin_real)
y_data_real_1=I_bin_real # lol
y_data_real_2=e_bin_real

alph_real=0.75
ax1.scatter(x_data_real,y_data_real_1,
marker='x',
color="k",
s=15,
alpha=alph_real,
label="observed binaries")

ax2.scatter(x_data_real,y_data_real_2,
marker='x',
color="k",
s=15,
alpha=alph_real)


padding_y=0.1
padding_x=padding_y
# ax1.set_xlim(numpy.amin(x_data)-padding_x,0.0+padding_x)
# # ax1.set_ylim(numpy.amin(y_data)-padding_y,0.0+padding_y)
# # ax4.set_xlim(0.0-padding_x,1.0+padding_x)
# ax4.set_ylim(0.0-padding_y,numpy.amax(10**y_data)+padding_y)

ax1.set_xlabel('$\\log(a_{{\\mathrm{{bin}}}}~(\\mathrm{{m}}))$')
ax1.set_ylabel("$i_{{\\mathrm{{bin}}}}~(\\mathrm{{degrees}})$")
ax2.set_ylabel("$e_{{\\mathrm{{bin}}}}$")

# Remove duplicates from legend
handles, labels = ax1.get_legend_handles_labels()
from collections import OrderedDict
by_label = OrderedDict(zip(labels, handles))

# ax1.legend(by_label.values(), by_label.keys(),prop={'size': 6})

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
