'''
Make Polar plot of TNBs inclination
'''

from matplotlib import rc
rc('font',**{'size':9})

import numpy
import pandas as pd
import matplotlib.pyplot as pyplot
import matplotlib.gridspec as gridspec
import os

import py_func as pf

# use LaTeX fonts in the plot
pyplot.rc('text', usetex=True)
pyplot.rc('font', family='serif')

a_R_hill_cut=0.1

path="/Users/jrobinson/xq1_grav_cloud/binary_stability/orbit_results/orbit_results_plots"
fname_df="df_plot_100_all_stable.txt"

markers=['^','s','o']
alph_norm=0.75

# load real binaries
df_tot=pd.read_csv("/Users/jrobinson/grav_cloud/python_stuff/acquire_binaries/Grundy2019_table_19.csv",sep=",",encoding='utf-8')
df_tot=df_tot.replace(',',"",regex=True)

print df_tot
print list(df_tot)

# Load binaries with inclination
df_complete=df_tot[~df_tot['incl(deg)'].isna()]
# print df_complete

I=numpy.array(df_complete['incl(deg)']).astype(float)
sep_hill=numpy.array(df_complete['a_bin/R_Hill']).astype(float)
print sep_hill,len(sep_hill)

# Select only tight objects
sep_mask=sep_hill<a_R_hill_cut
sep_hill=sep_hill[sep_mask]
I=I[sep_mask]

print sep_hill,len(sep_hill)
print numpy.log10(sep_hill)

#pyplot polar plot
fig = pyplot.figure()

# pc_tex=0.16605 # latex pc in inches
# text_width=39.0*pc_tex
# column_sep=2.0*pc_tex
# column_width=(text_width-column_sep)/2.0
# s_x=1.0
# s_y=0.6
# x_len=((text_width/2.0)+(1.5*pc_tex))*s_x
# y_len=(x_len)*s_y
# print "size: {}x{} inches".format(x_len,y_len)
# fig.set_size_inches(x_len,y_len)

gs = gridspec.GridSpec(1, 1,wspace=0.0)
ax1 = pyplot.subplot(gs[0,0],polar=True)

# ax1.set_xlabel('log(a/R_hill)')
# ax1.set_title("binary inclination (deg)", va='bottom')
# ax1.text(numpy.radians(90.0), 0.5, "binary inclination (deg)",horizontalalignment="center",fontsize=9)
# ax1.text(numpy.radians(90.0), -4.0, '(a/R_hill)',horizontalalignment="center",fontsize=9)
ax1.text(0.5, 1.0, "$\\mathrm{{binary~inclination}}~(\\mathrm{{degrees}})$",horizontalalignment="center",fontsize=9,transform=ax1.transAxes)
ax1.text(0.5, 0.0, '$\\log(a_{{\\mathrm{{bin}}}}/R_{{\\mathrm{{Hill}}}})$',horizontalalignment="center",fontsize=9,transform=ax1.transAxes)

ax1.set_thetamin(0)
ax1.set_thetamax(180)
ax1.set_theta_direction(-1)
ax1.set_theta_zero_location("W")
# ax1.set_thetagrids(numpy.arange(0,180+10,10))
ax1.set_thetagrids(numpy.arange(0,180+45,45))

# log scale
print I
ax1.scatter(numpy.radians(I),numpy.log10(sep_hill),
marker='x',
color="k",
s=15,
alpha=alph_norm,
label="observed binaries",
zorder=3)

# ax1.scatter(0, numpy.log10(1e-3),c="r",marker="+",zorder=4)
# ax1.set_rlim(numpy.log10(1e-3), numpy.log10(0.3))
ax1.set_rlim(numpy.log10(1e-3), numpy.log10(1.0))

# Add the simulated binaires
df=pd.read_csv("{}/{}".format(path,fname_df),sep="\t",index_col=0) # orbits post selection

# only plot the simple binary systems
df=df[df['N_sys']==2]

print len(df)
print len(df[~numpy.isnan(df['m1(kg)'])])
df=df[~numpy.isnan(df['m1(kg)'])] # drop entries without binaries
print list(df)

df['m2/m1']=numpy.array(df['m2(kg)']).astype(float)/numpy.array(df['m1(kg)']).astype(float) # move this to a separate script
print "all systems = {}".format(len(df))

# select "class iii" systems
# df=df[(df['m2/m1']>0.1) & ((df['m1(kg)']+df['m2(kg)'])/df['M_tot(kg)']>0.1)]
# df=df[(df['m2/m1']>0.5) & ((df['m1(kg)']+df['m2(kg)'])/df['M_tot(kg)']>0.5)]

int_list=numpy.array(df['M_tot(kg)'])
for i,M_tot in enumerate(numpy.unique(numpy.array(df['M_tot(kg)']).astype(float))):
    int_mask=(int_list==M_tot)
    int_list[int_mask]=i
int_list=int_list.astype(int)
col_list=[]
marker_list=[]
for i in int_list:
    col_list.append(pf.pyplot_colours[i])
    marker_list.append(markers[i])

a_bin=numpy.array(df['a(m)'])
e_bin=numpy.array(df['e'])
I_bin=numpy.array(df['I(rad)'])

R_hill=numpy.array(df['R_hill(m)'])
print R_hill
# R_hill=R_hill/(30.0/44.0)
print R_hill
m1=numpy.array(df['m1(kg)'])
m2=numpy.array(df['m2(kg)'])
m2m1=numpy.array(df['m2/m1'])
Mtot=numpy.array(df['M_tot(kg)'])

# # only use tight systems
# sep_mask_sim=(a_bin/R_hill)<a_R_hill_cut
# a_bin=a_bin[sep_mask_sim]
# R_hill=R_hill[sep_mask_sim]
# I_bin=I_bin[sep_mask_sim]

x_data=numpy.log10(a_bin/R_hill)
y_data=numpy.degrees(I_bin)

print y_data
print len(y_data)

print len(I[I<90.0]),len(I[I>=90.0])
print "observed retrograde/prograde={}".format(float(len(I[I>=90.0]))/float(len(I[I<90.0])))

print len(y_data[y_data<90.0]),len(y_data[y_data>=90.0])
print "simulated retrograde/prograde={}".format(float(len(y_data[y_data>=90.0]))/float(len(y_data[y_data<90.0])))
print "simulated retrograde fraction={}, mean={}, median={}, std={}".format(
float(len(y_data[y_data>=90.0]))/float(len(y_data)),
numpy.mean(y_data[y_data>=90.0]),
numpy.median(y_data[y_data>=90.0]),
numpy.std(y_data[y_data>=90.0]))
print "simulated prograde fraction={}, mean={}, median={}, std={}".format(
float(len(y_data[y_data<90.0]))/float(len(y_data)),
numpy.mean(y_data[y_data<90.0]),
numpy.median(y_data[y_data<90.0]),
numpy.std(y_data[y_data<90.0]))

print "class iii systems"
df_iii=df[(df['m2/m1']>0.1) & ((df['m1(kg)']+df['m2(kg)'])/df['M_tot(kg)']>0.1)]
print "class iii systems = {}".format(len(df_iii))
# df_iii=df[(df['m2/m1']>0.11383079847916318) & ((df['m1(kg)']+df['m2(kg)'])/df['M_tot(kg)']>0.13050000000007028)]
I_bin_iii=numpy.array(df_iii['I(rad)'])
a_bin_iii=numpy.array(df_iii['a(m)'])
R_hill_iii=numpy.array(df_iii['R_hill(m)'])
x_data_iii=numpy.log10(a_bin_iii/R_hill_iii)
y_data_iii=numpy.degrees(I_bin_iii)
Mtot_iii=numpy.array(df_iii['M_tot(kg)'])
int_list_iii=numpy.array(df_iii['M_tot(kg)'])
for i,M_tot in enumerate(numpy.unique(numpy.array(df_iii['M_tot(kg)']).astype(float))):
    int_mask=(int_list_iii==M_tot)
    int_list_iii[int_mask]=i
int_list_iii=int_list_iii.astype(int)
col_list_iii=[]
marker_list_iii=[]
for i in int_list_iii:
    col_list_iii.append(pf.pyplot_colours[i])
    marker_list_iii.append(markers[i])

# print numpy.degrees(I_bin_iii)
print len(y_data_iii[y_data_iii<90.0]),len(y_data_iii[y_data_iii>=90.0])
print "simulated retrograde/prograde={}".format(float(len(y_data_iii[y_data_iii>=90.0]))/float(len(y_data_iii[y_data_iii<90.0])))
print "simulated retrograde fraction={}, mean={}, median={}, std={}".format(
float(len(y_data_iii[y_data_iii>=90.0]))/float(len(y_data_iii)),
numpy.mean(y_data_iii[y_data_iii>=90.0]),
numpy.median(y_data_iii[y_data_iii>=90.0]),
numpy.std(y_data_iii[y_data_iii>=90.0]))
print "simulated prograde fraction={}, mean={}, median={}, std={}".format(
float(len(y_data_iii[y_data_iii<90.0]))/float(len(y_data_iii)),
numpy.mean(y_data_iii[y_data_iii<90.0]),
numpy.median(y_data_iii[y_data_iii<90.0]),
numpy.std(y_data_iii[y_data_iii<90.0]))

df_other=df[~((df['m2/m1']>0.1) & ((df['m1(kg)']+df['m2(kg)'])/df['M_tot(kg)']>0.1))]
print "other systems = {}".format(len(df_other))
I_bin_other=numpy.array(df_other['I(rad)'])
a_bin_other=numpy.array(df_other['a(m)'])
R_hill_other=numpy.array(df_other['R_hill(m)'])
x_data_other=numpy.log10(a_bin_other/R_hill_other)
y_data_other=numpy.degrees(I_bin_other)
Mtot_other=numpy.array(df_other['M_tot(kg)'])
int_list_other=numpy.array(df_other['M_tot(kg)'])
for i,M_tot in enumerate(numpy.unique(numpy.array(df_other['M_tot(kg)']).astype(float))):
    int_mask=(int_list_other==M_tot)
    int_list_other[int_mask]=i
int_list_other=int_list_other.astype(int)
col_list_other=[]
marker_list_other=[]
for i in int_list_other:
    col_list_other.append(pf.pyplot_colours[i])
    marker_list_other.append(markers[i])

# x_data=x_data_iii
# y_data=y_data_iii

# for i in range(len(x_data_iii)):
#     label="M_c={:.2e}kg".format(Mtot_iii[i])
#     ax1.scatter(numpy.radians(y_data_iii[i]),x_data_iii[i],
#     color="r",
#     s=5,
#     alpha=1.0,
#     zorder=5,
#     label=label)
#
# for i in range(len(x_data)):
#     label="M_c={:.2e}kg".format(Mtot[i])
#     ax1.scatter(numpy.radians(y_data[i]),x_data[i],
#     edgecolors=col_list[i],facecolors='none',
#     marker=marker_list[i],
#     s=20,
#     alpha=1.0,
#     label=label)

for i in range(len(x_data_iii)):
    label="M_c={:.2e}kg".format(Mtot_iii[i])
    ax1.scatter(numpy.radians(y_data_iii[i]),x_data_iii[i],
    edgecolors=col_list_iii[i],facecolors='none',
    marker=marker_list_iii[i],
    s=50,
    alpha=1.0,
    label=label)

for i in range(len(x_data_other)):
    label="M_c={:.2e}kg".format(Mtot_other[i])
    ax1.scatter(numpy.radians(y_data_other[i]),x_data_other[i],
    edgecolors=col_list_other[i],facecolors='none',
    marker=marker_list_other[i],
    s=20,
    alpha=0.25,
    label=label)

# ax1.set_rticks([-3.0,-2.0,-1.0])

#save the figure
# pyplot.tight_layout()

script_name=os.path.basename(__file__).split('.')[0]
picname="{}.png".format(script_name)
print "save {}".format(picname)
pyplot.savefig(picname,bbox_inches='tight',pad_inches=0.0)
# pyplot.savefig(picname)

picname="{}.pdf".format(script_name)
print "save {}".format(picname)
pyplot.savefig(picname,bbox_inches='tight',pad_inches=0.0)
# pyplot.savefig(picname)

pyplot.close()
# pyplot.show()
