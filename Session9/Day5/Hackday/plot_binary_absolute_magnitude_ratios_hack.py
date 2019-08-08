'''
Based on Nesvorny et al 2010 fig.3.
NB, compare similarities and differences?
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

# use LaTeX fonts in the plot
pyplot.rc('text', usetex=True)
pyplot.rc('font', family='serif')

def eformat(f, prec, exp_digits):
    s = "%.*e"%(prec, f)
    mantissa, exp = s.split('e')
    # add 1 to digits as 1 is taken by sign +/-
    return "%se%+0*d"%(mantissa, exp_digits+1, int(exp))

path="/Users/jrobinson/xq1_grav_cloud/binary_stability/orbit_results/orbit_results_plots"
fname_dfs=["df_plot_100_all_stable.txt"]
fname_df=fname_dfs[0]

df=pd.read_csv("{}/{}".format(path,fname_df),sep="\t",index_col=0) # orbits post selection

markers=['^','s','o']

# load real binaries
# df_tot=pd.read_csv("/Users/jrobinson/grav_cloud/python_stuff/acquire_binaries/df_tnb_tot_deets_18_06_2019.txt",sep="\t",index_col=0)
df_tot=pd.read_csv("/Users/jrobinson/grav_cloud/python_stuff/acquire_binaries/df_tnb_tot_deets_04_08_2019.txt",sep="\t",index_col=0)
print df_tot
print list(df_tot)

# Find anything that is a special case on Grundy's webpage, or a dwarf planet at: https://en.wikipedia.org/wiki/Dwarf_planet
DP_names=["ceres","pluto","haumea","makemake","eris","orcus","salacia","quaoar","sedna"]
DP_DES=["2002 ms4","2007 or10"]
df_weird=pd.DataFrame()
df_norm=pd.DataFrame()
for i in range(len(df_tot)):
    name_check=0
    name = df_tot.iloc[i]['Object']
    # print name
    DES = df_tot.iloc[i]['DES']
    print i,name,DES

    if str(df_tot.iloc[i]['Orbit Status']).lower()=="special case":
        print "weird"
        # if df_tot.iloc[i]['Deltamag']==0.0:
        #     continue
        # elif "lempo" in name.lower():
        #     continue
        # else:
        #     df_weird=df_weird.append(df_tot.iloc[i])
        #     continue
        df_weird=df_weird.append(df_tot.iloc[i])
        continue

    elif (DES.lower() in DP_DES):
        print "weird"
        df_weird=df_weird.append(df_tot.iloc[i])
        continue
    else:
        for n in (name.lower().split(" ")):
            if n in DP_names:
                print "weird"
                df_weird=df_weird.append(df_tot.iloc[i])
                name_check+=1
                break

    if name_check==0:
        print "append"
        df_norm=df_norm.append(df_tot.iloc[i])

print len(df_tot)
print len(df_norm)
print len(df_weird)

# df_tot=df_norm
# exit()

V1=numpy.array(df_tot['V1']).astype(float)
V2=numpy.array(df_tot['V2']).astype(float)
V1_norm=numpy.array(df_norm['V1']).astype(float)
V2_norm=numpy.array(df_norm['V2']).astype(float)
V2V1_norm=numpy.array(df_norm['Deltamag']).astype(float)
V1_weird=numpy.array(df_weird['V1']).astype(float)
V2_weird=numpy.array(df_weird['V2']).astype(float)
V2V1_weird=numpy.array(df_weird['Deltamag']).astype(float)

HV_combined=numpy.array(df_tot['HV(combined)']).astype(float)
HV_combined_norm=numpy.array(df_norm['HV(combined)']).astype(float)
HV_combined_weird=numpy.array(df_weird['HV(combined)']).astype(float)
HV1=HV_combined*(V2/V1)
HV1_norm=HV_combined_norm*(V2_norm/V1_norm)
HV1_weird=HV_combined_weird*(V2_weird/V1_weird)

print V1,V2

fig = pyplot.figure()

pc_tex=0.16605 # latex pc in inches
text_width=39.0*pc_tex
column_sep=2.0*pc_tex
column_width=(text_width-column_sep)/2.0

# x_len=text_width+(column_sep)+(2.2*pc_tex) # add an additional couple points to fit
# y_len=x_len/1.5
# print "size: {}x{} inches".format(x_len,y_len)
# fig.set_size_inches(x_len,y_len)

s_x=1.0
s_y=0.8
x_len=((text_width/2.0)+(1.5*pc_tex))*s_x
y_len=(x_len)*s_y
print "size: {}x{} inches".format(x_len,y_len)
fig.set_size_inches(x_len,y_len)

gs = gridspec.GridSpec(1,1)
ax1 = pyplot.subplot(gs[0,0])

print len(df)
print len(df[~numpy.isnan(df['m1(kg)'])])
df=df[~numpy.isnan(df['m1(kg)'])] # drop entries without binaries
print list(df)

# only plot the simple binary systems
df=df[df['N_sys']==2]

# # Only include f values 3,10,30
# df=df[(df['f']<100.0) & (df['f']>1.0)]

# plot size ratio against primary size (similar to Nesvorny et al 2010)

d=44.0 #object distance (AU)
p1=0.08 # albedo, changing this doesn't seem to affect the plot much
# p1=0.15 # albedo, changing this doesn't seem to affect the plot much
p2=p1
C=664.5e3 # constant for V band
d0=1.0

print "magnitude parameters:\nd={}AU\np_V={}\nC_V={}".format(d,p1,C)

# twin axis for extra axis scale
# ax2 = ax1.twiny()

# for i,M_tot in enumerate(numpy.unique(numpy.array(df['M_tot(kg)']).astype(float))):
#     df2=df[df['M_tot(kg)']==M_tot]
#     rho=numpy.array(df2['rho(kgm-3)']).astype(float)
#     R1=((3.0*numpy.array(df2['m1(kg)']).astype(float))/(4.0*numpy.pi*rho))**(1.0/3.0)
#     R2=((3.0*numpy.array(df2['m2(kg)']).astype(float))/(4.0*numpy.pi*rho))**(1.0/3.0)
#     R2R1=R2/R1
#
#     delta_mag=5.0*numpy.log10(numpy.sqrt(p1/p2)*(R1/R2))
#     mag1=5.0*numpy.log10(C/(numpy.sqrt(p1)*R1)*((d*(d-d0))/(d0**2.0)))
#     abs_mag1=5.0*numpy.log10(C/(numpy.sqrt(p1)*R1))
#
#     ax1.scatter(delta_mag,abs_mag1,
#     edgecolors=pf.pyplot_colours[i],facecolors='none',
#     marker=markers[i],
#     s=20,
#     label="M_c={:.2e}kg".format(M_tot),
#     alpha=1)

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

Mtot=numpy.array(df['M_tot(kg)']).astype(float)
rho=numpy.array(df['rho(kgm-3)']).astype(float)
R1=((3.0*numpy.array(df['m1(kg)']).astype(float))/(4.0*numpy.pi*rho))**(1.0/3.0)
R2=((3.0*numpy.array(df['m2(kg)']).astype(float))/(4.0*numpy.pi*rho))**(1.0/3.0)
R2R1=R2/R1

delta_mag=5.0*numpy.log10(numpy.sqrt(p1/p2)*(R1/R2))
mag1=5.0*numpy.log10(C/(numpy.sqrt(p1)*R1)*((d*(d-d0))/(d0**2.0)))
abs_mag1=5.0*numpy.log10(C/(numpy.sqrt(p1)*R1))

x_data=delta_mag
y_data=abs_mag1

for i in range(len(x_data)):
    label="$M_\\mathrm{{c}}={:.2e}~\\mathrm{{kg}}$".format(Mtot[i])
    ax1.scatter(x_data[i],y_data[i],
    edgecolors=col_list[i],facecolors='none',
    marker=marker_list[i],
    s=20,
    alpha=1.0,
    label=label)

ax1.scatter(V2V1_norm,HV1_norm,marker='x',color='k',s=15,alpha=0.25,label='observed binaries',zorder=0)
ax1.scatter(V2V1_weird,HV1_weird,marker='+',color='r',s=15,alpha=1.0,label='Special/D.P.',zorder=0)


# Empirical detection limits: plot some loose bounds for now

ax1.invert_yaxis()
# Set axis labels
ax1.set_ylabel('$\\mathrm{{primary~absolute~magnitude}}~\\mathrm{{m}}_{{V}}$')
ax1.set_xlabel("$\\Delta \\mathrm{{m}}_{{V}}$")

# add MU69 to plot, check if values match observables

# R1_MU69=19.46e3/2.0
# R2_MU69=14.24e3/2.0
# d_MU69=44.6
# p1_MU69=0.165

# Calculate size from Volume
V1_MU69=1400e9 # m^3
V2_MU69=1050e9 # m^3
V1_err=600e9 # m
V2_err=400e9 # m
R1_MU69=(3.0*V1_MU69/(4.0*numpy.pi))**(1.0/3.0)
R2_MU69=(3.0*V2_MU69/(4.0*numpy.pi))**(1.0/3.0)
R1_err=R1_MU69*V1_err/(3.0*V1_MU69)
R2_err=R2_MU69*V2_err/(3.0*V2_MU69)
d_MU69=44.0

# Albedo
p1_MU69=p1
p2_MU69=p1_MU69

print "MU69 magnitude parameters:\nd={}AU\np_V={}\nC_V={}".format(d_MU69,p1_MU69,C)
print "spherical size of MU69 = {} +/- {}, {} +/- {} m".format(R1_MU69,R1_err,R2_MU69,R2_err)

delta_mag_MU69=5.0*numpy.log10(numpy.sqrt(p1_MU69/p2_MU69)*(R1_MU69/R2_MU69))
mag1_MU69=5.0*numpy.log10(C/(numpy.sqrt(p1_MU69)*R1_MU69)*((d_MU69*(d_MU69-d0))/(d0**2.0)))
abs_mag1_MU69=5.0*numpy.log10(C/(numpy.sqrt(p1_MU69)*R1_MU69))

m1_err=5.0*(R1_err/(R1_MU69*numpy.log(10)))
m2_err=5.0*(R2_err/(R2_MU69*numpy.log(10)))
Dm_err=numpy.sqrt(((m1_err)**2.0)+((m2_err)**2.0))
print "m1 = {} +/- {}, Delta m = {} +/- {}".format(mag1_MU69,m1_err,delta_mag_MU69,Dm_err)

ax1.errorbar(delta_mag_MU69,abs_mag1_MU69,xerr=Dm_err,yerr=m1_err,color='r',capsize=3)
ax1.scatter(delta_mag_MU69,abs_mag1_MU69,marker='o',s=20,color='r',label='2014 MU69')

# ax1.text(delta_mag_MU69+shift_x,mag1_MU69,"2014 MU69")

# plot our orbit detection limit
ax1.axvline(5.0,color="k",alpha=0.2,zorder=0)

# # add mass ratio to axis
#
lim = ax1.get_xlim()
padding=0.1
ax1.set_xlim(0.0-padding,lim[1])
#
# # we get the ax2 y ticks, remove the m2/m1=0 and add m2/m1=1e-3
# # note that the ylims must be explicitly preserved
# lim = ax1.get_xlim() # set ax2 to have the exact same limits (and same default ticks) as ax1. Works provided we don't change ax1 from here on
# ax2.set_xlim(lim)
#
# print ax2.get_xticks()
#
# # use this function to convert an m2/m1 value into a delta mag value
# # FuncFormatter can be used as a decorator
# @ticker.FuncFormatter
# def major_formatter(x, pos):
#     return "{}".format(eformat((numpy.sqrt(p1/p2)*(10.0**(-x/5.0)))**3.0,1,1))
#
# ax2.xaxis.set_major_formatter(major_formatter)
# ax2.set_xlabel("m_2/m1")
#
# # add normalised mass to axis
#
# # we get the ax2 y ticks, remove the m2/m1=0 and add m2/m1=1e-3
# # note that the ylims must be explicitly preserved
# lim = ax2.get_ylim()
# ax2_yticks=list(ax2.get_yticks())
# print ax2_yticks
# ax2.set_ylim(lim)
#
# print ax2.get_yticks()
# # use this function to convert an m2/m1 value into a delta mag value
# # FuncFormatter can be used as a decorator
# @ticker.FuncFormatter
# def major_formatter_y(y, pos):
#     return "{}".format(eformat((numpy.sqrt(p1/p2)*(10.0**(-y/5.0)))**3.0,1,1))
#
# ax2.xaxis.set_major_formatter(major_formatter_y)
# ax2.set_ylabel("(m_2+m_1)/M_c")

# ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))

# ax1.legend()

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
