'''
This script reads binary data directly from Will Grundy's website.

Here we look at the binary period distribution
'''

import pandas as pd
import matplotlib.pyplot as pyplot
import matplotlib.gridspec as gridspec
import numpy
from astroquery.mpc import MPC
from pprint import pprint
import py_func as pf
import os

# use LaTeX fonts in the plot
pyplot.rc('text', usetex=True)
pyplot.rc('font', family='serif')

# load dataframe
df=pd.read_csv("/Users/jrobinson/grav_cloud/python_stuff/acquire_binaries/df_tnb_tot_deets_04_08_2019.txt",sep="\t",index_col=0)
# df=pd.read_csv("/Users/jrobinson/grav_cloud/python_stuff/acquire_binaries/df_tnb_pae.txt",sep="\t",index_col=0)

df=df[~numpy.isnan(df['Msys(10^18kg)'])] # drop entries without measurement

print df[['Object','a(km)','Msys(10^18kg)']].to_string()
print "There are {} binaries with system masses".format(len(df))

# #drop Eris
# for i in range(len(df)):
#     name = df.iloc[i]['Object']
#     print name
#     # name=name.encode('utf-8')
#     df.loc[i,'Object']=name
#     print df.iloc[i]['Object']
#
#     if 'eris' in name.lower():
#         print 'FOUND ERIS'
#         i_drop=i
#         #break
# df=df.drop([i_drop])

# arrays and units
a_bin=numpy.array(df['a(km)'])
M=numpy.array(df['Msys(10^18kg)'])
M_sys=M*1e18
a_bin=a_bin*1e3
log_a=numpy.log10(a_bin)

# print a_bin
# print M
# print len(a_bin)

fig = pyplot.figure() #open figure once
fig.set_size_inches(10,4)

gs = gridspec.GridSpec(1,2)
ax1 = pyplot.subplot(gs[0,0])
ax2 = pyplot.subplot(gs[0,1])

# center_M, hist_M, width_M=pf.hist_dist(numpy.log10(M_sys),100)
# ax1.bar(center_M, hist_M, align='center',zorder=0,edgecolor='k',width=width_M,label='log(System Mass (kg))')
# ax1.set_ylabel('N')
# ax1.set_xlabel('log(System Mass (kg))')

print "a_bin (m)\n{}".format(a_bin)
print "mean separation = {}m".format(numpy.mean(a_bin))
print "median separation = {}m".format(numpy.mean(a_bin))
print "minimum separation = {}m".format(numpy.amin(a_bin))
print "maximum separation = {}m".format(numpy.amax(a_bin))

print "M_sys (kg)\n{}".format(M_sys)
print "minimum mass = {}kg".format(numpy.amin(M_sys))
print "maximum mass = {}kg".format(numpy.amax(M_sys))

P=numpy.array(df['P(days)']).astype(float)
print "period (d)\n{}".format(P)
P_mean=numpy.mean(P)
P_med=numpy.median(P)
print "mean period = {}d".format(P_mean)
print "median period = {}d".format(P_med)
print "minimum period = {}d".format(numpy.amin(P))
print "maximum period = {}d".format(numpy.amax(P))

# center_M, hist_M, width_M=pf.hist_dist(P,100)
# ax1.axvline(100,c='r')
# ax1.set_xlabel('Period P (days)')

# Plot a histogram of the binary period
center_M, hist_M, width_M=pf.hist_dist(numpy.log10(P),10)
ax1.axvline(numpy.log10(P_mean),c='r',linestyle=':',label='mean')
ax1.axvline(numpy.log10(P_med),c='r',linestyle='--',label='median')
ax1.set_xlabel('$\\log(\\mathrm{{Period}} ~(\\mathrm{{days}}))$')
ax1.bar(center_M, hist_M, align='center',zorder=0,edgecolor='k',width=width_M)#,label='Period P (days)')
ax1.set_ylabel('$n$')
ax1.legend()

# Plot a scatter plot of the Period as a function of System mass
bin_samp=100
# ax2.scatter(M_sys,P)
s2=ax2.scatter(numpy.log10(P),numpy.log10(M_sys),c=numpy.log10(a_bin),vmin=numpy.amin(numpy.log10(a_bin)),vmax=numpy.amax(numpy.log10(a_bin)))
ax2.set_ylabel("$\\log(M_\\mathrm{{sys}} ~(\\mathrm{{kg}}))$")
ax2.set_xlabel('$\\log(\\mathrm{{Period}} ~(\\mathrm{{days}}))$')
ax2.axvline(numpy.log10(bin_samp),c="r",label="$t_\\mathrm{{samp}}$")
ax2.legend()

cbar2=fig.colorbar(s2)
cbar2.set_label("$\\log(a_\\mathrm{{\\mathrm{{bin}}}}~(m))$")

#save the figure
script_name=os.path.basename(__file__).split('.')[0]
picname="{}.png".format(script_name)
print "save {}".format(picname)
pyplot.savefig(picname,bbox_inches='tight',pad_inches=0.0)

picname="{}.pdf".format(script_name)
print "save {}".format(picname)
pyplot.savefig(picname,bbox_inches='tight',pad_inches=0.0)

# pyplot.show()
pyplot.close()
