'''
Plots a histogram which shows the distribution of:

most massive particle (normalised by cloud mass)
most massive binary primary (normalised by cloud mass)

and on a separate axis:

mass ratio m2/m1 of the binary

Try use the same bins for everything

REMEMBER: this only plots the single most massive binary for each run. We do not account for multiple systems or more than one bound system per run
'''
from matplotlib import rc
rc('font',**{'size':9})

import pandas as pd
import numpy
import py_func as pf
import matplotlib.pyplot as pyplot
import matplotlib.gridspec as gridspec
import os

# use LaTeX fonts in the plot
pyplot.rc('text', usetex=True)
pyplot.rc('font', family='serif')

path="/Users/jrobinson/xq1_grav_cloud/binary_stability/orbit_results/orbit_results_plots"
fname_df="df_plot_100_all_stable.txt"

df=pd.read_csv("{}/{}".format(path,fname_df),sep="\t",index_col=0) # orbits post selection

df=df.sort_values(['m1(kg)','m2(kg)'],axis=0,ascending=False)

print df[['run','m_largest(kg)']]
print numpy.unique(df['run'])
largest_masses=df.drop_duplicates(subset='run',keep='first')[['m_largest(kg)','M_tot(kg)','i_orig']]
largest_mass_norm=largest_masses['m_largest(kg)']/largest_masses['M_tot(kg)']
print largest_mass_norm.to_string()
print len(largest_mass_norm)

# only plot the simple binary systems
df=df[df['N_sys']==2]

binary_masses=df.drop_duplicates(subset='run',keep='first')[['run','m1(kg)','m2(kg)','M_tot(kg)','i_orig']]
print binary_masses.to_string()
bin_mass_norm=binary_masses['m1(kg)']/binary_masses['M_tot(kg)']
bin_mass_ratio=binary_masses['m2(kg)']/binary_masses['m1(kg)']
print bin_mass_norm
print bin_mass_ratio
print len(bin_mass_norm)

nan_runs=binary_masses[numpy.array(pd.isnull(binary_masses['m1(kg)']))]

n_bins=26

# Drop any nan values
bin_mass_norm=numpy.array(bin_mass_norm)
bin_mass_ratio=numpy.array(bin_mass_ratio)
largest_mass_norm=numpy.array(largest_mass_norm)
bin_mass_norm=bin_mass_norm[~numpy.isnan(bin_mass_norm)]
bin_mass_ratio=bin_mass_ratio[~numpy.isnan(bin_mass_ratio)]

print bin_mass_norm,len(bin_mass_norm)
print "mass ratio:\n",bin_mass_ratio,len(bin_mass_ratio)
print "fraction with m2/m1>0.1 = {}".format(float(len(bin_mass_ratio[bin_mass_ratio>0.1]))/float(len(bin_mass_ratio)))

print "number of largest mass particles = {}".format(len(largest_mass_norm))
print "number of most massive binary primaries = {}".format(len(bin_mass_ratio))
print "number of runs with no binaries = {}".format(len(nan_runs))
# log_mass=numpy.log10(largest_mass_norm)

# check if most massive particle is the primary
print df[['run','i_orig','i_largest']]
print "\nfraction of primaries that have binaries: {}\n".format(float(sum(df['i_orig']==df['i_largest']))/float(len(df)))

fig = pyplot.figure()

pc_tex=0.16605 # latex pc in inches
text_width=39.0*pc_tex
column_sep=2.0*pc_tex
column_width=(text_width-column_sep)/2.0
s_x=1.0
s_y=0.8
x_len=((text_width/2.0)+(1.5*pc_tex))*s_x
y_len=(x_len)*s_y
print "size: {}x{} inches".format(x_len,y_len)
fig.set_size_inches(x_len,y_len)

gs = gridspec.GridSpec(1,1)
ax1 = pyplot.subplot(gs[0,0])

# log x lin y
# We also set the bins to be the same!
a1=0.5
a2=1.0
# center_large,hist_large,width_large=pf.hist_dist(numpy.log10(largest_mass_norm),n_bins)
# center_bin,hist_bin,width_bin=pf.hist_dist(numpy.log10(bin_mass_norm),n_bins)
# center_rat,hist_rat,width_rat=pf.hist_dist(numpy.log10(bin_mass_ratio),n_bins)

x_min_array=numpy.array([numpy.amin(numpy.log10(largest_mass_norm)),numpy.amin(numpy.log10(bin_mass_norm)),numpy.amin(numpy.log10(bin_mass_ratio))])
print "smallest x value = ",x_min_array
# x_min=-4.5
x_min=-5.0
x_max=0.0
if x_min>numpy.amin(x_min_array):
    print "axis limit error"
    exit()

hist_large,bins=numpy.histogram(numpy.log10(largest_mass_norm),bins=numpy.linspace(x_min,x_max,n_bins))
center_large = (bins[:-1] + bins[1:]) / 2
width_large = numpy.diff(bins)

hist_bin,bins=numpy.histogram(numpy.log10(bin_mass_norm),bins=numpy.linspace(x_min,x_max,n_bins))
center_bin = (bins[:-1] + bins[1:]) / 2
width_bin = numpy.diff(bins)

hist_rat,bins=numpy.histogram(numpy.log10(bin_mass_ratio),bins=numpy.linspace(x_min,x_max,n_bins))
center_rat = (bins[:-1] + bins[1:]) / 2
width_rat = numpy.diff(bins)

# print bin_mass_norm,len(bin_mass_norm)
# print numpy.log10(bin_mass_norm),len(numpy.log10(bin_mass_norm)),numpy.amin(numpy.log10(bin_mass_norm)),numpy.amax(numpy.log10(bin_mass_norm))
print numpy.sum(hist_large),numpy.sum(hist_bin),numpy.sum(hist_rat)
print "total difference in distributions = {}".format(numpy.sum(numpy.absolute(hist_large-hist_bin))) # find the difference between each bar, shows deviations in dist (requires numpy.absolute!)
if numpy.sum(hist_bin)!=numpy.sum(hist_rat):
    print "binary counting error"
    exit()

lw=1

ax1.bar(center_large, hist_large, align='center',zorder=1,edgecolor='k',
width=width_large,alpha=a1,label="most massive particles, total={}".format(len(largest_mass_norm)))
ax1.bar(center_bin, hist_bin, align='center',zorder=2,edgecolor='k',
width=width_bin,alpha=a1,label="most massive binary primary, total={}".format(len(bin_mass_norm)))
ax2 = ax1.twiny()
ax2.bar(center_rat, hist_rat, align='center',zorder=0,edgecolor='r',fill=False,linewidth=lw,
width=width_rat,alpha=a2,label="most massive binary mass ratio, total={}".format(len(bin_mass_norm)))
x_vals=numpy.append(numpy.append(center_large,center_bin),center_rat)
x_vals=x_vals[~numpy.isnan(x_vals)]
x_vals=x_vals[~numpy.isinf(x_vals)]
xlim1=numpy.amin(x_vals)*1.1
xlim2=0
ax1.set_xlim(xlim1,xlim2)
ax2.set_xlim(xlim1,xlim2)
ax1.set_xlabel("$\\log(m_1/M_\\mathrm{{c}})$")
ax1.set_ylabel("$n$")
ax2.set_xlabel('$\\log(m_2/m_1)$', color='r')

print "mass ratio:"
print hist_rat

bar_rat=ax1.bar(center_rat, hist_rat, align='center',zorder=0,edgecolor='r',fill=False,linewidth=lw,
width=width_rat,alpha=a2,label="most massive binary mass ratio, total={}".format(len(bin_mass_norm)))

# plot our orbit detection limit
ax2.axvline(-3,color="k",alpha=0.2,zorder=0)

ax1.legend(prop={'size': 6})

bar_rat.remove()

ax2.tick_params('x', colors='r')

# ax2.legend()

# ax2.tick_params('x', colors='g')

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
