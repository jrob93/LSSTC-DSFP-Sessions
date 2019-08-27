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

path="/Users/jrobinson/xq1_grav_cloud/binary_stability/orbit_results/orbit_results_plots"
fname_df="df_plot_100_all_stable.txt"

df=pd.read_csv("{}/{}".format(path,fname_df),sep="\t",index_col=0) # orbits post selection

# print len(numpy.unique(df['run']))
# exit()

n_bins=20

print "all systems"
print df[['run','i','i_orig','m1(kg)','m_largest(kg)']]#.to_string()
print "all systems with a nan entry"
print df[numpy.isnan(df['m1(kg)'])][['run','i','i_orig','m1(kg)','m_largest(kg)']]#.to_string()
print "all runs that never had a binary"
print df[(numpy.isnan(df['i'])) & (numpy.isnan(df['i_orig']))][['run','i','i_orig','m1(kg)','m_largest(kg)']]#.to_string()
print numpy.array(df[(numpy.isnan(df['i'])) & (numpy.isnan(df['i_orig']))]['run_name']).tolist()
# exit()

# missing_runs=df[numpy.isnan(df['m1(kg)'])]['run']
# print missing_runs
nan_runs=[]

runs=numpy.unique(numpy.array(df['run']).astype(int))
largest_mass_norm=[]
bin_mass_norm=[]
bin_mass_ratio=[]
for r in runs:

    # if r!=84:
    #     continue

    print r
    df_run=df[df['run']==r].sort_values(by=['m1(kg)','m2(kg)','a(m)'],ascending=[False,False,True])
    mass=numpy.unique(numpy.array(df_run['m_largest(kg)']).astype(float))
    cloud_mass=numpy.unique(numpy.array(df_run['M_tot(kg)']).astype(float))

    # Ensure we only have one run
    if (len(mass)>1) | (len(cloud_mass)>1):
        print df_run[['t(s)','i','j','m_largest(kg)','run','run_name']]
        print mass,cloud_mass
        print len(mass),len(cloud_mass)
        print "error"
        break
        # exit()

    # print df_run[['i','j','m1(kg)','m2(kg)','M_tot(kg)','i_orig']]

    # Get primary and secondary masses, the arrays should already be descending sort
    pri_masses=numpy.array(df_run['m1(kg)']).astype(float)
    sec_masses=numpy.array(df_run['m2(kg)']).astype(float)
    print "number of primaries: ",len(pri_masses)
    bin_mass=pri_masses[0]
    sec_mass=sec_masses[0]

    print bin_mass,type(bin_mass)
    if numpy.isnan(bin_mass):
        print "nan run"
        nan_runs.append(r)
        # continue

    # If there are multiple orbits, test that we have grabbed the most massive pair
    print "check for most massive pair"
    print mass,pri_masses
    if len(mass)>1:
        print pri_masses
        print pri_masses[~numpy.isnan(pri_masses)]
        if bin_mass!=numpy.amax(pri_masses[~numpy.isnan(pri_masses)]) or sec_mass!=numpy.amax(sec_masses[~numpy.isnan(sec_masses)]):
            print "binary mass error"
            print bin_mass,numpy.amax(pri_masses[~numpy.isnan(pri_masses)])
            print sec_mass,numpy.amax(sec_masses[~numpy.isnan(sec_masses)])
            print pri_masses
            print numpy.array(df_run['i']).astype(int)
            print sec_masses
            exit()
    # print numpy.array(df_run['m2(kg)']).astype(float),sec_mass
    # print numpy.array(df_run['m1(kg)']).astype(float)
    # print bin_mass
    largest_mass=mass[0]
    cloud_mass=cloud_mass[0]
    largest_mass_norm.append(largest_mass/cloud_mass)

    bin_mass_norm.append(bin_mass/cloud_mass)
    bin_mass_ratio.append([sec_mass/bin_mass])
        
    print cloud_mass,largest_mass
    # Note some of the high mass ratios are just extremely small particles
    if sec_mass/bin_mass>0.9:
        print r
        print "high mass ratio, {}, {} {}".format(sec_mass/bin_mass,bin_mass,sec_mass)
        print (bin_mass+sec_mass)/cloud_mass
# exit()

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
ax1.set_xlabel("log(m_1/M_c)")
ax1.set_ylabel("N")
ax2.set_xlabel('log(m_2/m_1)', color='r')

print "mass ratio:"
print hist_rat

bar_rat=ax1.bar(center_rat, hist_rat, align='center',zorder=0,edgecolor='r',fill=False,linewidth=lw,
width=width_rat,alpha=a2,label="most massive binary mass ratio, total={}".format(len(bin_mass_norm)))

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

pyplot.show()
# pyplot.close()
