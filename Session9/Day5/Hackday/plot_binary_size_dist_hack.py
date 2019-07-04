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
import glob

import matplotlib.ticker as ticker

def eformat(f, prec, exp_digits):
    s = "%.*e"%(prec, f)
    mantissa, exp = s.split('e')
    # add 1 to digits as 1 is taken by sign +/-
    return "%se%+0*d"%(mantissa, exp_digits+1, int(exp))

from sklearn.neighbors import KernelDensity

def kde_sklearn(data, grid, bandwidth = 1.0, **kwargs):
    kde_skl = KernelDensity(bandwidth = bandwidth, **kwargs)
    kde_skl.fit(data[:, numpy.newaxis])
    log_pdf = kde_skl.score_samples(grid[:, numpy.newaxis]) # sklearn returns log(density)

    return numpy.exp(log_pdf)

numpy.random.seed(0)

dat_path="/Users/jrobinson/cloud_runs_data/data/jakita_raid2/jer/grav_cloud"

path="/Users/jrobinson/xq1_grav_cloud/binary_stability/orbit_results/orbit_results_plots"
fname_dfs=["df_plot_100_all_stable.txt"]
fname_df=fname_dfs[0]

df_runs=pd.read_csv("{}/{}".format(path,fname_df),sep="\t",index_col=0) # orbits post selection

print len(numpy.unique(df_runs['run']))

print len(df_runs)
print len(df_runs[~numpy.isnan(df_runs['m1(kg)'])])
df=df_runs
df=df_runs[~numpy.isnan(df_runs['m1(kg)'])] # drop entries without binaries
print list(df)

# only plot the simple binary systems
# df=df[df['N_sys']==2]


# Load some data at t=100 yrs
print len(numpy.unique(df['run']))
print df[['run_name','file','file_largest']]
df_unique=df.drop_duplicates(subset=['run_name','file','file_largest'])
print df_unique[['run_name','file','file_largest']]
# runs=numpy.unique(df['run_dir'])
# files=numpy.unique(df['run_dir'])

# for i,run_dir in enumerate(df_unique['run_dir']):
#     split_run_dir=run_dir.split("/")
#     set=split_run_dir[-2]
#     dir=split_run_dir[-1]
#     dir_path="{}/{}/{}".format(dat_path,set,dir)
#     print dir_path
#     dat_files=pf.create_file_list(dir_path)
#     dat_i=str(int(df_unique.iloc[i]['file_largest']))
#     dat_files=[f for f in dat_files if dat_i in f]
#     dat_files.sort()
#     dat_file=dat_files[-1]
#     t,df_dat=pf.load_dat_file("{}/{}".format(dir_path,dat_file))
#     df_rp=pf.load_run_params("{}/run_params_0.txt".format(dir_path))
#
#     Ntot=float(df_rp['N_tot'].iloc[0])
#     rho=float(df_rp['rho(kgm-3)'].iloc[0])
#     M_tot=float(df_rp['M_tot(kg)'].iloc[0])
#     mp=M_tot/Ntot #kg
#     print mp
#
#     print len(df_dat)
#     m=numpy.array(df_dat['m(kg)']).astype(float)
#     m_norm=m/mp
#     print m_norm
#     print len(m_norm)
#     print len(m_norm[m_norm>1])
#     print len(m_norm[m_norm>2])
#     print len(m_norm[m_norm>3])
#     print len(m_norm[m_norm>4])
#     print len(m_norm[m_norm>5])
#     print len(m_norm[m_norm>6])
#
#     print numpy.amin(m)
#     print numpy.amax(m)
#     R=((3.0*numpy.array(df_dat['m(kg)']).astype(float))/(4.0*numpy.pi*rho))**(1.0/3.0)
#     print numpy.amin(R)
#     print numpy.amax(R)
#
#     # fig = pyplot.figure()
#     # gs = gridspec.GridSpec(1,1)
#     # ax1 = pyplot.subplot(gs[0,0])
#     #
#     # x_data=R
#     #
#     # # Add the histograms
#     # n_bins='auto'
#     # n,bins, _ =ax1.hist(x_data,density=True,bins=n_bins,edgecolor='k',alpha=0.5,color=pf.pyplot_colours[i])
#     #
#     # # add kde
#     # N_bin=500
#     # grid_x=numpy.linspace(numpy.amin(x_data),numpy.amax(x_data),N_bin)
#     #
#     # bw=1e4
#     # PDF_x = kde_sklearn(x_data, grid_x,bandwidth=bw)
#     # ax1.plot(grid_x,PDF_x,color=pf.pyplot_colours[i])
#     #
#     # # Set axis labels
#     # ax1.set_ylabel('R(m)')
#     # ax1.set_xlabel("P.D.")
#     #
#     # pyplot.show()
#
#     exit()

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

m1=numpy.array(df['m1(kg)']).astype(float)
m2=numpy.array(df['m2(kg)']).astype(float)
m2m1=numpy.array(df['m2/m1']).astype(float)
Mtot=numpy.array(df['M_tot(kg)']).astype(float)
rho=numpy.array(df['rho(kgm-3)']).astype(float)
R1=((3.0*numpy.array(df['m1(kg)']).astype(float))/(4.0*numpy.pi*rho))**(1.0/3.0)
R2=((3.0*numpy.array(df['m2(kg)']).astype(float))/(4.0*numpy.pi*rho))**(1.0/3.0)
R2R1=R2/R1
m_norm=(m1+m2)/(Mtot/1e5)
print m_norm
print numpy.amin(m_norm)

# x_data=R1

# for i,x_data in enumerate([R1,R2,numpy.append(R1,R2)]):
for i,x_data in enumerate([m_norm]):

    # Add the histograms
    # n_bins='auto'
    n_bins = 1000
    n,bins, _ =ax1.hist(x_data,density=True,bins=n_bins,edgecolor='k',alpha=0.5,color=pf.pyplot_colours[i])

    # # add kde
    # N_bin=500
    # grid_x=numpy.linspace(numpy.amin(x_data),numpy.amax(x_data),N_bin)
    #
    # # bw=1e4
    # # PDF_x = kde_sklearn(x_data, grid_x,bandwidth=bw)
    # PDF_x = kde_sklearn(x_data, grid_x)
    # ax1.plot(grid_x,PDF_x,color=pf.pyplot_colours[i])



# Set axis labels
ax1.set_ylabel('R(m)')
ax1.set_xlabel("P.D.")

ax1.legend()

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
