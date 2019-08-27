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
print len(df)
print len(df[df['N_sys']==2])
print len(df[df['N_sys']==3])
print len(df[df['N_sys']==4])
print len(df[df['N_sys']==5])

print list(df)

# print df[df['N_sys']>2][['run','N_sys','i_orig','m1(kg)','m2(kg)','m_tot(kg)']]

# only plot the simple binary systems
# df=df[df['N_sys']==4]


# Load some data at t=100 yrs
print len(numpy.unique(df['run']))
print df[['run_name','file','file_largest']]
df_unique=df.drop_duplicates(subset=['run_name','file','file_largest'])
print df_unique[['run_name','file','file_largest']]
# runs=numpy.unique(df['run_dir'])
# files=numpy.unique(df['run_dir'])

print numpy.mean(df['N_sys'])
print numpy.median(df['N_sys'])

print df[['run','N_sys','i_orig','m1(kg)','m2(kg)','m_tot(kg)']]

compile=0

m_norm_avg=0
count=0
m_mag=[]
M_cloud=[]

# magnitude to size to mass
d=44.0 #object distance (AU)
p1=0.15 # albedo
p2=p1
C=664.5e3 # constant for V band
d0=1.0
mag=25
rho=1e3

R=(10.0**(-0.2*mag))*(C/numpy.sqrt(p1))*(d*(d-d0))
M=(4.0/3.0)*numpy.pi*rho*(R**3.0)
print M
M_clouds=numpy.array([4.19e18,6.54e19,1.77e21])
print M_clouds/1e5
print M/M_clouds*1e5

# calculate number of bound systems that are above the mass limit (note for N>2 we miss the mass of any extra staellites)
print len(df)
print len(df[(df['m1(kg)'])>=M])
print len(df[(df['m1(kg)']+df['m2(kg)'])>=M])
print len(df_unique)
print len(df_unique[(df_unique['m1(kg)'])>=M])
print len(df_unique[(df_unique['m1(kg)']+df_unique['m2(kg)'])>=M])

if compile==1:

    for i,run_dir in enumerate(df_unique['run_dir']):
        split_run_dir=run_dir.split("/")
        set=split_run_dir[-2]
        dir=split_run_dir[-1]
        dir_path="{}/{}/{}".format(dat_path,set,dir)
        print dir_path
        dat_files=pf.create_file_list(dir_path)
        dat_i=str(int(df_unique.iloc[i]['file_largest']))
        dat_files=[f for f in dat_files if dat_i in f]
        dat_files.sort()
        dat_file=dat_files[-1]
        t,df_dat=pf.load_dat_file("{}/{}".format(dir_path,dat_file))
        df_rp=pf.load_run_params("{}/run_params_0.txt".format(dir_path))

        Ntot=float(df_rp['N_tot'].iloc[0])
        rho=float(df_rp['rho(kgm-3)'].iloc[0])
        M_tot=float(df_rp['M_tot(kg)'].iloc[0])
        mp=M_tot/Ntot #kg
        # print mp

        print len(df_dat)
        m=numpy.array(df_dat['m(kg)']).astype(float)
        m_norm=m/mp
        m_test=M/M_tot*Ntot

        m_mag.append(len(m_norm[m_norm>=m_test]))
        M_cloud.append(M_tot)

        m_norm_avg+=len(m_norm[m_norm>=m_test])
        count+=1
        # exit()

    numpy.save("plot_binary_size_dist_hack_m_mag.npy",m_mag)
    numpy.save("plot_binary_size_dist_hack_M_cloud.npy",M_cloud)

m_mag=numpy.load("plot_binary_size_dist_hack_m_mag.npy")
M_cloud=numpy.load("plot_binary_size_dist_hack_M_cloud.npy")

for i in range(len(M_cloud)):
    print M_cloud[i],m_mag[i]

# m_norm_avg=m_norm_avg/count
# print m_norm_avg
# print numpy.sort(m_mag)
# print M_cloud
print numpy.mean(m_mag)
print numpy.median(m_mag)

cloud_masses=numpy.unique(M_cloud)
print cloud_masses
for M_c in cloud_masses:
    print M_c,numpy.mean(m_mag[M_cloud==M_c]),numpy.median(m_mag[M_cloud==M_c])

# plot number of particles above mass limit
fig = pyplot.figure()
gs = gridspec.GridSpec(1,1)
ax1 = pyplot.subplot(gs[0,0])

ax1.scatter(numpy.log10(M_cloud),m_mag)
ax1.set_xlabel("log M_cloud")
ax1.set_ylabel("n particles mass > mag 25 ")
pyplot.show()
