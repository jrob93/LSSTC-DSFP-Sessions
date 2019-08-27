'''
plot the orbital elements vs time time, for extra dynamical evolution.

Note this plots ALL detected orbits, these systems have not been filtered (by e.g. pf.binary_selector)
'''

import py_func as pf
import numpy
import matplotlib.pyplot as pyplot
import matplotlib.gridspec as gridspec
import glob
import os

# path="../restart_dirs"
# dir="226_cloud_order_kelvin_fix_dt_2_cloud_runs_fix_dt_2_56381"
# path="../restart_dirs_N100"
# dir="002_cloud_order_kelvin_fix_dt_2_cloud_runs_fix_dt_2_5996"
# dir="013_cloud_order_kelvin_fix_dt_2_cloud_runs_fix_dt_2_12280"
# dir="022_cloud_order_kelvin_fix_cloud_runs_fix_20669"
# dir="206_cloud_order_kelvin_fix_dt_2_cloud_runs_fix_dt_2_11921"
# path="../restart_dirs_N100_com"
#
# dirs=pf.create_dir_list(path)
# dirs=["232_cloud_order_kelvin_fix_dt_2_cloud_runs_fix_dt_2_16284","232_cloud_order_kelvin_fix_dt_2_cloud_runs_fix_dt_2_16964"]
#
# path="/Users/jrobinson/"
# dirs=["167_cloud_order_kelvin_fix_cloud_runs_fix_42810_helio"]

# dirs=["cloud_runs_slurm_189_cloud_runs_slurm_933"]
#
# path="../test_dt"
# dirs=["167_cloud_order_kelvin_fix_cloud_runs_fix_42810"]

path="../cloud_runs_helio/restart_dirs_N100_f1"
# dirs=pf.create_dir_list(path)
dirs=["011_cloud_order_kelvin_fix_cloud_runs_fix_37521"]

t_max1=(1e2)*(pf.year_s)
t_max2=(1e4+1e2)*(pf.year_s)
t_tolerance=0.99

plot_option=3

for dir in dirs:

    # if dir=="068_cloud_order_kelvin_fix_cloud_runs_fix_6513":
    #     continue

    dir_path="{}/{}".format(path,dir)

    orbfname="{}_orbit_search_faster_hel.txt".format(dir)

    df_orb=pf.load_orb_file("{}/{}".format(dir_path,orbfname))
    df_orb=df_orb[df_orb['m1(kg)']<pf.M_sun] # drop any helio orbits
    print df_orb

    list_of_files = glob.glob('{}/run_params*'.format(dir_path)) # * means all if need specific format then *.csv
    rp_file= min(list_of_files, key=os.path.getctime).split("/")[-1]
    print rp_file
    df_rp=pf.load_run_params("{}/{}".format(dir_path,rp_file))
    M_tot=df_rp.iloc[0]['M_tot(kg)'].astype(float)
    N_tot=df_rp.iloc[0]['N_tot'].astype(float)
    mp=M_tot/N_tot

    t=numpy.array(df_orb['t(s)']).astype(float)
    a=numpy.array(df_orb['a(m)']).astype(float)
    e=numpy.array(df_orb['e']).astype(float)
    I=numpy.array(df_orb['I(rad)']).astype(float)
    m1=numpy.array(df_orb['m1(kg)']).astype(float)
    m2=numpy.array(df_orb['m2(kg)']).astype(float)

    I=numpy.degrees(I)

    pri=numpy.array(df_orb['i']).astype(int)

    for i in range(len(t)):
        print t[i],pri[i],m1[i]

    fig = pyplot.figure()
    # fig.set_size_inches(15, 10)
    # gs = gridspec.GridSpec(4, 2, width_ratios=[1.0,0.1])
    gs = gridspec.GridSpec(3, 2, width_ratios=[1.0,0.1])
    ax1 = pyplot.subplot(gs[0,0])
    ax2 = pyplot.subplot(gs[1,0],sharex=ax1)
    ax3 = pyplot.subplot(gs[2,0],sharex=ax1)
    # ax4 = pyplot.subplot(gs[3,0],sharex=ax1)
    ax5 = pyplot.subplot(gs[:,1])

    ax3.set_xlabel('t (s)')
    ax1.set_ylabel('log(a (m))')
    ax2.set_ylabel('e')
    ax3.set_ylabel('I (degrees)')
    # ax4.set_ylabel('m1 (kg)')

    ax1.set_xlim(0.0,t_max2+t_max1)

    if plot_option==0:
        # mass of secondary colour scheme
        color=m2    #mass:color scale
        s1=ax1.scatter(t,numpy.log10(a),c=color)
        ax2.scatter(t,e,c=color)
        ax3.scatter(t,I,c=color)
        # ax4.scatter(t,m1)
        # ax4.scatter(t,m2)
        cbar1=fig.colorbar(s1,ax5,use_gridspec=True)
        cbar1.set_label('m2 (kg)')

    if plot_option==1:
        # mass ratio color scale
        color=m2/m1
        s1=ax1.scatter(t,numpy.log10(a),c=color,vmin=0,vmax=1)
        ax2.scatter(t,e,c=color,vmin=0,vmax=1)
        ax3.scatter(t,I,c=color,vmin=0,vmax=1)
        cbar1=fig.colorbar(s1,ax5,use_gridspec=True)
        cbar1.set_label('m2/m1')

    if plot_option==2:
        # particle mass as a fraction of cloud mass, logarthmic scale
        color=(numpy.log10(m1)-numpy.log10(mp))/(numpy.log10(M_tot)-numpy.log10(mp))
        s1=ax1.scatter(t,a,c=color,vmin=0,vmax=1)
        ax2.scatter(t,e,c=color,vmin=0,vmax=1)
        ax3.scatter(t,I,c=color,vmin=0,vmax=1)
        # ax4.scatter(t,m1,c=color,vmin=0,vmax=1)
        # ax1.scatter(t,a,c=color)
        cbar1=fig.colorbar(s1,ax5,use_gridspec=True)
        cbar1.set_label('$\\frac{\log{m}-\log{m_p}}{\log{M_{tot}}-\log{m_p}}$')

    if plot_option==3:
        # log of mass ratio colour scheme
        color=numpy.log10(m2/m1)
        marker_size=1
        s1=ax1.scatter(t,numpy.log10(a),s=marker_size,c=color,vmin=-3,vmax=0)
        ax2.scatter(t,e,c=color,s=marker_size,vmin=-3,vmax=0)
        ax3.scatter(t,I,c=color,s=marker_size,vmin=-3,vmax=0)
        # ax1.plot(t,numpy.log10(a),color="k",alpha=0.1)
        # ax2.plot(t,e,color="k",alpha=0.1)
        # ax3.plot(t,I,color="k",alpha=0.1)
        cbar1=fig.colorbar(s1,ax5,use_gridspec=True)
        cbar1.set_label('log(m2/m1)')

    # df_last_orb=df_orb.iloc[-1]
    t_last=numpy.amax(numpy.array(df_orb['t(s)']).astype(float))
    print t_last/t_max2
    if (t_last/t_max2)<t_tolerance:
        N_sys=numpy.nan
    else:
        print numpy.array(df_orb[df_orb['t(s)']==t_last][['i','j']]).flatten()
        N_sys=len(numpy.unique(numpy.array(df_orb[df_orb['t(s)']==t_last][['i','j']]).flatten()))
    fig.suptitle('{}: N_sys={}'.format(dir,N_sys))

    # # ADD A HILL RADIUS LINE TO AX1!
    # R_hill=[]
    # for i in range(len(df_orb)):
    #     orb=df_orb.iloc[i]
    #     fnum=int(orb['file'])
    #     pri=int(orb['i'])
    #     print fnum,pri

    picname="{}/{}_{}_{}.png".format(dir_path,os.path.basename(__file__).split('.')[0],dir,plot_option)
    # picname="{}_{}_{}.png".format(os.path.basename(__file__).split('.')[0],dir,plot_option)
    print "save {}".format(picname)
    pyplot.savefig(picname)

    pyplot.show()
    # pyplot.close()
