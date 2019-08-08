'''
This script looks at (size of primary particle in collision)/(timestep*collisional velocity) as a function of simulation time
Load the final data set as a dataframe, and select runs with the highest timestep.
Select runs and then look at their collision velocities.
Load all the collision files (from local repo)

Note that some runs (cloud_runs_slurm/cloud_runs_slurm_189,cloud_runs_slurm/cloud_runs_slurm_194) have unusual collision files...?
'''
from matplotlib import rc
rc('font',**{'size':9})

import numpy
import pandas as pd
import matplotlib.pyplot as pyplot
import matplotlib.gridspec as gridspec

# use LaTeX fonts in the plot
pyplot.rc('text', usetex=True)
pyplot.rc('font', family='serif')

import py_func as pf
import os

# df=pd.read_csv("/Users/jrobinson/grav_cloud/orbit_results/particle_mass_distribution/df_cloud_runs_data_all.txt",sep="\t",index_col=0) # orbits post selection
df=pd.read_csv("/Users/jrobinson/grav_cloud/python_stuff/orbit_search_faster/orbit_search_results/df_orb_all_100_select.txt",sep="\t",index_col=0) # orbits post selection

run_path='/Users/jrobinson/cloud_runs_data/data/jakita_raid2/jer/grav_cloud'
save_path='.'

w_mass=0
script_name=os.path.basename(__file__).split('.')[0]

# run_path='/Users/jrobinson/cloud_runs_slurm_unusual_collision_restarts'
# save_path=run_path

dirs=numpy.unique(df['run_dir'])
# dirs=[d for d in dirs if ("/cloud_runs_fix_dt_2/" in d) or ("/cloud_runs_slurm/" in d)]
# dirs=[d for d in dirs if ("/cloud_runs_fix_dt_2/" in d)]
# dirs=[d for d in dirs if ("/cloud_runs_slurm/" in d)]
df_test=df[numpy.isin(df['run_dir'],dirs)][['run','f','X','R_eq(m)']]
print df_test.drop_duplicates()
dirs=["/".join(d.split("/")[-2:]) for d in dirs]
dirs.sort()
# print dirs

col_errors=[]

# dirs=["cloud_runs_slurm/cloud_runs_slurm_189","cloud_runs_slurm/cloud_runs_slurm_194"]
# dirs=["cloud_runs_fix_dt_2/226_cloud_order_kelvin_fix_dt_2"]
# dirs=["cloud_runs_fix/048_cloud_order_kelvin_fix"]
# dirs=["cloud_runs_slurm_189","cloud_runs_slurm_194"]
dirs=["cloud_runs_fix/022_cloud_order_kelvin_fix"]

for d in dirs:

    d="{}/{}".format(run_path,d)
    print d
    # continue
    # try:
    #     rp="{}/run_params_0.txt".format(d)
    #     df_rp=pf.load_run_params(rp)
    # except:
    #     continue
    # Always load the most recent run params file in case of restarts
    files=next(os.walk(d))[2] #retrieve the files in the run directory
    rp_files=[fi for fi in files if (fi.endswith(".txt") and fi.startswith("run_params"))]
    rp_files.sort()
    print "load {}".format(rp_files[-1])
    df_rp=pf.load_run_params("{}/{}".format(d,rp_files[-1]))

    fig = pyplot.figure() #open figure once

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

    gs = gridspec.GridSpec(2, 1, height_ratios=[0.05,1],wspace=0.0,hspace=0.1)
    ax1 = pyplot.subplot(gs[1,0])
    ax2 = pyplot.subplot(gs[0,0])

    # find all collision files and combine
    # files=next(os.walk(d))[2]
    coll_files = [ fi for fi in files if 'collision' in fi]
    print coll_files

    # Load the all collisions, accounting for duplicates
    i=0
    for c_file in coll_files:
        cf="{}/{}".format(d,c_file)
        if i==0:
            try:
                df_coll=pf.load_collision_file(cf)
            except:
                print "error loading collision file"
                col_errors.append(d)
                try:
                    df_coll=pf.load_unusual_collision_files(cf)
                except:
                    print "restart error"
                # continue
            # df_coll=pf.load_collision_file(cf)
            print len(df_coll)
        else:
            try:
                _df_coll=pf.load_collision_file(cf)
            except:
                print "error loading collision file"
                col_errors.append(d)
                try:
                    df_coll=pf.load_unusual_collision_files(cf)
                except:
                    print "restart error"
                # continue
            print len(_df_coll)
            df_coll=df_coll.append(_df_coll)
        df_coll=df_coll.drop_duplicates()
        i+=1

    dt=(df_rp.iloc[0]['dt(s)']).astype(float)
    M_tot=df_rp.iloc[0]['M_tot(kg)'].astype(float)
    N_tot=df_rp.iloc[0]['N_tot'].astype(float)
    mp=M_tot/N_tot
    v_i=numpy.array(df_coll[['vxi(ms^-1)','vyi(ms^-1)','vzi(ms^-1)']])
    v_j=numpy.array(df_coll[['vxj(ms^-1)','vyj(ms^-1)','vzj(ms^-1)']])
    N_c=len(df_coll)

    rho=df_rp.iloc[0]['rho(kgm-3)'].astype(float)
    f=df_rp.iloc[0]['f'].astype(float)
    Req=df_rp.iloc[0]['R_eq(m)'].astype(float)
    X=df_rp.iloc[0]['X'].astype(float)
    m1=numpy.array(df_coll['mi(kg)'])

    print "dt = {}, rho = {}, f = {}, R_eq = {}".format(dt,rho,f,Req)

    r1=numpy.power(3.0*m1/(4.0*numpy.pi*rho),(1.0/3.0))*f # from the mass of the primary in collison, calculate the INFLATED RADIUS

    v_rel=numpy.linalg.norm(v_i-v_j,axis=1)
    t_coll=numpy.array(df_coll['t(s)'])

    len_ratio=(dt*v_rel)/r1

    if 'tree' in cf:
        l="Tree: dt={},N_c={}".format(dt,N_c)
    elif 'line' in cf:
        l="Line: dt={},N_c={}".format(dt,N_c)
    else:
        l="dt={},N_c={}".format(dt,N_c)
    #mass:color scale
    color=(numpy.log10(m1)-numpy.log10(mp))/(numpy.log10(M_tot)-numpy.log10(mp))

    #set size scales for particles
    s_min=1e0
    s_max1=2e1
    s_max2=2e2#e0
    r_min=Req*(N_tot**(-1.0/3.0))*f
    r_max=Req*f
    print "particle sizes: ",r_min,r_max
    m_s = (s_max1-s_min)/(r_max-r_min)
    m_s2=(s_max2-s_min)/(r_max**2.0-r_min**2.0)
    c2=s_min-(m_s2*(r_min**2.0))

    size1 = (m_s*r1)+(s_min-(m_s*r_min))

    m_size=1
    alph=1.0
    s1=ax1.scatter(t_coll,len_ratio,label=l,s=m_size,c=color,edgecolor=None,alpha=alph, vmin=0,vmax=1,rasterized=True)

    cbar1=fig.colorbar(s1,ax2,use_gridspec=True,orientation='horizontal')
    ax2.set_ylabel('$m_\\mathrm{{rel}}$')
    ax2.yaxis.set_label_coords(-0.14,1.02)
    ax2.xaxis.tick_top()

    ax1.set_xlabel('$t~(\\mathrm{{s}})$')
    ax1.set_ylabel('$dt v_\\mathrm{{rel}} / r_1$')
    # fig.suptitle('Ratio of collisional distance to primary particle size, vs time')

    criterion_frac=float(len(len_ratio[len_ratio<1.0]))/float(len(len_ratio))
    print "fraction within criterion = {}".format(criterion_frac)
    title="dt = {:.1e} s, f = {:d}, X = {}, R_eq = {:.1e}m, within criterion: {:.2}".format(dt,int(f),X,Req,criterion_frac)
    # ax1.set_title(title)

    if w_mass==1:
        # add the N_mass largest particles to the plot
        N_mass=100
        dat_files=[fi for fi in files if (fi.endswith(".txt") and fi.startswith("dat"))]
        dat_files.sort()
        t,df_dat=pf.load_dat_file("{}/{}".format(d,dat_files[-1]))
        df_dat=df_dat.sort_values(by=['m(kg)'],ascending=False)
        # find the lowest mass of particle that we are interested in at the end of the sim (Account for >N_mass particles)
        if len(df_dat)>=N_mass:
            m_min=float(df_dat.iloc[N_mass]['m(kg)'])
        else:
            m_min=float(df_dat.iloc[-1]['m(kg)'])
        _df_coll=df_coll[df_coll['mi(kg)']>m_min]
        m1=numpy.array(_df_coll['mi(kg)'])
        r1=numpy.power(3.0*m1/(4.0*numpy.pi*rho),(1.0/3.0))*f # from the mass of the primary in collison, calculate the INFLATED RADIUS
        v_i=numpy.array(_df_coll[['vxi(ms^-1)','vyi(ms^-1)','vzi(ms^-1)']])
        v_j=numpy.array(_df_coll[['vxj(ms^-1)','vyj(ms^-1)','vzj(ms^-1)']])
        v_rel=numpy.linalg.norm(v_i-v_j,axis=1)
        t_coll=numpy.array(_df_coll['t(s)'])
        len_ratio=(dt*v_rel)/r1
        ax1.scatter(t_coll,len_ratio,label="{} most massive particles".format(N_mass),s=m_size,edgecolor='r',facecolor=None,rasterized=True)

        picname="{}/{}_{}_{}_w_mass.png".format(save_path,script_name,d.split('/')[-2],d.split('/')[-1])

    else:
        picname_png="{}/{}_{}_{}.png".format(save_path,script_name,d.split('/')[-2],d.split('/')[-1])
        picname_pdf="{}/{}_{}_{}.pdf".format(save_path,script_name,d.split('/')[-2],d.split('/')[-1])

    pyplot.tight_layout()

    #save the figure
    print "save {}".format(picname_png)
    print "save {}".format(picname_pdf)
    pyplot.savefig(picname_png, bbox_inches='tight')
    pyplot.savefig(picname_pdf, bbox_inches='tight')

    # ax1.legend()

    # pyplot.show()
    pyplot.close()

    # break

# When things are restarted the collision files may be spread out
print "\nerrors loading collision files:\n{}".format(col_errors)
