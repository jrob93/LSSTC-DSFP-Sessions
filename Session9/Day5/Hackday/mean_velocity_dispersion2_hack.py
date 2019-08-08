'''
'''

from matplotlib import rc
rc('font',**{'size':9})

import numpy
import pandas as pd
import matplotlib.pyplot as pyplot
import matplotlib.gridspec as gridspec
import sys
import py_func as pf
import os

# use LaTeX fonts in the plot
pyplot.rc('text', usetex=True)
pyplot.rc('font', family='serif')

markers=['^','s','o']

n_samp=5 # sample each simulation at 5 points
t_max=1e2*pf.year_s

time_array=numpy.linspace(0,t_max,5)

df=pd.read_csv("/Users/jrobinson/grav_cloud/orbit_results/particle_mass_distribution/df_cloud_runs_data_all.txt",sep="\t",index_col=0) # orbits post selection
run_path='/data/jakita_raid2/jer/grav_cloud'
dirs=numpy.unique(df['run_dir'])

df_test=df[numpy.isin(df['run_dir'],dirs)][['run','f','X','R_eq(m)']]
print df_test.drop_duplicates()
dirs=["/".join(d.split("/")[-2:]) for d in dirs]
dirs.sort()


df_vel=pd.read_csv("/Users/jrobinson/grav_cloud/python_stuff/dt_run_time_analysis/test_final_data_set/df_velocities.txt",sep="\t",index_col=0)
print df_vel

df_vel_mean=pd.DataFrame(columns=['M_tot(kg)','v_mean(ms^-1)','v_mean_std(ms^-1)','t(s)'])

for i,M_tot in enumerate(numpy.unique(numpy.array(df_vel['M_tot(kg)']).astype(float))):
    df=df_vel[df_vel['M_tot(kg)']==M_tot]

    for j in range(len(time_array)):
        # times=df[df['t(s)']==time_array[j]]['t(s)']
        df_times=df[numpy.isclose(df['t(s)'],time_array[j],rtol=1e-1)]
        print time_array[j],df_times
        vel=df_times['v_mean(ms^-1)']

        df_vel_mean=df_vel_mean.append(pd.DataFrame([[M_tot,numpy.mean(vel),numpy.std(vel),time_array[j]]],
        columns=['M_tot(kg)','v_mean(ms^-1)','v_mean_std(ms^-1)','t(s)']))

print df_vel_mean

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

gs = gridspec.GridSpec(1, 1)
ax1 = pyplot.subplot(gs[0,0])

ax1.set_xlabel('$t~(\\mathrm{{s}})$')
ax1.set_ylabel('$\\mathrm{{mean}}~\\mathrm{{velocity}}~(\\mathrm{{ms}}^{-1})$')

for i,M_tot in enumerate(numpy.unique(numpy.array(df_vel['M_tot(kg)']).astype(float))):
    df_plot=df_vel_mean[df_vel_mean['M_tot(kg)']==M_tot]

    # ax1.errorbar(df['t(s)'],df['v_mean(ms^-1)'],yerr=df['v_std(ms^-1)'],capsize=5,fmt='o')#,label='{} v_mean'.format(dir))
    ax1.errorbar(df_plot['t(s)'],df_plot['v_mean(ms^-1)'],yerr=df_plot['v_mean_std(ms^-1)'],
    capsize=5,fmt='o',label="$M_\\mathrm{{c}}=$~{:.2e}$~\\mathrm{{kg}}$".format(M_tot),marker=markers[i],markersize=5)#,label='{} v_mean'.format(dir))

# ax1.legend()
ax1.legend(prop={'size': 6})

pyplot.tight_layout()

#save the figure
script_name=os.path.basename(__file__).split('.')[0]
picname="{}.png".format(script_name)
print "save {}".format(picname)
pyplot.savefig(picname,bbox_inches='tight',pad_inches=0.0)
picname="{}.pdf".format(script_name)
print "save {}".format(picname)
pyplot.savefig(picname,bbox_inches='tight',pad_inches=0.0)

pyplot.close()
# pyplot.show()
