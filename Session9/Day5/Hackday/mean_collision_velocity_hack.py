'''
distribution of collisional velocities
'''

from matplotlib import rc
rc('font',**{'size':9})

import numpy
import pandas as pd
import matplotlib.pyplot as pyplot
import matplotlib.gridspec as gridspec
import py_func as pf
import os

# use LaTeX fonts in the plot
pyplot.rc('text', usetex=True)
pyplot.rc('font', family='serif')

from sklearn.neighbors import KernelDensity

def kde_sklearn(data, grid, bandwidth = 1.0, **kwargs):
    kde_skl = KernelDensity(bandwidth = bandwidth, **kwargs)
    kde_skl.fit(data[:, numpy.newaxis])
    log_pdf = kde_skl.score_samples(grid[:, numpy.newaxis]) # sklearn returns log(density)

    return numpy.exp(log_pdf)

error_list=[]
t_max=1e2*pf.year_s
t_cut=t_max/2.0

# ls=['-.',':','--']
ls=[(0, (3, 1, 1, 1)),':','--']

df_rel=pd.read_csv("/Users/jrobinson/grav_cloud/python_stuff/dt_run_time_analysis/df_mean_collision_velocity.txt",sep="\t",index_col=0)
df_rel=df_rel.dropna()

print df_rel

print "errors: {}".format(error_list)

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

gs = gridspec.GridSpec(1, 1)
ax1 = pyplot.subplot(gs[0,0])

ax1.set_xlabel('$\\mathrm{{median}}~v_{{\\mathrm{{rel}}}}~(\\mathrm{{ms}}^{{-1}})$')
ax1.set_ylabel('$\\mathrm{{probability}}~\\mathrm{{density}}$')

for i,M_tot in enumerate(numpy.unique(numpy.array(df_rel['M_tot(kg)']).astype(float))):
    df=df_rel[df_rel['M_tot(kg)']==M_tot]

    v_med=numpy.array(df['v_rel_med(ms^-1)'])

    print "M_tot={}kg, median(v_median)={}".format(M_tot,numpy.median(v_med))

    x_data=v_med

    # n_bins=100
    # n,bins, _ =ax1.hist(x_data,density=True,bins=n_bins,edgecolor='k',alpha=0.5,color=pf.pyplot_colours[i])

    ax1.axvline(numpy.median(v_med),color=pf.pyplot_colours[i],alpha=0.5,zorder=0)

    # add KDE
    N_bin=500
    # grid_x=numpy.linspace(numpy.amin(x_data),numpy.amax(x_data),N_bin)
    grid_x=numpy.linspace(-10,80,N_bin)
    bw_x=1.06*numpy.std(x_data)*(len(x_data)**(-0.2)) #silverman's rule
    PDF_x = kde_sklearn(x_data, grid_x,bandwidth=bw_x)
    ax1.plot(grid_x,PDF_x,color=pf.pyplot_colours[i],linestyle=ls[i],label="$M_\\mathrm{{c}}=$~{:.2e}$~\\mathrm{{kg}}$".format(M_tot))
    # ax1.scatter(grid_x,PDF_x,color=pf.pyplot_colours[i])

ax1.set_xlim(0,80)

ax1.legend(prop={'size': 6})

pyplot.tight_layout()

script_name=os.path.basename(__file__).split('.')[0]
picname="{}.png".format(script_name)
print "save {}".format(picname)
pyplot.savefig(picname,bbox_inches='tight',pad_inches=0.0)

picname="{}.pdf".format(script_name)
print "save {}".format(picname)
pyplot.savefig(picname,bbox_inches='tight',pad_inches=0.0)

# pyplot.show()
pyplot.close()
