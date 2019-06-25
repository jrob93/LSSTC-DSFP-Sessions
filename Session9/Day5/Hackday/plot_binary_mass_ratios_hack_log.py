'''
Scatter plot showing the binary mass ratio as a function of normalised binary mass

based on APR report fig 4, see: /Users/jrobinson/grav_cloud/analysis_stuff/binary_mass_ratios.py
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
import astropy.stats.bayesian_blocks as bb

from sklearn.neighbors import KernelDensity

def kde_sklearn(data, grid, bandwidth = 1.0, **kwargs):
    kde_skl = KernelDensity(bandwidth = bandwidth, **kwargs)
    kde_skl.fit(data[:, numpy.newaxis])
    log_pdf = kde_skl.score_samples(grid[:, numpy.newaxis]) # sklearn returns log(density)

    return numpy.exp(log_pdf)

path="/Users/jrobinson/xq1_grav_cloud/binary_stability/orbit_results/orbit_results_plots"
fname_dfs=["df_plot_100_all_stable.txt"]

for fname_df in fname_dfs:
    df=pd.read_csv("{}/{}".format(path,fname_df),sep="\t",index_col=0) # orbits post selection

    # only plot the simple binary systems
    df=df[df['N_sys']==2]

    # # Only include f values 3,10,30
    # df=df[(df['f']<100.0) & (df['f']>1.0)]

    fig = pyplot.figure()
    gs = gridspec.GridSpec(2,2,width_ratios=[1,0.2],height_ratios=[0.2,1],wspace=0.0, hspace=0.0)
    # gs = gridspec.GridSpec(2,2)
    ax1 = pyplot.subplot(gs[1,0])
    ax2 = pyplot.subplot(gs[0,0],sharex=ax1)
    ax3 = pyplot.subplot(gs[1,1],sharey=ax1)

    ax2.tick_params(labelbottom=False)
    ax3.tick_params(labelleft=False)
    # ax2.yaxis.set_visible(False)
    # ax3.xaxis.set_visible(False)

    size=3.30709 # 84mm in inches
    # fig.set_size_inches(2.0*size,2.0*size)
    fig.set_size_inches(size,size)

    print len(df)
    print len(df[~numpy.isnan(df['m1(kg)'])])
    df=df[~numpy.isnan(df['m1(kg)'])] # drop entries without binaries
    print list(df)

    df['m2/m1']=numpy.array(df['m2(kg)']).astype(float)/numpy.array(df['m1(kg)']).astype(float) # move this to a separate script

    # plot mass ratio against normalised binary mass, with twin axis for delta mag

    # ax1_twin = ax1.twinx()

    # magnitude variable
    d=44.0 #object distance (AU)
    p1=0.08 # albedo
    p2=0.08
    C=664.5e3 # constant for V band
    d0=1.0

    for i,M_tot in enumerate(numpy.unique(numpy.array(df['M_tot(kg)']).astype(float))):
        df2=df[df['M_tot(kg)']==M_tot]
        m1=numpy.array(df2['m1(kg)']).astype(float)
        m2=numpy.array(df2['m2(kg)']).astype(float)
        m2m1=numpy.array(df2['m2/m1']).astype(float)
        Mtot=numpy.array(df2['M_tot(kg)']).astype(float)

        ax1.scatter(numpy.log10(m2m1),numpy.log10((m1+m2)/Mtot),
        edgecolors=pf.pyplot_colours[i],facecolors='none',
        label="M_c={:.2e}kg".format(M_tot),
        s=50,
        alpha=0.5)

        # we set up ax2 to have the exact same y axis as ax1
        # ax1_twin.scatter((m1+m2)/Mtot,m2m1,marker='o',color='None')

    # # we get the ax1_twin y ticks, remove the m2/m1=0 and add m2/m1=1e-3
    # # note that the ylims must be explicitly preserved
    # lim = ax1_twin.get_ylim()
    # ax2_yticks=list(ax1_twin.get_yticks())
    # print ax2_yticks
    # ax2_yticks.remove(0)
    # print ax2_yticks
    # ax1_twin.set_yticks(ax2_yticks + [1e-3])
    # ax1_twin.set_ylim(lim)
    #
    # print ax1_twin.get_yticks()
    # print 5.0*numpy.log10(numpy.sqrt(p1/p2)*(1.0/numpy.array(ax1_twin.get_yticks())))
    #
    # # use this function to convert an m2/m1 value into a delta mag value
    # # FuncFormatter can be used as a decorator
    # @ticker.FuncFormatter
    # def major_formatter(x, pos):
    #     # return "[%.2f]" % x
    #     if x==0:
    #         return ""
    #     else:
    #         return "{:.2f}".format(5.0*numpy.log10(numpy.sqrt(p1/p2)*(1.0/(x)**(1.0/3.0))))
    #
    # ax1_twin.yaxis.set_major_formatter(major_formatter)
    # ax1_twin.set_ylabel('delta mag')

    x_data=numpy.log10(m2m1)
    y_data=numpy.log10((m1+m2)/Mtot)

    x_min,x_max=ax1.get_xlim()
    y_min,y_max=ax1.get_ylim()
    print x_min,x_max

    # # Add the histograms
    # n_bins=50 #'auto'
    # n,bins, _ =ax2.hist(x_data,density=True,bins=n_bins)
    # print numpy.sum(n*numpy.diff(bins))
    # ax3.hist(y_data,density=True,bins=n_bins,orientation='horizontal')

    # add KDE
    bw=0.2
    grid_x=numpy.linspace(-3.0,0.0)
    PDF_x = kde_sklearn(x_data, grid_x,bandwidth=bw)
    ax2.plot(grid_x,PDF_x,color="k")
    grid_y=numpy.linspace(y_min,0.0)
    PDF_y = kde_sklearn(y_data, grid_y,bandwidth=bw)
    ax3.plot(PDF_y,grid_y,color="k")

    print PDF_x
    print numpy.diff(grid_x)[0]

    from scipy.integrate import simps
    # Compute the area using the composite Simpson's rule.
    # area = simps(PDF_x, grid_x)
    area = simps(PDF_x, dx=numpy.diff(grid_x)[0])
    print("area =", area)
    # area = simps(PDF_y, grid_y)
    area = simps(PDF_y, dx=numpy.diff(grid_y)[0])
    print("area =", area)

    # Set axis labels
    # ax1_twin.set_yticklabels([])
    # ax1_twin.set_xticklabels([])

    # ax1.set_ylabel('$m_2/m_1$')
    # ax1.set_xlabel("$(m_2+m_1)/M_\\text{tot}$")
    ax1.set_xlabel('m_2/m_1')
    ax1.set_ylabel("(m_2+m_1)/M_tot")
    ax2.set_ylabel("P.D.")
    ax3.set_xlabel("P.D.")
    ax1.legend(prop={'size': 6})

    pyplot.tight_layout()

    #save the figure
    script_name=os.path.basename(__file__).split('.')[0]
    picname="{}_{}.png".format(script_name,fname_df.split("/")[-1].split(".")[0])
    print "save {}".format(picname)
    pyplot.savefig(picname)

    picname="{}_{}.pdf".format(script_name,fname_df.split("/")[-1].split(".")[0])
    print "save {}".format(picname)
    pyplot.savefig(picname)

    pyplot.show()
    # pyplot.close()
