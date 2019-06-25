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

numpy.random.seed(0)

path="/Users/jrobinson/xq1_grav_cloud/binary_stability/orbit_results/orbit_results_plots"
fname_dfs=["df_plot_100_all_stable.txt"]

markers=['^','s','o']

for fname_df in fname_dfs:
    df=pd.read_csv("{}/{}".format(path,fname_df),sep="\t",index_col=0) # orbits post selection

    # only plot the simple binary systems
    df=df[df['N_sys']==2]

    # # Only include f values 3,10,30
    # df=df[(df['f']<100.0) & (df['f']>1.0)]

    fig = pyplot.figure()

    pc_tex=0.16605 # latex pc in inches
    text_width=39.0*pc_tex
    column_sep=2.0*pc_tex
    # column_width=3.30709 # 84mm in inches
    column_width=(text_width-column_sep)/2.0
    s_x=2.0
    s_y=1.0
    # fig.set_size_inches(2.0*size,2.0*size)
    # fig.set_size_inches(s_x*size,s_y*size)
    x_len=text_width+(column_sep)+(2.0*pc_tex) # add an additional couple points to fit
    y_len=x_len/2.0
    print "size: {}x{} inches".format(x_len,y_len)
    fig.set_size_inches(x_len,y_len)

    print [1,(0.2)/s_x],[0.2/s_y,1]
    gs = gridspec.GridSpec(2,2,width_ratios=[1,(0.2)/s_x],height_ratios=[0.2/s_y,1],wspace=0.0, hspace=0.0)
    # gs = gridspec.GridSpec(2,2)
    ax1 = pyplot.subplot(gs[1,0])
    ax2 = pyplot.subplot(gs[0,0],sharex=ax1)
    ax3 = pyplot.subplot(gs[1,1],sharey=ax1)

    ax2.tick_params(labelbottom=False)
    ax3.tick_params(labelleft=False)
    # ax2.yaxis.set_visible(False)
    # ax3.xaxis.set_visible(False)

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

    x_data=m2m1
    y_data=(m1+m2)/Mtot

    for i in range(len(x_data)):
        label="M_c={:.2e}kg".format(Mtot[i])
        ax1.scatter(x_data[i],y_data[i],
        edgecolors=col_list[i],facecolors='none',
        marker=marker_list[i],
        s=50,
        alpha=1.0,
        label=label)

    print "max values: ",numpy.amax(m2m1),numpy.amax((m1+m2)/Mtot)
    print "min values: ",numpy.amin(m2m1),numpy.amin((m1+m2)/Mtot)

    # # Plot each cloud mass separately for legend
    # for i,M_tot in enumerate(numpy.unique(numpy.array(df['M_tot(kg)']).astype(float))):
    #     df2=df[df['M_tot(kg)']==M_tot]
    #     m1=numpy.array(df2['m1(kg)']).astype(float)
    #     m2=numpy.array(df2['m2(kg)']).astype(float)
    #     m2m1=numpy.array(df2['m2/m1']).astype(float)
    #     Mtot=numpy.array(df2['M_tot(kg)']).astype(float)
    #
    #     x_data=m2m1
    #     y_data=(m1+m2)/Mtot
    #     print x_data,y_data
    #
    #     ax1.scatter(x_data,y_data,
    #     edgecolors=pf.pyplot_colours[i],facecolors='none',
    #     marker=markers[i],
    #     label="M_c={:.2e}kg".format(M_tot),
    #     s=50,
    #     alpha=1.0)
    #
    #     print numpy.amax(m2m1),numpy.amax((m1+m2)/Mtot)
    #
    #     # we set up ax2 to have the exact same y axis as ax1
    #     # ax1_twin.scatter((m1+m2)/Mtot,m2m1,marker='o',color='None')

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

    # ensure we pass the entire data set, ordered
    df_kde=df.sort_values(by=['m2/m1'],ascending=False)
    print len(df_kde)
    x_data=numpy.array(df_kde['m2/m1']).astype(float)
    y_data=(numpy.array(df_kde['m1(kg)']).astype(float)+numpy.array(df_kde['m2(kg)']).astype(float))/numpy.array(df_kde['M_tot(kg)']).astype(float)
    # x_data=m2m1
    # y_data=(m1+m2)/Mtot
    # print len(x_data)


    x_min,x_max=ax1.get_xlim()
    y_min,y_max=ax1.get_ylim()
    print "x max: ",x_min,x_max
    print "y max: ",y_min,y_max

    # # Add the histograms
    # n_bins=50 #'auto'
    # n,bins, _ =ax2.hist(x_data,density=True,bins=n_bins)
    # print numpy.sum(n*numpy.diff(bins))
    # ax3.hist(y_data,density=True,bins=n_bins,orientation='horizontal')

    # add KDE
    N_bin=500
    # grid_x=numpy.linspace(0.0,1.0,N_bin)
    # grid_y=numpy.linspace(0.0,y_max)
    grid_x=numpy.linspace(-0.1,1.1,N_bin)
    grid_y=numpy.linspace(-0.1,1.1,N_bin)

    bw=0.01/2.0
    PDF_x = kde_sklearn(x_data, grid_x,bandwidth=bw)
    ax2.plot(grid_x,PDF_x,color="k")
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

    # set axis limits
    padding_y=(0.1/2.0)
    padding_x=padding_y/2.5
    ax1.set_xlim(0.0-padding_x,1.0+(1.5*padding_x))
    ax1.set_ylim(0.0-padding_y,numpy.amax(y_data)+(2.0*padding_y))

    # # remove a single label
    # import matplotlib.ticker as mticker
    # def update_ticks(x, pos):
    #     if x == 0.8:
    #         return ''
    #     else:
    #         return '{:.1f}'.format(x)
    # ax1.yaxis.set_major_formatter(mticker.FuncFormatter(update_ticks))

    # ax1.set_ylabel('$m_2/m_1$')
    # ax1.set_xlabel("$(m_2+m_1)/M_\\text{tot}$")
    ax1.set_xlabel('m_2/m_1')
    ax1.set_ylabel("(m_2+m_1)/M_c")
    ax2.set_ylabel("P.D.")
    ax3.set_xlabel("P.D.")
    # ax1.legend(loc='upper right',prop={'size': 6})

    # Remove duplicates from legend
    handles, labels = ax1.get_legend_handles_labels()
    from collections import OrderedDict
    by_label = OrderedDict(zip(labels, handles))

    ax1.legend(by_label.values(), by_label.keys(),loc='upper right',prop={'size': 6})

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
