'''
Scatter plot showing the binary mass ratio as a function of normalised binary mass

based on APR report fig 4, see: /Users/jrobinson/grav_cloud/analysis_stuff/binary_mass_ratios.py
'''

from matplotlib import cm
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
from scipy import stats
import matplotlib.tri as tri
from scipy.spatial import ConvexHull

# use LaTeX fonts in the plot
pyplot.rc('text', usetex=True)
pyplot.rc('font', family='serif')

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
    df_bin=df[df['N_sys']==2]
    df_mult=df[df['N_sys']>2]

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
    # gs = gridspec.GridSpec(2,2,width_ratios=[1,(0.2)/s_x],height_ratios=[0.2/s_y,1],wspace=0.0, hspace=0.0)
    gs = gridspec.GridSpec(1,1)

    # gs = gridspec.GridSpec(3,2,width_ratios=[1,0.2],height_ratios=[0.2,1,1])
    # ax1 = pyplot.subplot(gs[1,0])
    # ax2 = pyplot.subplot(gs[0,0],sharex=ax1)
    # ax3 = pyplot.subplot(gs[1,1],sharey=ax1)
    ax1 = pyplot.subplot(gs[0,0])

    # ax2.tick_params(labelbottom=False)
    # ax3.tick_params(labelleft=False)

    for count,df in enumerate([df_bin,df_mult]):

        print len(df)
        print len(df[~numpy.isnan(df['m1(kg)'])])
        df=df[~numpy.isnan(df['m1(kg)'])] # drop entries without binaries
        print list(df)

        df['m2/m1']=numpy.array(df['m2(kg)']).astype(float)/numpy.array(df['m1(kg)']).astype(float) # move this to a separate script

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

        x_data=(m2m1)
        y_data=((m1+m2)/Mtot)

        # for i in range(len(x_data)):
        #     ax4.scatter(x_data[i],y_data[i],
        #     edgecolors=col_list[i],facecolors='none',
        #     marker=marker_list[i],
        #     s=50,
        #     alpha=1.0)

        x_data=numpy.log10(m2m1)
        y_data=numpy.log10((m1+m2)/Mtot)

        if count==0:
            marker_size=50
            marker_alpha=1.0
        else:
            marker_size=25
            marker_alpha=0.5

        for i in range(len(x_data)):

            if count==0:
                marker_label="$M_\\mathrm{{c}}=$~{:.2e}$~\\mathrm{{kg}}$".format(Mtot[i])
            else:
                marker_label=None

            print marker_label

            ax1.scatter(x_data[i],y_data[i],
            edgecolors=col_list[i],facecolors='none',
            marker=marker_list[i],
            s=marker_size,
            alpha=marker_alpha,
            label=marker_label)

        print "max values: ",numpy.amax(m2m1),numpy.amax((m1+m2)/Mtot)
        print "min values: ",numpy.amin(m2m1),numpy.amin((m1+m2)/Mtot)

        if count==0:
            # ensure we pass the entire data set, ordered
            df_kde=df.sort_values(by=['m2/m1'],ascending=False)
            print len(df_kde)
            x_data=numpy.log10(numpy.array(df_kde['m2/m1']).astype(float))
            y_data=numpy.log10((numpy.array(df_kde['m1(kg)']).astype(float)+numpy.array(df_kde['m2(kg)']).astype(float))/numpy.array(df_kde['M_tot(kg)']).astype(float))
            I_bin=numpy.array(df_kde['I(rad)']).astype(float)

            x_min,x_max=ax1.get_xlim()
            y_min,y_max=ax1.get_ylim()
            print "x max: ",x_min,x_max
            print "y max: ",y_min,y_max

            # # add KDE
            N_bin=500
            grid_x=numpy.linspace(-4.0,0.0,N_bin)
            grid_y=numpy.linspace(-5.0,0.0,N_bin)
            #
            # # bw=0.1
            # bw_x=0.1/2.0
            # bw_y=0.1
            #
            # PDF_x = kde_sklearn(x_data, grid_x,bandwidth=bw_x)
            # ax2.plot(grid_x,PDF_x,color="k")
            # PDF_y = kde_sklearn(y_data, grid_y,bandwidth=bw_y)
            # ax3.plot(PDF_y,grid_y,color="k")
            #
            # # print PDF_x
            # # print numpy.diff(grid_x)[0]
            #
            # from scipy.integrate import simps
            # # Compute the area using the composite Simpson's rule.
            # # area = simps(PDF_x, grid_x)
            # area = simps(PDF_x, dx=numpy.diff(grid_x)[0])
            # print("area =", area)
            # # area = simps(PDF_y, grid_y)
            # area = simps(PDF_y, dx=numpy.diff(grid_y)[0])
            # print("area =", area)
            #
            # # # Add lines denoting rough classes
            # # ax1.vlines([-2.0], 0, 1, transform=ax1.get_xaxis_transform(), colors='r')
            # # ax1.hlines([-2.0], 0, 1, transform=ax1.get_yaxis_transform(), colors='r')

            #-------------------------------------------------------------------------------
            # Perform cluster search
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import MinMaxScaler, StandardScaler
            from sklearn import metrics
            from sklearn.metrics.pairwise import euclidean_distances
            from sklearn.cluster import DBSCAN

            dbscan_eps=0.5#0.075
            print "eps={}".format(dbscan_eps)

            # DBscan search, SEARCH RESULTS DEPEND ENTIRELY ON dbscan_eps VARIABLE
            df_fit=pd.DataFrame(numpy.array([x_data,y_data,I_bin]).T,columns=['x','y','I(rad)'])
            # print df_fit

            X=df_fit[['x','y']] #select only position labels (would mass help?)
            print X,numpy.amin(X['x']),numpy.amax(X['x']),numpy.std(X['x']),numpy.amin(X['y']),numpy.amax(X['y']),numpy.std(X['y'])
            X = StandardScaler().fit_transform(X) #rescale the positions: Standardize features by removing the mean and scaling to unit variance
            print X,numpy.amin(X[:,0]),numpy.amax(X[:,0]),numpy.std(X[:,0]),numpy.amin(X[:,1]),numpy.amax(X[:,1]),numpy.amax(X[:,1])
            # exit()

            db = DBSCAN(eps=dbscan_eps, min_samples=10).fit(X) # do DBSCAN, eps: The maximum distance between two samples for them to be considered as in the same neighborhood.
            core_samples_mask = numpy.zeros_like(db.labels_, dtype=bool)
            # print core_samples_mask
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_
            # print core_samples_mask
            # print labels,len(labels)
            # print len(core_samples_mask)
            # print sum(core_samples_mask)

            # Number of clusters in labels, ignoring noise if present.
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)

            print('Estimated number of clusters: %d' % n_clusters_)
            print('Estimated number of noise points: %d' % n_noise_)

            unique_labels = set(labels)
            print "labels = ",unique_labels

            # cluster_cols=['r','b','g','y','p','p']
            cluster_markers=[(3,2,0),(4,2,0),(5,2,0)]
            cluster_labels=['class (i)','class (ii)','class (iii)']
            # unique_labels_order=[0,2,1,-1]
            unique_labels_order=unique_labels
            # cluster_linestyle=[':','--','-.']
            cluster_linestyle=[[3,1,3,1],[3,1,1,1],[3,1,1,1,1,1]]

            # cont_cols=['r','b','g']
            colormap = cm.viridis# LinearSegmentedColormap
            Ncolors=[0,0.5,1.0]
            cont_cols = [colormap(int(x*colormap.N)) for x in Ncolors]
            print cont_cols

            for i,k in enumerate(unique_labels_order):

                class_member_mask = (labels == k)

                # cluster members
                df_cluster=df_fit[class_member_mask & core_samples_mask] # note that the core mask removes some points
                # df_cluster=df_fit[class_member_mask]

                print len(df_cluster)

                # Plot clusters

                if k==-1:
                    continue
                else:

                    # ax1.scatter(df_cluster['x'],df_cluster['y'],s=50,c='k',marker=cluster_markers[i],alpha=0.5,label=cluster_labels[i])

                    # Add contours to data
                    Z_data=numpy.zeros(len(df_cluster['x']))+k
                    # triang = tri.Triangulation(df_cluster['x'], df_cluster['y'])
                    # ax1.tricontour(df_cluster['x'], df_cluster['y'], Z_data, 15, linewidths=10, colors='k',zorder=0)
                    # ax1.tricontourf(df_cluster['x'], df_cluster['y'], Z_data,color=cont_cols[i],
                    # zorder=0,alpha=0.2,label=cluster_labels[i])
                    # ax1.scatter(df_cluster['x'], df_cluster['y'],color=cont_cols[i],
                    # zorder=0,label=cluster_labels[i])

                    # use scipy to find the outer edges
                    allPoints=numpy.column_stack((df_cluster['x'],df_cluster['y']))
                    hullPoints = ConvexHull(allPoints)
                    # ax1.plot(df_cluster['x'].iloc[hullPoints.vertices], df_cluster['y'].iloc[hullPoints.vertices], 'r--', lw=2)
                    for simplex in hullPoints.simplices:
                        ax1.plot(allPoints[simplex, 0], allPoints[simplex, 1],
                        color=cont_cols[i],label=cluster_labels[i],zorder=0,dashes=cluster_linestyle[i])

                    if k==1:
                        # Find the inclination properties of class iii systems
                        print "\n\ninclination of {} systems".format(cluster_labels[i])
                        print "class iii systems"
                        df_iii=df_fit[class_member_mask & core_samples_mask]
                        # print 10**numpy.amin(df_iii['x']),10**numpy.amin(df_iii['y'])
                        I_bin_iii=numpy.array(df_iii['I(rad)'])
                        y_data_iii=numpy.degrees(I_bin_iii)
                        print y_data_iii
                        # print numpy.degrees(I_bin_iii)
                        print len(y_data_iii[y_data_iii<90.0]),len(y_data_iii[y_data_iii>=90.0])
                        print "simulated retrograde/prograde={}".format(float(len(y_data_iii[y_data_iii>=90.0]))/float(len(y_data_iii[y_data_iii<90.0])))
                        print "simulated retrograde fraction={}, mean={}, median={}, std={}".format(
                        float(len(y_data_iii[y_data_iii>=90.0]))/float(len(y_data_iii)),
                        numpy.mean(y_data_iii[y_data_iii>=90.0]),
                        numpy.median(y_data_iii[y_data_iii>=90.0]),
                        numpy.std(y_data_iii[y_data_iii>=90.0]))
                        print "simulated prograde fraction={}, mean={}, median={}, std={}".format(
                        float(len(y_data_iii[y_data_iii<90.0]))/float(len(y_data_iii)),
                        numpy.mean(y_data_iii[y_data_iii<90.0]),
                        numpy.median(y_data_iii[y_data_iii<90.0]),
                        numpy.std(y_data_iii[y_data_iii<90.0]))
                        # ax1.scatter(df_iii["x"],df_iii['y'],color="k",marker="x")
                        print "\n"
            # class_member_mask = (labels == -1)
            # # cluster members
            # df_cluster=df_fit[class_member_mask & core_samples_mask]
            # # Plot clusters
            # ax1.scatter(df_cluster['x'],df_cluster['y'],s=10,c='k',alpha=0.5)
            # ax4.scatter(10**numpy.array(df_cluster['x']),10**numpy.array(df_cluster['y']),s=10,c='k',alpha=0.5)

            #-------------------------------------------------------------------------------

            # fit linear clusters with straight line

            class_member_mask = ((labels == 0) | (labels == 2))
            # cluster members
            df_cluster=df_fit[class_member_mask & core_samples_mask]
            N_obs=len(df_cluster)
            # ax1.scatter(df_cluster['x'],df_cluster['y'],s=10,c='k')

            # do linear fit
            slope, intercept, r_value, p_value, std_err = stats.linregress(df_cluster['x'],df_cluster['y'])
            print "slope={}\nintercept={}\nr_value={}\np_value={}\nstd_err={}".format(slope,intercept,r_value,p_value,std_err)
            print "log space: y = {}*x + {}".format(slope,intercept)
            print "linear space: y = {}*x^{}".format(intercept,slope)
            if intercept<0:
                sign="-"
            else:
                sign="+"
            ax1.plot(grid_x,pf.line_fit(grid_x,slope,intercept),c='k',label="$y={:.2f}x{}{:.2f}$".format(slope,sign,numpy.absolute(intercept)),zorder=0)
            # ax4.plot(10**grid_x,10**pf.line_fit(grid_x,slope,intercept),c='k')

            # # try chi square?
            # # chisq,p=stats.chisquare(numpy.array(df_cluster['y']), f_exp=pf.line_fit(numpy.array(df_cluster['x']),slope,intercept), ddof=2)
            # f_obs=numpy.array(df_cluster['y'])
            # f_exp=pf.line_fit(numpy.array(df_cluster['x']),slope,intercept)
            # chisq=numpy.sum(((f_obs-f_exp)**2.0)/f_exp)
            # ax1.plot(numpy.array(df_cluster['x']),pf.line_fit(numpy.array(df_cluster['x']),slope,intercept),c='k')
            # print "chisq={}".format(chisq)

            #-------------------------------------------------------------------------------
            # the smallest possible secondary is m2 = 2e-5 M_c, for the range of mass ratios what is m1?
            m2_min=2e-5 # units of cloud mass
            mass_ratios=numpy.logspace(-3,0)
            m1_min=m2_min/mass_ratios
            print mass_ratios
            print m1_min
            # print numpy.log10(mass_ratios),numpy.log10(m2_min+m1_min)
            ax1.plot(numpy.log10(mass_ratios),numpy.log10(m2_min+m1_min),c='k',linestyle=":",label="selection limit")
            # ax4.plot(mass_ratios,m2_min+m1_min,c='k',linestyle=":")
            #-------------------------------------------------------------------------------

            #-------------------------------------------------------------------------------

            padding_y=0.1
            padding_x=padding_y
            ax1.set_xlim(numpy.amin(x_data)-padding_x,0.0+padding_x)
            ax1.set_ylim(numpy.amin(y_data)-(2.0*padding_y),0.0+padding_y)

            ax1.set_xlabel('$\\log(m_2/m_1)$')
            ax1.set_ylabel("$\\log((m_2+m_1)/M_\\mathrm{{c}})$")
            # ax2.set_ylabel("P.D.")
            # ax3.set_xlabel("P.D.")

            # Remove duplicates from legend
            handles, labels = ax1.get_legend_handles_labels()
            from collections import OrderedDict
            by_label = OrderedDict(zip(labels, handles))

            print by_label.values()
            print by_label.keys()

            # for i in range(len(by_label)):
            #     print by_label.values(), by_label.keys()
            #     for j in range(len(by_label.keys())):
            #         print by_label.keys()[j],len(by_label.keys()[j])

            ax1.legend(by_label.values(), by_label.keys(),loc='lower left',prop={'size': 6})
            # ax1.legend(loc='lower left',prop={'size': 6})

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
