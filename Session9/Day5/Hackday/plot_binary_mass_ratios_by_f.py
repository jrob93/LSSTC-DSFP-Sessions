'''
Scatter plot showing the binary mass ratio as a function of normalised binary mass

based on APR report fig 4, see: /Users/jrobinson/grav_cloud/analysis_stuff/binary_mass_ratios.py
'''

import pandas as pd
import numpy
import py_func as pf
import matplotlib.pyplot as pyplot
import matplotlib.gridspec as gridspec
import os
import matplotlib.ticker as ticker

# "../df_cloud_runs_data_binary_stability.txt"

# fname_dfs=["/Users/jrobinson/grav_cloud/python_stuff/orbit_search_faster/orbit_search_results/df_orb_all_100_select.txt",
# "/Users/jrobinson/grav_cloud/python_stuff/orbit_search_faster/orbit_search_results/df_orb_all_200_select.txt",
# "/Users/jrobinson/grav_cloud/python_stuff/orbit_search_faster/orbit_search_results/df_orb_all_100_select_stable.txt",
# "df_plot_100_all.txt",
# "df_plot_100_all_stable.txt"]
fname_dfs=["df_plot_100_all_stable.txt"]

marker_size_factor=100.0

for fname_df in fname_dfs:
    df=pd.read_csv(fname_df,sep="\t",index_col=0) # orbits post selection

    df=df[df['N_sys']==2]

    fig = pyplot.figure()
    gs = gridspec.GridSpec(2,2)
    ax1 = pyplot.subplot(gs[0,0])
    ax2 = pyplot.subplot(gs[0,1])
    ax3 = pyplot.subplot(gs[1,0])
    ax4 = pyplot.subplot(gs[1,1])

    axes=[ax1,ax2,ax3,ax4]

    ax1.get_xaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)

    ax2.get_yaxis().set_visible(False)
    ax4.get_yaxis().set_visible(False)

    # Set axis labels
    # ax1.set_ylabel('m2/m1')
    # ax3.set_xlabel("(m2+m1)/M_tot")

    fig.text(0.5, 0.01, '(m2+m1)/M_tot', ha='center')
    fig.text(0.01, 0.5, 'm2/m1', va='center', rotation='vertical')

    f_vals = numpy.unique(numpy.array(df['f']).astype(float))

    print len(df)
    print len(df[~numpy.isnan(df['m1(kg)'])])
    df=df[~numpy.isnan(df['m1(kg)'])] # drop entries without binaries
    print list(df)

    df['m2/m1']=numpy.array(df['m2(kg)']).astype(float)/numpy.array(df['m1(kg)']).astype(float) # move this to a separate script

    # plot mass ratio against normalised binary mass
    for i,f in enumerate(f_vals):
        for M_tot in numpy.unique(numpy.array(df['M_tot(kg)']).astype(float)):
            df2=df[(df['M_tot(kg)']==M_tot) & (df['f']==f)]
            m1=numpy.array(df2['m1(kg)']).astype(float)
            m2=numpy.array(df2['m2(kg)']).astype(float)
            m2m1=numpy.array(df2['m2/m1']).astype(float)
            Mtot=numpy.array(df2['M_tot(kg)']).astype(float)
            print f,M_tot,len(df2)

            # Use marker size to represent separation
            a_R_hill=numpy.array(df2['a(m)']).astype(float)/numpy.array(df2['R_hill(m)']).astype(float)
            marker_size=a_R_hill*marker_size_factor
            if len(a_R_hill[a_R_hill>=0.5])>0:
                print "an orbit has a/R_hill>0.5"
                exit()

            # axes[i].scatter((m1+m2)/Mtot,m2m1,s=marker_size,label="M_tot={:.2e}kg, {} orbits".format(M_tot,len(m2m1)))
            axes[i].scatter((m1+m2)/Mtot,m2m1,s=marker_size)
            axes[i].set_title("f = {}".format(f))

            #set limits
            padding=0.1
            axes[i].set_xlim(0.0-padding,1.0+padding)
            axes[i].set_ylim(0.0-padding,1.0+padding)


    # Add points showing marker size for the legend
    # marker_size_leg=numpy.array([0.05,0.25,1.0])*marker_size_factor
    marker_size_leg=numpy.array([0.05,0.25,0.5])*marker_size_factor
    temp_points=[]
    for s in marker_size_leg:
        # ax1.scatter(0,0,s=s,color=pf.pyplot_colours[0],label="a/R_hill = {}".format(s/marker_size_factor))
        temp_points.append(ax2.scatter(0,0,s=s,c="None",edgecolors="k",label="a/R_hill = {:.2f}".format(s/marker_size_factor)))
    # make legend with these points representing size

    ax2.legend()

    # after legend is made remove the points
    for tp in temp_points:
        tp.remove()


    # pyplot.tight_layout()
    #save the figure
    script_name=os.path.basename(__file__).split('.')[0]
    picname="{}_{}.png".format(script_name,fname_df.split("/")[-1].split(".")[0])
    print "save {}".format(picname)
    pyplot.savefig(picname)

    pyplot.show()
    # pyplot.close()
