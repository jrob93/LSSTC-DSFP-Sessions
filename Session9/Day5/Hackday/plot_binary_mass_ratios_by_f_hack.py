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

# use LaTeX fonts in the plot
pyplot.rc('text', usetex=True)
pyplot.rc('font', family='serif')

path="/Users/jrobinson/xq1_grav_cloud/binary_stability/orbit_results/orbit_results_plots"
fname_dfs=["df_plot_100_all_stable.txt"]

marker_size_factor=100.0
markers=['^','s','o']

for fname_df in fname_dfs:

    df=pd.read_csv("{}/{}".format(path,fname_df),sep="\t",index_col=0) # orbits post selection

    df=df[df['N_sys']==2]

    fig = pyplot.figure()

    # size=3.30709 # 84mm in inches
    s_x=1.0
    s_y=0.8

    pc_tex=0.16605 # latex pc in inches
    text_width=39.0*pc_tex
    column_sep=2.0*pc_tex
    # column_width=3.30709 # 84mm in inches
    x_len=((text_width/2.0)+(1.5*pc_tex))*s_x
    y_len=(x_len)*s_y
    print "size: {}x{} inches".format(x_len,y_len)
    fig.set_size_inches(x_len,y_len)

    gs = gridspec.GridSpec(2,2,wspace=0.0,hspace=0.0)
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

    fig.text(0.5, 0.01, '$m_2/m_1$', ha='center')
    fig.text(0.0, 0.5, '$(m_2+m_1)/M_\\mathrm{{c}}$', va='center', rotation='vertical')

    f_vals = numpy.unique(numpy.array(df['f']).astype(float))

    print len(df)
    print len(df[~numpy.isnan(df['m1(kg)'])])
    df=df[~numpy.isnan(df['m1(kg)'])] # drop entries without binaries
    print list(df)

    df['m2/m1']=numpy.array(df['m2(kg)']).astype(float)/numpy.array(df['m1(kg)']).astype(float) # move this to a separate script

    # plot mass ratio against normalised binary mass
    y_max=0.0

    # for i,f in enumerate(f_vals):
    #     for j,M_tot in enumerate(numpy.unique(numpy.array(df['M_tot(kg)']).astype(float))):
    #         df2=df[(df['M_tot(kg)']==M_tot) & (df['f']==f)]
    #         m1=numpy.array(df2['m1(kg)']).astype(float)
    #         m2=numpy.array(df2['m2(kg)']).astype(float)
    #         m2m1=numpy.array(df2['m2/m1']).astype(float)
    #         Mtot=numpy.array(df2['M_tot(kg)']).astype(float)
    #         print f,M_tot,len(df2)
    #
    #         # Use marker size to represent separation
    #         a_R_hill=numpy.array(df2['a(m)']).astype(float)/numpy.array(df2['R_hill(m)']).astype(float)
    #         marker_size=a_R_hill*marker_size_factor
    #         if len(a_R_hill[a_R_hill>=0.5])>0:
    #             print "an orbit has a/R_hill>0.5"
    #             exit()
    #
    #         # axes[i].scatter((m1+m2)/Mtot,m2m1,s=marker_size,label="M_tot={:.2e}kg, {} orbits".format(M_tot,len(m2m1)))
    #         y_data=(m1+m2)/Mtot
    #         x_data=m2m1
    #         axes[i].scatter(x_data,y_data,
    #         s=marker_size,
    #         edgecolors=pf.pyplot_colours[j],facecolors='none',
    #         marker=markers[j],
    #         alpha=1.0)
    #         # axes[i].set_title("f = {}".format(f))
    #
    #         if numpy.amax(x_data)>y_max:
    #             y_max=numpy.amax(y_data)
    #
    #         print numpy.amax(m2m1),numpy.amax((m1+m2)/Mtot)

    for j,f in enumerate(f_vals):
        df2=df[df['f']==f]

        int_list=numpy.array(df2['M_tot(kg)'])
        for i,M_tot in enumerate(numpy.unique(numpy.array(df2['M_tot(kg)']).astype(float))):
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

        m1=numpy.array(df2['m1(kg)']).astype(float)
        m2=numpy.array(df2['m2(kg)']).astype(float)
        m2m1=numpy.array(df2['m2/m1']).astype(float)
        Mtot=numpy.array(df2['M_tot(kg)']).astype(float)

        x_data=m2m1
        y_data=(m1+m2)/Mtot

        # Use marker size to represent separation
        a_R_hill=numpy.array(df2['a(m)']).astype(float)/numpy.array(df2['R_hill(m)']).astype(float)
        marker_size=a_R_hill*marker_size_factor
        if len(a_R_hill[a_R_hill>=0.5])>0:
            print "an orbit has a/R_hill>0.5"
            exit()
        print marker_size

        if numpy.amax(x_data)>y_max:
            y_max=numpy.amax(y_data)

        for i in range(len(x_data)):
            label="$M_\\mathrm{{c}}={:.2e}~\\mathrm{{kg}}$".format(Mtot[i])
            axes[j].scatter(x_data[i],y_data[i],
            edgecolors=col_list[i],facecolors='none',
            marker=marker_list[i],
            s=marker_size[i],
            alpha=1.0)

    print y_max

    # set labels and limits
    plot_label=['a','b','c','d']
    for i,f in enumerate(f_vals):

        padding=0.1/2.0

        # label plots
        f_label="{}. $f={}$".format(plot_label[i],f)
        # f_label="f={}".format(f)
        # f_label="{}.".format(plot_label[i])
        if i%2!=0:
            axes[i].text(1.0,y_max-padding,f_label,fontdict={'size': 6},ha='right')
        else:
            axes[i].text(0.0,y_max-padding,f_label,fontdict={'size': 6},ha='left')

        #format
        axes[i].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        axes[i].xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

        # Remove a single label
        if i==3:
            import matplotlib.ticker as mticker
            def update_ticks(x, pos):
                if x == 0:
                    return ''
                else:
                    return x
            ax4.xaxis.set_major_formatter(mticker.FuncFormatter(update_ticks))

        # set axis limits
        axes[i].set_xlim(0.0-padding,1.0+padding)
        axes[i].set_ylim(0.0-padding,y_max+padding)


    # Add points showing marker size for the legend
    # marker_size_leg=numpy.array([0.05,0.25,1.0])*marker_size_factor
    marker_size_leg=numpy.array([0.05,0.25,0.5])*marker_size_factor
    temp_points=[]
    for s in marker_size_leg:
        # ax1.scatter(0,0,s=s,color=pf.pyplot_colours[0],label="a/R_hill = {}".format(s/marker_size_factor))
        temp_points.append(ax1.scatter(0,0,s=s,c="None",edgecolors="k",label="$\\frac{{a_{{\\mathrm{{bin}}}}}}{{R_{{\\mathrm{{Hill}}}}}}={:.2f}$".format(s/marker_size_factor)))
    # make legend with these points representing size

    # Remove duplicates from legend
    handles, labels = ax1.get_legend_handles_labels()
    from collections import OrderedDict
    by_label = OrderedDict(zip(labels, handles))

    ax1.legend(by_label.values(), by_label.keys(),loc='upper right',prop={'size': 6})

    # after legend is made remove the points
    for tp in temp_points:
        tp.remove()


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
