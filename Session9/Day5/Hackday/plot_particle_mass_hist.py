'''
Plots a histogram which shows the distribution of:

most massive particle (normalised by cloud mass)
most massive binary primary (normalised by cloud mass)

and on a separate axis:

mass ratio m2/m1 of the binary

Try use the same bins for everything

REMEMBER: this only plots the single most massive binary for each run. We do not account for multiple systems or more than one bound system per run
'''

import pandas as pd
import numpy
import py_func as pf
import matplotlib.pyplot as pyplot
import matplotlib.gridspec as gridspec
import os

# fname_dfs=["/Users/jrobinson/grav_cloud/python_stuff/orbit_search_faster/orbit_search_results/df_orb_all_100_select.txt",
# "/Users/jrobinson/grav_cloud/python_stuff/orbit_search_faster/orbit_search_results/df_orb_all_200_select.txt",
# "/Users/jrobinson/grav_cloud/python_stuff/orbit_search_faster/orbit_search_results/df_orb_all_100_select_stable.txt",
# "df_plot_100_all.txt",
# "df_plot_100_all_stable.txt"]
# fname_dfs=["/Users/jrobinson/grav_cloud/python_stuff/orbit_search_faster/orbit_search_results/df_orb_all_100_select_stable.txt"]
fname_dfs=["df_plot_100_all_stable.txt"]
# fname_dfs=["df_plot_100_all.txt","df_plot_100_all_stable.txt"]

for fname_df in fname_dfs:
    df=pd.read_csv(fname_df,sep="\t",index_col=0) # orbits post selection

    n_bins=20
    plot_option=1

    # # WEIRD MTOT SHENAGANS
    # print "FIX MTOT"
    # Mass=numpy.unique(numpy.array(df['M_tot(kg)']))
    # for M in Mass:
    #     print M
    # print Mass[-2]==Mass[-1]
    # print numpy.allclose(Mass[-2],Mass[-1])
    # print numpy.allclose(Mass,Mass[-1])
    # for i in range(len(df)):
    #     if df.iloc[i]['M_tot(kg)']==Mass[-2]:
    #         df.loc[i,'M_tot(kg)']=Mass[-1]
    #         # print "error"
    # Mass=numpy.unique(numpy.array(df['M_tot(kg)']))
    # for M in Mass:
    #     print M

    print "all systems"
    print df[['run','i','i_orig','m1(kg)','m_largest(kg)']]#.to_string()
    print "all systems with a nan entry"
    print df[numpy.isnan(df['m1(kg)'])][['run','i','i_orig','m1(kg)','m_largest(kg)']]#.to_string()
    print "all runs that never had a binary"
    print df[(numpy.isnan(df['i'])) & (numpy.isnan(df['i_orig']))][['run','i','i_orig','m1(kg)','m_largest(kg)']]#.to_string()
    print numpy.array(df[(numpy.isnan(df['i'])) & (numpy.isnan(df['i_orig']))]['run_name']).tolist()
    # exit()

    # missing_runs=df[numpy.isnan(df['m1(kg)'])]['run']
    # print missing_runs
    nan_runs=[]

    runs=numpy.unique(numpy.array(df['run']).astype(int))
    largest_mass_norm=[]
    bin_mass_norm=[]
    bin_mass_ratio=[]
    for r in runs:

        # if r!=84:
        #     continue

        print r
        df_run=df[df['run']==r].sort_values(by=['m1(kg)','m2(kg)','a(m)'],ascending=[False,False,True])
        mass=numpy.unique(numpy.array(df_run['m_largest(kg)']).astype(float))
        cloud_mass=numpy.unique(numpy.array(df_run['M_tot(kg)']).astype(float))

        # Ensure we only have one run
        if (len(mass)>1) | (len(cloud_mass)>1):
            print df_run[['t(s)','i','j','m_largest(kg)','run','run_name']]
            print mass,cloud_mass
            print len(mass),len(cloud_mass)
            print "error"
            break
            # exit()

        # print df_run[['i','j','m1(kg)','m2(kg)','M_tot(kg)','i_orig']]

        # Get primary and secondary masses, the arrays should already be descending sort
        pri_masses=numpy.array(df_run['m1(kg)']).astype(float)
        sec_masses=numpy.array(df_run['m2(kg)']).astype(float)
        bin_mass=pri_masses[0]
        sec_mass=sec_masses[0]

        print bin_mass,type(bin_mass)
        if numpy.isnan(bin_mass):
            print "nan run"
            nan_runs.append(r)
            # continue

        # If there are multiple orbits, test that we have grabbed the most massive pair
        print "check for most massive pair"
        print mass,pri_masses
        if len(mass)>1:
            print pri_masses
            print pri_masses[~numpy.isnan(pri_masses)]
            if bin_mass!=numpy.amax(pri_masses[~numpy.isnan(pri_masses)]) or sec_mass!=numpy.amax(sec_masses[~numpy.isnan(sec_masses)]):
                print "binary mass error"
                print bin_mass,numpy.amax(pri_masses[~numpy.isnan(pri_masses)])
                print sec_mass,numpy.amax(sec_masses[~numpy.isnan(sec_masses)])
                print pri_masses
                print numpy.array(df_run['i']).astype(int)
                print sec_masses
                exit()
        # print numpy.array(df_run['m2(kg)']).astype(float),sec_mass
        # print numpy.array(df_run['m1(kg)']).astype(float)
        # print bin_mass
        largest_mass=mass[0]
        cloud_mass=cloud_mass[0]
        largest_mass_norm.append(largest_mass/cloud_mass)
        bin_mass_norm.append(bin_mass/cloud_mass)
        bin_mass_ratio.append([sec_mass/bin_mass])
        print cloud_mass,largest_mass
        # Note some of the high mass ratios are just extremely small particles
        if sec_mass/bin_mass>0.9:
            print r
            print "high mass ratio, {}, {} {}".format(sec_mass/bin_mass,bin_mass,sec_mass)
            print (bin_mass+sec_mass)/cloud_mass
    # exit()

    # Drop any nan values
    bin_mass_norm=numpy.array(bin_mass_norm)
    bin_mass_ratio=numpy.array(bin_mass_ratio)
    largest_mass_norm=numpy.array(largest_mass_norm)
    bin_mass_norm=bin_mass_norm[~numpy.isnan(bin_mass_norm)]
    bin_mass_ratio=bin_mass_ratio[~numpy.isnan(bin_mass_ratio)]

    print bin_mass_norm,len(bin_mass_norm)
    print bin_mass_ratio,len(bin_mass_ratio)

    print "number of largest mass particles = {}".format(len(largest_mass_norm))
    print "number of most massive binary primaries = {}".format(len(bin_mass_ratio))
    print "number of runs with no binaries = {}".format(len(nan_runs))
    # log_mass=numpy.log10(largest_mass_norm)

    fig = pyplot.figure()
    gs = gridspec.GridSpec(1,1)
    ax1 = pyplot.subplot(gs[0,0])

    # lin x lin y
    if plot_option==0:
        a1=1
        a2=0.5
        center_large,hist_large,width_large=pf.hist_dist(largest_mass_norm,n_bins)
        center_bin,hist_bin,width_bin=pf.hist_dist(bin_mass_norm,n_bins)
        center_rat,hist_rat,width_rat=pf.hist_dist(bin_mass_ratio,n_bins)
        ax1.bar(center_large, hist_large, align='center',zorder=1,edgecolor='k',
        width=width_large,alpha=a1,label="most massive particles, total={}".format(len(largest_mass_norm)))
        ax1.bar(center_bin, hist_bin, align='center',zorder=2,edgecolor='k',
        width=width_bin,alpha=a1,label="most massive binary primary, total={}".format(len(bin_mass_norm)))
        ax2 = ax1.twiny()
        ax2.bar(center_rat, hist_rat, align='center',zorder=0,edgecolor='g',fill=False,
        width=width_rat,alpha=a2,label="most massive binary mass ratio, total={}".format(len(bin_mass_norm)))
        ax1.set_xlim(0,1)
        ax2.set_xlim(0,1)
        ax1.set_xlabel("mass/total cloud mass")
        ax1.set_ylabel("N")
        ax2.set_xlabel('mass ratio m2/m1', color='g')

    # log x lin y
    # We also set the bins to be the same!
    if plot_option==1:
        a1=0.5
        a2=1.0
        # center_large,hist_large,width_large=pf.hist_dist(numpy.log10(largest_mass_norm),n_bins)
        # center_bin,hist_bin,width_bin=pf.hist_dist(numpy.log10(bin_mass_norm),n_bins)
        # center_rat,hist_rat,width_rat=pf.hist_dist(numpy.log10(bin_mass_ratio),n_bins)

        x_min_array=numpy.array([numpy.amin(numpy.log10(largest_mass_norm)),numpy.amin(numpy.log10(bin_mass_norm)),numpy.amin(numpy.log10(bin_mass_ratio))])
        print "smallest x value = ",x_min_array
        # x_min=-4.5
        x_min=-5.0
        x_max=0.0
        if x_min>numpy.amin(x_min_array):
            print "axis limit error"
            exit()

        hist_large,bins=numpy.histogram(numpy.log10(largest_mass_norm),bins=numpy.linspace(x_min,x_max,n_bins))
        center_large = (bins[:-1] + bins[1:]) / 2
        width_large = numpy.diff(bins)

        hist_bin,bins=numpy.histogram(numpy.log10(bin_mass_norm),bins=numpy.linspace(x_min,x_max,n_bins))
        center_bin = (bins[:-1] + bins[1:]) / 2
        width_bin = numpy.diff(bins)

        hist_rat,bins=numpy.histogram(numpy.log10(bin_mass_ratio),bins=numpy.linspace(x_min,x_max,n_bins))
        center_rat = (bins[:-1] + bins[1:]) / 2
        width_rat = numpy.diff(bins)

        # print bin_mass_norm,len(bin_mass_norm)
        # print numpy.log10(bin_mass_norm),len(numpy.log10(bin_mass_norm)),numpy.amin(numpy.log10(bin_mass_norm)),numpy.amax(numpy.log10(bin_mass_norm))
        print numpy.sum(hist_large),numpy.sum(hist_bin),numpy.sum(hist_rat)
        print "total difference in distributions = {}".format(numpy.sum(numpy.absolute(hist_large-hist_bin))) # find the difference between each bar, shows deviations in dist (requires numpy.absolute!)
        if numpy.sum(hist_bin)!=numpy.sum(hist_rat):
            print "binary counting error"
            exit()

        ax1.bar(center_large, hist_large, align='center',zorder=1,edgecolor='k',
        width=width_large,alpha=a1,label="most massive particles, total={}".format(len(largest_mass_norm)))
        ax1.bar(center_bin, hist_bin, align='center',zorder=2,edgecolor='k',
        width=width_bin,alpha=a1,label="most massive binary primary, total={}".format(len(bin_mass_norm)))
        ax2 = ax1.twiny()
        ax2.bar(center_rat, hist_rat, align='center',zorder=0,edgecolor='r',fill=False,linewidth=2,
        width=width_rat,alpha=a2,label="most massive binary mass ratio, total={}".format(len(bin_mass_norm)))
        x_vals=numpy.append(numpy.append(center_large,center_bin),center_rat)
        x_vals=x_vals[~numpy.isnan(x_vals)]
        x_vals=x_vals[~numpy.isinf(x_vals)]
        xlim1=numpy.amin(x_vals)*1.1
        xlim2=0
        ax1.set_xlim(xlim1,xlim2)
        ax2.set_xlim(xlim1,xlim2)
        ax1.set_xlabel("log(mass/total cloud mass)")
        ax1.set_ylabel("N")
        ax2.set_xlabel('log(mass ratio m2/m1)', color='r')

        print "mass ratio:"
        print hist_rat

        bar_rat=ax1.bar(center_rat, hist_rat, align='center',zorder=0,edgecolor='r',fill=False,linewidth=2,
        width=width_rat,alpha=a2,label="most massive binary mass ratio, total={}".format(len(bin_mass_norm)))
        ax1.legend()
        bar_rat.remove()

        ax2.tick_params('x', colors='r')

        # ax2.legend()

    # log x log y
    if plot_option==2:
        a1=0.5
        a2=0.5
        center_large,hist_large,width_large=pf.hist_dist(numpy.log10(largest_mass_norm),n_bins)
        center_bin,hist_bin,width_bin=pf.hist_dist(numpy.log10(bin_mass_norm),n_bins)
        center_rat,hist_rat,width_rat=pf.hist_dist(numpy.log10(bin_mass_ratio),n_bins)
        ax1.bar(center_large, numpy.log10(hist_large), align='center',zorder=1,edgecolor='k',
        width=width_large,alpha=a1,label="most massive particles, total={}".format(len(largest_mass_norm)))
        ax1.bar(center_bin, numpy.log10(hist_bin), align='center',zorder=2,edgecolor='k',
        width=width_bin,alpha=a1,label="most massive binary primary, total={}".format(len(bin_mass_norm)))
        ax2 = ax1.twiny()
        ax2.bar(center_rat, numpy.log10(hist_rat), align='center',zorder=0,edgecolor='g',fill=False,
        width=width_rat,alpha=a2,label="most massive binary mass ratio, total={}".format(len(bin_mass_norm)))
        x_vals=numpy.append(numpy.append(center_large,center_bin),center_rat)
        x_vals=x_vals[~numpy.isnan(x_vals)]
        x_vals=x_vals[~numpy.isinf(x_vals)]
        xlim1=numpy.amin(x_vals)*1.1
        xlim2=0
        ax1.set_xlim(xlim1,xlim2)
        ax2.set_xlim(xlim1,xlim2)
        ax1.set_xlabel("log(mass/total cloud mass)")
        ax1.set_ylabel("log(N)")
        ax2.set_xlabel('log(mass ratio m2/m1)', color='g')

    # lin x log y
    if plot_option==3:
        a1=0.9
        a2=1
        center_large,hist_large,width_large=pf.hist_dist(largest_mass_norm,n_bins)
        center_bin,hist_bin,width_bin=pf.hist_dist(bin_mass_norm,n_bins)
        center_rat,hist_rat,width_rat=pf.hist_dist(bin_mass_ratio,n_bins)
        ax1.bar(center_large, numpy.log10(hist_large), align='center',zorder=1,edgecolor='k',
        width=width_large,alpha=a1,label="most massive particles, total={}".format(len(largest_mass_norm)))
        ax1.bar(center_bin, numpy.log10(hist_bin), align='center',zorder=2,edgecolor='k',
        width=width_bin,alpha=a1,label="most massive binary primary, total={}".format(len(bin_mass_norm)))
        ax2 = ax1.twiny()
        ax2.bar(center_rat, numpy.log10(hist_rat), align='center',zorder=0,edgecolor='g',fill=False,
        width=width_rat,alpha=a2,label="most massive binary mass ratio, total={}".format(len(bin_mass_norm)))
        ax1.set_xlim(0,1)
        ax2.set_xlim(0,1)
        ax1.set_xlabel("mass/total cloud mass")
        ax1.set_ylabel("log(N)")
        ax2.set_xlabel('mass ratio m2/m1', color='g')


    # ax2.tick_params('x', colors='g')

    #save the figure
    script_name=os.path.basename(__file__).split('.')[0]
    picname="{}_{}_{}.png".format(script_name,fname_df.split("/")[-1].split(".")[0],plot_option)
    print "save {}".format(picname)
    pyplot.savefig(picname)

    pyplot.show()
    # pyplot.close()
