'''
Based on Nesvorny et al 2010 fig.3.
NB, compare similarities and differences?
'''

import pandas as pd
import numpy
import py_func as pf
import matplotlib.pyplot as pyplot
import matplotlib.gridspec as gridspec
import os
import matplotlib.ticker as ticker

# df=pd.read_csv("df_cloud_runs_data_all.txt",sep="\t",index_col=0) # orbits post selection
fname_df="df_plot_100_all_stable.txt"

df=pd.read_csv(fname_df,sep="\t",index_col=0)

plot_option=0

# load real binaries
df_tot=pd.read_csv("/Users/jrobinson/grav_cloud/python_stuff/acquire_binaries/df_tnb_tot_14_05_2019.txt",sep="\t",index_col=0)
print df_tot
print list(df_tot)

# Find anything that is a special case on Grundy's webpage, or a dwarf planet at: https://en.wikipedia.org/wiki/Dwarf_planet
DP_names=["ceres","pluto","haumea","makemake","eris","orcus","salacia","quaoar","sedna"]
DP_DES=["2002 ms4","2007 or10"]
df_weird=pd.DataFrame()
df_norm=pd.DataFrame()
for i in range(len(df_tot)):
    name_check=0
    name = df_tot.iloc[i]['Object']
    # print name
    DES = df_tot.iloc[i]['DES']
    print i,name,DES

    if str(df_tot.iloc[i]['Orbit Status']).lower()=="special case":
        print "weird"
        df_weird=df_weird.append(df_tot.iloc[i])
        continue
    elif (DES.lower() in DP_DES):
        print "weird"
        df_weird=df_weird.append(df_tot.iloc[i])
        continue
    else:
        for n in (name.lower().split(" ")):
            if n in DP_names:
                print "weird"
                df_weird=df_weird.append(df_tot.iloc[i])
                name_check+=1
                break

    if name_check==0:
        print "append"
        df_norm=df_norm.append(df_tot.iloc[i])

print len(df_tot)
print len(df_norm)
print len(df_weird)

# df_tot=df_norm
# exit()

V1=numpy.array(df_tot['V1']).astype(float)
V2=numpy.array(df_tot['V2']).astype(float)
V1_norm=numpy.array(df_norm['V1']).astype(float)
V2_norm=numpy.array(df_norm['V2']).astype(float)
V1_weird=numpy.array(df_weird['V1']).astype(float)
V2_weird=numpy.array(df_weird['V2']).astype(float)

print V1,V2

fig = pyplot.figure()
gs = gridspec.GridSpec(1,1)
ax1 = pyplot.subplot(gs[0,0])

# fig.set_size_inches(15.5, 10.5)

print len(df)
print len(df[~numpy.isnan(df['m1(kg)'])])
df=df[~numpy.isnan(df['m1(kg)'])] # drop entries without binaries
print list(df)

# only plot the simple binary systems
df=df[df['N_sys']==2]

# # Only include f values 3,10,30
# df=df[(df['f']<100.0) & (df['f']>1.0)]

if plot_option==0:
    # plot size ratio against primary size (similar to Nesvorny et al 2010)

    d=44.0 #object distance (AU)
    p1=0.08 # albedo, changing this doesn't seem to affect the plot much
    p2=p1
    C=664.5e3 # constant for V band
    d0=1.0

    # twin axis for extra axis scale
    ax2 = ax1.twiny()

    for M_tot in numpy.unique(numpy.array(df['M_tot(kg)']).astype(float)):
        df2=df[df['M_tot(kg)']==M_tot]
        rho=numpy.array(df2['rho(kgm-3)']).astype(float)
        R1=((3.0*numpy.array(df2['m1(kg)']).astype(float))/(4.0*numpy.pi*rho))**(1.0/3.0)
        R2=((3.0*numpy.array(df2['m2(kg)']).astype(float))/(4.0*numpy.pi*rho))**(1.0/3.0)
        R2R1=R2/R1

        delta_mag=5.0*numpy.log10(numpy.sqrt(p1/p2)*(R1/R2))
        mag1=5.0*numpy.log10(C/(numpy.sqrt(p1)*R1)*((d*(d-d0))/(d0**2.0)))

        ax1.scatter(delta_mag,mag1,label="M_tot={:.2e}kg, {}".format(M_tot,len(mag1)))

        # we set up ax2 to have the exact same y axis as ax1
        ax2.scatter(delta_mag,mag1,marker='o',color='None')

    ax1.scatter(V2_norm-V1_norm,V1_norm,marker='x',color='k',label='real binaries')
    # ax1.scatter(V2_weird-V1_weird,V1_weird,marker='x',color='r',label='\'weird\' binaries')
    ax1.scatter(V2_weird-V1_weird,V1_weird,marker='o',color='None',edgecolors="r",label='Dwarf Planets or Special case binaries')

    # we set up ax2 to have the exact same y axis as ax1
    ax2.scatter(V2_norm-V1_norm,V1_norm,marker='o',color='None')
    ax2.scatter(V2_weird-V1_weird,V1_weird,marker='o',color='None')

    # for i in range(len(df_tot)):
    #     ax1.text(V2[i]-V1[i],V1[i],numpy.array(df_tot.iloc[i]['DES']))
    #     ax2.text(V2[i]-V1[i],V1[i],"")

    # variables to shift text position
    shift_x=0.05
    shift_y=0.05

    for i in range(len(df_weird)):
        name=numpy.array(df_weird.iloc[i]['DES'])
        if name in ['2002 VD131','2013 SQ99','2014 UD225']:
            if name=='2013 SQ99':
                ax1.text((V2_weird[i]-V1_weird[i])-shift_x,V1_weird[i],name,color="r",horizontalalignment='right',verticalalignment='center')
            elif name=='2002 VD131':
                ax1.text((V2_weird[i]-V1_weird[i])-shift_y,V1_weird[i],name,color="r",horizontalalignment='center',verticalalignment='bottom')
            elif name=='2014 UD225':
                ax1.text((V2_weird[i]-V1_weird[i])+shift_y,V1_weird[i],name,color="r",horizontalalignment='center',verticalalignment='top')
            else:
                continue
        else:
            ax1.text((V2_weird[i]-V1_weird[i])+shift_x,V1_weird[i],name,color="r")
            # ax2.text(V2_weird[i]-V1_weird[i],V1_weird[i],"")

    # # TRY DERIVE RESOLUTION LIMIT!
    # # calculate the limit
    # alpha=75.0e-3*(numpy.pi/648000) #rad
    # _d=(d-d0)*pf.AU #m
    # mag_lim=numpy.linspace(19.0,25.0)
    # R1=C*((d*(d-d0))/(d0**2.0))*(10**(-0.2*mag_lim))/numpy.sqrt(p1)
    # # R1=((3.0*numpy.array(df['m1(kg)']).astype(float))/(4.0*numpy.pi*numpy.array(df['rho(kgm-3)']).astype(float)))**(1.0/3.0)
    # # mag1=5.0*numpy.log10(C/(numpy.sqrt(p1)*R1)*((d*(d-d0))/(d0**2.0)))
    # print R1
    # R_lim=alpha*_d
    # print R_lim
    # delta_mag_lim=-5.0*numpy.log10((R_lim/R1)-1.0)+9.1
    # print delta_mag_lim
    # ax1.scatter(delta_mag_lim,mag_lim,marker='.',color='k',label='resolution limit')

    #plot some loose bounds for now
    x_dat=[1.0,3.0]
    y_dat=[26.0,19.0]
    m=(y_dat[1]-y_dat[0])/(x_dat[1]-x_dat[0])
    c=y_dat[1]-(m*x_dat[1])
    y_lim=25
    x_lim1=0
    x_lim2=(y_lim-c)/m
    ax1.plot([x_lim1,x_lim2],[y_lim,y_lim],color="r")
    x_plot=numpy.array([x_lim2,(19.0-c)/m])
    y_plot=(m*x_plot)+c
    ax1.plot(x_plot,y_plot,color="r")

    ax1.invert_yaxis()
    # Set axis labels
    ax1.set_ylabel('primary mag')
    ax1.set_xlabel("delta mag")

    # add MU69 to plot, check if values match observables
    R1_MU69=19.46e3/2.0
    R2_MU69=14.24e3/2.0
    d_MU69=44.53998
    p1_MU69=0.14
    p2_MU69=p1_MU69
    delta_mag_MU69=5.0*numpy.log10(numpy.sqrt(p1_MU69/p2_MU69)*(R1_MU69/R2_MU69))
    mag1_MU69=5.0*numpy.log10(C/(numpy.sqrt(p1_MU69)*R1_MU69)*((d_MU69*(d_MU69-d0))/(d0**2.0)))
    print mag1_MU69,delta_mag_MU69
    R_tot_MU69=((R1_MU69**3.0)+(R2_MU69**3.0))**(1.0/3.0)
    print R_tot_MU69
    mag1_MU69_tot=5.0*numpy.log10(C/(numpy.sqrt(p1_MU69)*R_tot_MU69)*((d_MU69*(d_MU69-d0))/(d0**2.0)))
    print mag1_MU69_tot
    ax1.scatter(delta_mag_MU69,mag1_MU69,marker='+',s=50,color='r',label='MU69')
    ax1.text(delta_mag_MU69+shift_x,mag1_MU69,"2014 MU69")

    # plot our orbit detection limit
    ax1.axvline(5.0,color="k",alpha=0.2,zorder=0)

    # add mass ratio to axis

    # we get the ax2 y ticks, remove the m2/m1=0 and add m2/m1=1e-3
    # note that the ylims must be explicitly preserved
    lim = ax2.get_xlim()
    ax2_xticks=list(ax2.get_xticks())
    print ax2_xticks
    # ax2_xticks.remove(0)
    # print ax2_xticks
    # ax2.set_xticks(ax2_xticks + [1e-3])
    ax2.set_xlim(lim)

    print ax2.get_xticks()
    # print 5.0*numpy.log10(numpy.sqrt(p1/p2)*(1.0/numpy.array(ax2.get_yticks())))

    # use this function to convert an m2/m1 value into a delta mag value
    # FuncFormatter can be used as a decorator
    @ticker.FuncFormatter
    def major_formatter(x, pos):
        # return "[%.2f]" % x
        # if x==0:
        #     return ""
        # else:
        return "{:.1e}".format((numpy.sqrt(p1/p2)*(10.0**(-x/5.0)))**3.0)

    ax2.xaxis.set_major_formatter(major_formatter)
    ax2.set_xlabel("m2/m1")

    ax1.legend()

#save the figure
script_name=os.path.basename(__file__).split('.')[0]
picname="{}_{}_{}.png".format(script_name,fname_df.split("/")[-1].split(".")[0],plot_option)
print "save {}".format(picname)
pyplot.savefig(picname)

pyplot.show()
# pyplot.close()
