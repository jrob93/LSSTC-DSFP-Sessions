'''
Plot the position data of a simulation snapshot.
Note that we sort the particles by mass before plotting, which results in the the highest mass particles appearing on top

Use dbscan to find clusters.
Test clusters for binding and then find their centre of mass.
Search for binding between clusters.
'''
from matplotlib import rc
rc('font',**{'size':9})

import numpy
import matplotlib.pyplot as pyplot
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import rebound
import py_func as pf
import networkx as nx
import pandas as pd
import os

# pyplot.rc('text', usetex=True)

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import metrics
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import DBSCAN

cluster_search=0
dbscan_eps=0.1#0.075
find_orbits=0
dat_file="/Users/jrobinson/grav_cloud/python_stuff/orbit_search_all/121_probe/dat0000202_0.txt"
orb_file="/Users/jrobinson/grav_cloud/python_stuff/orbit_search_all/121_probe/121_cluster_orbits.txt"
N_particle_min=20 # maximum number of particles in a cluster of interest
N_particle_max=2000 # maximum number of particles in a cluster that we will check for binding (algorthim too slow for lareg numbers)

# load the particle data and run parameters
t,df_dat=pf.load_dat_file(dat_file)
df_dat=df_dat.sort_values('m(kg)')
df_rp=pf.load_run_params("/Users/jrobinson/grav_cloud/python_stuff/orbit_search_all/121_probe/run_params_0.txt")
Ntot=float(df_rp['N_tot'].iloc[0])
rho=float(df_rp['rho(kgm-3)'].iloc[0])
Req=float(df_rp['R_eq(m)'].iloc[0])
M_tot=float(df_rp['M_tot(kg)'].iloc[0])
R_c=float(df_rp['R_c(m)'].iloc[0])
f=float(df_rp['f'].iloc[0])
lim=10*R_c
mp=M_tot/Ntot #kg
r_min=Req*(Ntot**(-1.0/3.0))*f
r_max=Req*f
print df_rp
print "time = {} s, {} yr, {}".format(t,t/(pf.year_s),t/(1e2*pf.year_s))

# exit()

fig = pyplot.figure() #open figure once

pc_tex=0.16605 # latex pc in inches
text_width=39.0*pc_tex
column_sep=2.0*pc_tex
column_width=(text_width-column_sep)/2.0
s_x=1.0
s_y=1.0
x_len=((text_width/2.0)+(1.5*pc_tex))*s_x
y_len=(x_len)*s_y
print "size: {}x{} inches".format(x_len,y_len)
fig.set_size_inches(x_len,y_len)

gs = gridspec.GridSpec(1, 1,wspace=0.0,hspace=0.0)
ax1 = pyplot.subplot(gs[0,0])
ax1.set_aspect("equal")
ax1.set_xlim(-lim,lim)
ax1.set_ylim(-lim,lim)
ax1.set_xlabel("$x~(\\mathrm{{m}})$")
ax1.set_ylabel("$y~(\\mathrm{{m}})$")

# Use mass and particle sizes to set up the color and marker sizes for plotting
m=numpy.array(df_dat['m(kg)'])
r=numpy.array(df_dat['r(m)'])
color=(numpy.log10(df_dat['m(kg)'])-numpy.log10(mp))/(numpy.log10(M_tot)-numpy.log10(mp))
s_min=1e-1
s_max1=2e1
s_max2=2e2#e0
print "particle sizes: ",r_min,r_max
m_s = (s_max1-s_min)/(r_max-r_min)
m_s2=(s_max2-s_min)/(r_max**2.0-r_min**2.0)
c2=s_min-(m_s2*(r_min**2.0))
size1 = (m_s*r)+(s_min-(m_s*r_min))
size2=(m_s2*(numpy.power(r,2.0)))+c2
print "marker sizes s1={}, s2={}".format(numpy.amax(size1),numpy.amax(size2))

# Plot all the particles
ax1.scatter(df_dat['x(m)'],df_dat['y(m)'],c=color,vmin=0,vmax=1,s=size1, rasterized=True)
# ax1.scatter(df_dat['x(m)'],df_dat['z(m)'],c=color,vmin=0,vmax=1,s=size1)

#-------------------------------------------------------------------------------
# search all the particles for clustering in xyz space
if cluster_search==1:

    # # kmeans search
    # X=df_cut[['x(m)','y(m)','z(m)']]
    # # X=df_dat[['x(m)','y(m)','z(m)']]
    # y_pred = KMeans(n_clusters=6, random_state=0).fit_predict(X)
    # # y_pred=runKMeans(numpy.array(df_cut[['x(m)','y(m)']]))
    # print y_pred
    # ax1.scatter(df_cut['x(m)'],df_cut['y(m)'],c=y_pred,s=1)
    # # ax1.scatter(df_dat['x(m)'],df_dat['y(m)'],c=y_pred,s=1)

    # DBscan search, SEARCH RESULTS DEPEND ENTIRELY ON dbscan_eps VARIABLE
    df_fit=df_dat
    X=df_fit[['x(m)','y(m)','z(m)']] #select only position labels (would mass help?)
    X = StandardScaler().fit_transform(X) #rescale the positions: Standardize features by removing the mean and scaling to unit variance

    db = DBSCAN(eps=dbscan_eps, min_samples=10).fit(X) # do DBSCAN, eps: The maximum distance between two samples for them to be considered as in the same neighborhood.
    core_samples_mask = numpy.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    unique_labels = set(labels)
    print "labels = ",unique_labels

    append_count=0
    for k in unique_labels:

        class_member_mask = (labels == k)

        # cluster members
        # xy = X[class_member_mask & core_samples_mask]
        df_cluster=df_fit[class_member_mask & core_samples_mask]
        df_cluster['cluster']=labels[class_member_mask & core_samples_mask] # add a cluster identifier number
        print labels[class_member_mask & core_samples_mask]

        # k=-1 for outliers
        # xy = X[class_member_mask & ~core_samples_mask]

        # append the df
        if append_count==0:
            df_dat_cluster=df_cluster
            append_count+=1
        else:
            print "append"
            df_dat_cluster=df_dat_cluster.append(df_cluster)

    print len(df_dat)
    print len(df_dat_cluster)

    print df_dat_cluster.to_string()
    df_dat_cluster.to_csv("df_dat_cluster.txt",na_rep='NaN',sep="\t") # save the dataframe so we can skip the cluster fit

# load the dataframe of particle clusters
df_dat_cluster=pd.read_csv("/Users/jrobinson/grav_cloud/python_stuff/orbit_search_all/121_probe/df_dat_cluster.txt",sep="\t",index_col=0)

clusters=numpy.unique(df_dat_cluster['cluster'])

# Assess binding
append_count=0
print "starting particles = {}".format(len(df_dat_cluster))
for k in clusters:
    unbound_particles=-1
    df_cluster=df_dat_cluster[df_dat_cluster['cluster']==k]
    while unbound_particles!=0:

        # only look at clusters with a sufficient number of particles
        if len(df_cluster)<N_particle_min:
            unbound_particles=0
            continue
        print k, len(df_cluster)

        #plot the cluster particles
        # ax1.scatter(df_cluster['x(m)'],df_cluster['y(m)'],s=1)
        # ax1.scatter(df_cluster['x(m)'],df_cluster['z(m)'],s=1)

        # get the centre of mass particle
        pos_com=pf.centre_of_mass(numpy.array(df_cluster[['x(m)','y(m)','z(m)']]),numpy.array(df_cluster['m(kg)']))
        vel_com=pf.centre_of_mass(numpy.array(df_cluster[['vx(ms^-1)','vy(ms^-1)','vz(ms^-1)']]),numpy.array(df_cluster['m(kg)']))
        total_mass=numpy.sum(numpy.array(df_cluster['m(kg)']))
        # cluster_pos.append(pos_com)
        # cluster_vel.append(vel_com)
        # cluster_m.append(total_mass)

        #check binding
        if len(df_cluster)>N_particle_max:# too many particles to check
            unbound_particles=0
            # ax1.scatter(df_cluster['x(m)'],df_cluster['y(m)'],marker ='+')

        else:
            print k,len(df_cluster)
            pos=numpy.array(df_cluster[['x(m)','y(m)','z(m)']])
            vel=numpy.array(df_cluster[['vx(ms^-1)','vy(ms^-1)','vz(ms^-1)']])
            pos,vel=pf.rotating_to_heliocentric_array(pos,vel,float(df_rp.iloc[0]['a_orb(m)']),t)
            # transform to relative coords
            pos_com,vel_com=pf.rotating_to_heliocentric_array(pos_com,vel_com,float(df_rp.iloc[0]['a_orb(m)']),t)
            pos=pos-pos_com
            vel=vel-vel_com
            m=numpy.array(df_cluster['m(kg)'])
            # calculate energies
            GPE=-pf.grav_pot_energy_direct2(pos,m)
            KE=pf.kinetic_energy(vel,m)
            E=KE+GPE
            print "bound particles = {}".format(len(E[E<0]))
            unbound_particles=len(df_cluster)-len(E[E<0])
            print "unbound particles = {}".format(unbound_particles)
            df_cluster=df_cluster[E<0]

        if unbound_particles==0:
            #plot the cluster particles
            # ax1.scatter(df_cluster['x(m)'],df_cluster['y(m)'],marker ='+')
            # append the df
            if append_count==0:
                df_dat_cluster_bound=df_cluster
                append_count+=1
            else:
                print "append"
                df_dat_cluster_bound=df_dat_cluster_bound.append(df_cluster)

    # break
print "remaining particles = {}".format(len(df_dat_cluster_bound))
df_dat_cluster=df_dat_cluster_bound

#-------------------------------------------------------------------------------
# plot the clusters, and the cluster com
cluster_pos=[]
cluster_vel=[]
cluster_m=[]
cluster_N=[]
for k in clusters:
    df_cluster=df_dat_cluster[df_dat_cluster['cluster']==k]
    # only look at clusters with a sufficient number of particles
    if len(df_cluster)<N_particle_min:
        continue
    print k, len(df_cluster)
    # get the centre of mass particle
    pos_com=pf.centre_of_mass(numpy.array(df_cluster[['x(m)','y(m)','z(m)']]),numpy.array(df_cluster['m(kg)']))
    vel_com=pf.centre_of_mass(numpy.array(df_cluster[['vx(ms^-1)','vy(ms^-1)','vz(ms^-1)']]),numpy.array(df_cluster['m(kg)']))
    total_mass=numpy.sum(numpy.array(df_cluster['m(kg)']))
    cluster_pos.append(pos_com)
    cluster_vel.append(vel_com)
    cluster_m.append(total_mass)
    cluster_N.append(len(df_cluster))

    #plot the cluster particles
    # ax1.scatter(df_cluster['x(m)'],df_cluster['y(m)'],s=1)

    # plot the Hill radius of each cluster
    R_hill=30.0*pf.AU*((total_mass/(3.0*pf.M_sun))**(1.0/3.0))
    print "R_hill={}m".format(R_hill)
    circle1 = pyplot.Circle((pos_com[0], pos_com[1]), R_hill, color='r',fill=False)
    ax1.add_artist(circle1)

# pyplot.show()
# exit()

# create a dataframe of centre of mass particles
cluster_pos=numpy.array(cluster_pos)
cluster_vel=numpy.array(cluster_vel)
cluster_m=numpy.array(cluster_m)
df_cluster_com=pd.DataFrame(columns=['x(m)','y(m)','z(m)','vx(ms^-1)','vy(ms^-1)','vz(ms^-1)','m(kg)'])
df_cluster_com['m(kg)']=cluster_m
df_cluster_com[['x(m)','y(m)','z(m)']]=cluster_pos
df_cluster_com[['vx(ms^-1)','vy(ms^-1)','vz(ms^-1)']]=cluster_vel
df_cluster_com['r(m)']=(3.0*df_cluster_com['m(kg)']/(4.0*numpy.pi*rho))**(1.0/3.0)
df_cluster_com['N']=cluster_N

print df_cluster_com.to_string()

# # Plot the centre of mass particles
# ax1.scatter(df_cluster_com['x(m)'],df_cluster_com['y(m)'],c='r',marker='x')
# # ax1.scatter(df_cluster_com['x(m)'],df_cluster_com['z(m)'],c='r',marker='x')
#
# for i in range(len(df_cluster_com)):
#     ax1.text(float(df_cluster_com.iloc[i]['x(m)']),float(df_cluster_com.iloc[i]['y(m)']),"{:.2e}".format(df_cluster_com.iloc[i]['m(kg)']))

# pyplot.show()
# exit()

#-------------------------------------------------------------------------------
# Search for orbits amongst the centre of mass particles

# fnum=202
fnum=dat_file.split('/')[-1].split('_')[0][3:]

if find_orbits==1:
    #create rebound sim once at start
    sim = rebound.Simulation()
    sim.G=pf.G

    # Find any orbits, no cuts!
    df_dat=df_cluster_com
    orbf = open(orb_file, 'w')

    df_dat=df_dat.sort_values(by=['m(kg)'],ascending=False)
    print df_dat
    N=len(df_dat)
    df_dat['i']=range(len(df_dat)) # add an index to rank by mass, 0 is most massive

    #Do coordinate transform from the simulation rotating frame, to the heliocentric frame
    r=numpy.array(df_dat.loc[:,['x(m)','y(m)','z(m)']])
    v=numpy.array(df_dat.loc[:,['vx(ms^-1)','vy(ms^-1)','vz(ms^-1)']])
    R,V=pf.rotating_to_heliocentric_array(r,v,float(df_rp.iloc[0]['a_orb(m)']),t)
    # # OPTIONAL: transform to relative frame
    # R=R-R[0,:]
    # V=V-V[0,:]
    df_dat.loc[:,['x(m)','y(m)','z(m)']]=R
    df_dat.loc[:,['vx(ms^-1)','vy(ms^-1)','vz(ms^-1)']]=V

    # Search for orbits
    for i in df_dat['i'][:-1]: # the index means we don't search the last particle, it will have already been searched (symmetry!)
        print "search particle {} out of {}".format(i,N-1)
        pi=df_dat.iloc[i] #primary properties
        sim.add(x=pi['x(m)'],y=pi['y(m)'],z=pi['z(m)'],
        vx=pi['vx(ms^-1)'],vy=pi['vy(ms^-1)'],vz=pi['vz(ms^-1)'],
        m=pi['m(kg)'])

        for j in df_dat['i']:
            if i==j or j<i: # Do not search for orbit with self, or with a particle of higher mass (it's already been checked)
                continue
            # print "{}: search particles {}, {}".format(f_path,i,j)
            pj=df_dat.iloc[j] # secondary properties

            # We find the orbit of particle j RELATIVE to particle i
            sim.add(x=pj['x(m)'],y=pj['y(m)'],z=pj['z(m)'],
            vx=pj['vx(ms^-1)'],vy=pj['vy(ms^-1)'],vz=pj['vz(ms^-1)'],
            m=pj['m(kg)'])
            orbit = sim.particles[1].calculate_orbit(sim.particles[0])
            # Only save closed orbits
            if orbit.e>=0 and orbit.e<1.0:
                # Save the orbit as: ['t(s)','file','i','j','a(m)','e','I(rad)','omega(rad)','OMEGA(rad)','f_true(rad)','m1(kg)','m2(kg)'] where i,j is now the actual particle index in the file
                print("WRITE: {}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(t,fnum,int(pi.name),int(pj.name),orbit.a,orbit.e,orbit.inc,orbit.omega,orbit.Omega,orbit.f,pi['m(kg)'],pj['m(kg)']))
                orbf.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(t,fnum,int(pi.name),int(pj.name),orbit.a,orbit.e,orbit.inc,orbit.omega,orbit.Omega,orbit.f,pi['m(kg)'],pj['m(kg)']))

            sim.remove(sim.particles[1])

        # print "remove last particle"
        sim.remove(sim.particles[0])

    orbf.close()

# #-------------------------------------------------------------------------------
# # Load and plot the orbits
# df_orb=pf.load_orb_file(orb_file)
# a_hel=[]
# for i in range(len(df_orb)):
#     # Get the orbit
#     orb=df_orb.iloc[i]
#     _orb=numpy.array(orb[['a(m)','e','I(rad)','OMEGA(rad)','omega(rad)']])
#     pos_orb=pf.planet_orbit(_orb,100)
#
#     # get the rotating frame coordinates of the primary and secondary
#     primary_int=int(orb['i'])
#     secondary_int=int(orb['j'])
#     print df_cluster_com.loc[primary_int]
#     _pos=numpy.array(df_cluster_com.loc[primary_int][['x(m)','y(m)','z(m)']]) # note iloc accesses the nth row, whereas loc access the row with index n
#     _pos2=numpy.array(df_cluster_com.loc[secondary_int][['x(m)','y(m)','z(m)']]) # note iloc accesses the nth row, whereas loc access the row with index n
#
#     #transform the relative orbit coordinates to heliocentric coords
#     pos_frame,V=pf.rotating_to_heliocentric_array(_pos,numpy.zeros(3),float(df_rp.iloc[0]['a_orb(m)']),t)
#     a_hel.append(numpy.linalg.norm(pos_frame)) # find heliocentric distance for the hill radius
#     pos_orb=pos_orb+pos_frame # orbit position = relative orbit position + heliocentric primary position
#     #transform coords back to rotating frame
#     pos_orb,V=pf.heliocentric_to_rotating_array(pos_orb,numpy.zeros(3),float(df_rp.iloc[0]['a_orb(m)']),t)
#
#     ax1.plot(pos_orb[:,0],pos_orb[:,1]) # plot in m scale
#     ax1.scatter(_pos[0],_pos[1],c='k',marker='+',zorder=3) # plot in m scale
#     ax1.scatter(_pos2[0],_pos2[1],c='k',marker='+',zorder=3) # plot in m scale
#     # ax1.plot(pos_orb[:,0],pos_orb[:,2]) # plot in m scale
#     # ax1.scatter(_pos[0],_pos[2],c='k',marker='+') # plot in m scale
#     # ax1.scatter(_pos2[0],_pos2[2],c='k',marker='x') # plot in m scale
#
# print a_hel
# df_orb['a_hel(m)']=numpy.array(a_hel)
# df_orb['R_hill(m)']=numpy.array(df_orb['a_hel(m)']).astype(float)*((numpy.array(df_orb['m1(kg)']).astype(float)/(3.0*pf.M_sun))**(1.0/3.0))
# df_orb['a/R_hill']=df_orb['a(m)']/df_orb['R_hill(m)']
# print df_orb.to_string()

pyplot.tight_layout()

#save the figure
script_name=os.path.basename(__file__).split('.')[0]
picname="{}_{}.png".format(script_name,fnum)
# picname="{}_1.pdf".format(script_name)
print "save {}".format(picname)
pyplot.savefig(picname,bbox_inches='tight')

picname="{}_{}.pdf".format(script_name,fnum)
print "save {}".format(picname)
pyplot.savefig(picname,bbox_inches='tight')

# pyplot.show()
pyplot.close()
#-------------------------------------------------------------------------------
