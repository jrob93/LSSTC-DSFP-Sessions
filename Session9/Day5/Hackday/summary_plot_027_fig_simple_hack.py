# -*- coding: utf-8 -*-
from matplotlib import rc
rc('font',**{'size':9})

import matplotlib.pyplot as pyplot
from matplotlib.collections import EllipseCollection
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import numpy
from pylab import *
import os
import sys
from optparse import OptionParser
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch
sys.path.insert(0, '../python_stuff/') #use this line in other completed runs places
import py_func as pf
reload(pf)

# use LaTeX fonts in the plot
pyplot.rc('text', usetex=True)
pyplot.rc('font', family='serif')

run_path="/Users/jrobinson/grav_cloud/cloud_runs_fix"
save_path="."
stat_loc="/Users/jrobinson/grav_cloud/python_stuff/storage_estimates/cloud_runs_fix_stats"

dirname='027_cloud_order_kelvin_fix'

#find dat*.txt files in directory and sort them
try:
    print '{}/{}'.format(run_path,dirname)
    files=next(os.walk('{}/{}'.format(run_path,dirname)))[2] #retrieve the files in the run directory
except:
    print("directory {} has no dat*.txt files?".format(dirname))

files = [ fi for fi in files if fi.endswith(".txt") and fi.startswith("dat") ] #keep only dat*.txt files
if len(files)==0:
    print("directory {} has no dat*.txt files?".format(dirname))
files.sort() #ensure that the files are always sorted the same?
final_files=[files[0]]
for j in range(1,len(files)):
    fnum=int(files[j][3:10])
    fnum2=int(files[j-1][3:10])
    if fnum!=fnum2:
        final_files.append(files[j])
    else:
        final_files[-1]=files[j]

#print final_files
final_files=numpy.array([final_files[0],final_files[len(final_files)/2],final_files[-1]])
print final_files

#READ THESE IN FROM RUN_PARAMS?
try:
    run_params=numpy.genfromtxt("{}/{}/run_params_0.txt".format(run_path,dirname),skip_header=1,dtype=None) #genfromtxt to avoid problems with string 'new' in run_params
except:
    #run_params=numpy.genfromtxt("{}/{}/run_params.txt".format(run_path,dirname),skip_header=1,dtype=None) #genfromtxt to avoid problems with string 'new' in run_params
    print 'not complete'

print run_params
#run parameters REMOVE THESE BITS?
Ntot=float(run_params[0])
X=float(run_params[7])
f=float(run_params[9])
Req=float(run_params[2])
rho=float(run_params[3])
r_min=Req*(Ntot**(-1.0/3.0))*f
r_max=Req*f
M_tot=4.0*numpy.pi*rho*(Req**3.0)/3.0 #kg
mp=M_tot/Ntot #kg
R_c=float(run_params[5])
print f,X,M_tot

#load r_b from problem file
damp_check=0
for line in open('{}/{}/problem.c'.format(run_path,dirname)):
    if 'double r_boulder=' in line:
        r_b=float(line.split('=')[-1][:-2])
    if 'reb_collision_resolve_merge_out_damp_cut' in line: #check which collision routine is used
        damp_check=1
if damp_check==0: #if the damped routine is not used we return a value of NaN for R_b
    r_b='NaN'

#set size scales for particles
s_min=1e0
s_max1=2e1
s_max2=2e2#e0
print "particle sizes: ",r_min,r_max
m_s = (s_max1-s_min)/(r_max-r_min)
m_s2=(s_max2-s_min)/(r_max**2.0-r_min**2.0)
c2=s_min-(m_s2*(r_min**2.0))

lim=10*R_c

#lim=2*R_c
print "lim = {}".format(lim)

#make plot
fig = pyplot.figure() #open figure once
cmap="viridis"
pyplot.rcParams['image.cmap'] = cmap

pc_tex=0.16605 # latex pc in inches
text_width=(39.0)*pc_tex
column_sep=1.0*pc_tex
column_width=(text_width-column_sep)/2.0
x_len=text_width+(column_sep)+(2.2*pc_tex) # add an additional couple points to fit
y_len=x_len/2.1
print "size: {}x{} inches".format(x_len,y_len)
fig.set_size_inches(x_len,y_len)

# fig.set_size_inches(15.5, 5.5)
# fig.set_size_inches(15.5, 10.5)

# gs = gridspec.GridSpec(1, 4,width_ratios=[1,1,1,0.1],wspace=0.0)
# ax1 = pyplot.subplot(gs[0,0])
# ax2 = pyplot.subplot(gs[0,1])
# ax3 = pyplot.subplot(gs[0,2])
# ax4 = pyplot.subplot(gs[0,3])

gs = gridspec.GridSpec(2, 3,height_ratios=[0.05,1],wspace=0.0,hspace=0.0)
ax1 = pyplot.subplot(gs[1,0])
ax2 = pyplot.subplot(gs[1,1])
ax3 = pyplot.subplot(gs[1,2])
ax4 = pyplot.subplot(gs[0,:])


a=[ax1,ax2,ax3]

ax1.set_xlabel('$x ~(\\mathrm{{m}})$')
ax2.set_xlabel('$x ~(\\mathrm{{m}})$')
ax3.set_xlabel('$x ~(\\mathrm{{m}})$')
ax1.set_ylabel('$y ~(\\mathrm{{m}})$')

ax2.set_yticks([])
ax3.set_yticks([])

for i in range(len(a)):
    a[i].set_aspect("equal")
    # a[i].set_xlabel('$x (m)$')
    # a[i].set_ylabel('$y (m)$')
    a[i].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    a[i].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    a[i].set_xlim(-lim,lim)
    a[i].set_ylim(-lim,lim)
    #load from data file
    fname="{}/{}/{}".format(run_path,dirname,final_files[i])
    t,df_dat=pf.load_dat_file(fname)
    df_dat=df_dat.sort_values('m(kg)')
    pos=numpy.array(df_dat[['x(m)','y(m)','z(m)']])
    vel=numpy.array(df_dat[['vx(ms^-1)','vy(ms^-1)','vz(ms^-1)']])
    m=numpy.array(df_dat['m(kg)'])
    r=numpy.array(df_dat['r(m)'])
    #find CoM
    CoM=pf.centre_of_mass(pos,m)
    #find index of largest element
    i_M=numpy.argmax(m)
    print "com ={} \nlargest particle = {}".format(CoM,i_M)
    #mass:color scale
    color=(numpy.log10(m)-numpy.log10(mp))/(numpy.log10(M_tot)-numpy.log10(mp))
    #marker sizes
    size1 = (m_s*r)+(s_min-(m_s*r_min))
    # pyplot.setp(a[i], title='$t={:.2e}s$'.format(t))
    a[i].text(0.98,0.97,'$t={:.1f}~\\mathrm{{yr}}$'.format(t/(365.0*24*60*60)),horizontalalignment='right',verticalalignment='top', transform=a[i].transAxes)

    alph_x=1.0

    if i==0:
        s1=a[i].scatter(pos[:,0],pos[:,1],lw=0.0,c=color,s=size1, vmin=0,vmax=1,zorder=1,rasterized=True)
        a[i].scatter(CoM[0],CoM[1],marker='x',c='k',zorder=2,alpha=alph_x)
    if i==1:
        #s1=a[i].scatter(pos[:,0]-pos[i_M,0],pos[:,1]-pos[i_M,1],lw=0.0,c=color,s=size1, vmin=0,vmax=1,zorder=1)
        s1=a[i].scatter(pos[:,0],pos[:,1],lw=0.0,c=color,s=size1, vmin=0,vmax=1,zorder=1,rasterized=True)
        a[i].scatter(0,0,marker='x',c='r',zorder=0)
        a[i].scatter(pos[i_M,0],pos[i_M,1],marker='+',c='r',zorder=2,alpha=alph_x)
        a[i].scatter(CoM[0],CoM[1],marker='x',c='k',zorder=1,alpha=alph_x)
        # a[i].scatter(CoM[0]-pos[i_M,0],CoM[1]-pos[i_M,1],marker='x',c='k',zorder=2,alpha=alph_x)
    if i==2:
        s1=a[i].scatter(pos[:,0],pos[:,1],lw=0.0,c=color,s=size1, vmin=0,vmax=1,zorder=1,rasterized=True)
        a[i].scatter(pos[i_M,0],pos[i_M,1],marker='+',c='r',zorder=2,alpha=alph_x)
        a[i].scatter(CoM[0],CoM[1],marker='x',c='k',zorder=1,alpha=alph_x)

#colourbar
cbar=fig.colorbar(s1,ax4,use_gridspec=True,orientation='horizontal') #ticks here correspond to the cmap colours
# cbar.set_label('$\\frac{\log{m}-\log{m_0}}{\log{M_{tot}}-\log{m_0}}$')
# ax4.set_title('$\\frac{\log{m}-\log{m_0}}{\log{M_{tot}}-\log{m_0}}$')
# ax4.set_ylabel('log(m_rel)')
ax4.set_ylabel('$m_\\mathrm{{rel}}$')
ax4.yaxis.set_label_coords(-0.05,1.02)
ax4.xaxis.tick_top()

#add inset
lim2=2.0*4.217537e+06#0.1*lim#1e8
zoom=200.0

axins = zoomed_inset_axes(ax3, zoom, loc=3)
axins.scatter(pos[:,0],pos[:,1],lw=0.0,c=color,s=size1*numpy.sqrt(zoom)/2.0, vmin=0,vmax=1,rasterized=True)
axins.scatter(pos[i_M,0],pos[i_M,1],marker='+',c='r',zorder=2,alpha=alph_x)

shr=1.0
x1, x2, y1, y2 = pos[i_M,0]-lim2/shr, pos[i_M,0]+lim2/shr, pos[i_M,1]-lim2/shr, pos[i_M,1]+lim2/shr # specify the limits

axins.set_xlim(x1, x2) # apply the x-limits
axins.set_ylim(y1, y2) # apply the y-limits

axins.set_xticks([])
axins.set_yticks([])

mark_inset(ax3, axins, loc1=2, loc2=4, fc="none", ec="0.0")
pyplot.draw()

print "load","{}/{}/{}_orbit_search.txt".format(run_path,dirname,dirname)
df_orb=pf.load_orb_file("{}/{}/{}_orbit_search.txt".format(run_path,dirname,dirname))
n_orb=len(df_orb[df_orb['t(s)']==numpy.amax(df_orb['t(s)'])])
print 'number of orbits = {}'.format(n_orb)
aph_list=[]
for i in range(n_orb):
    print df_orb.iloc[i]
    orbit=numpy.array(df_orb.iloc[i][['a(m)','e','I(rad)','omega(rad)','OMEGA(rad)','f_true(rad)']])
    pos_orb=pf.planet_orbit(orbit,100)
    axins.plot(pos_orb[:,0]+pos[i_M,0],pos_orb[:,1]+pos[i_M,1],zorder=0,c='r')
    break

pyplot.tight_layout()

#save the figure
script_name=os.path.basename(__file__).split('.')[0]
picname="{}/{}_{}.png".format(save_path,script_name,dirname)
print picname
pyplot.savefig(picname, bbox_inches='tight')

picname="{}/{}_{}.pdf".format(save_path,script_name,dirname)
print picname
pyplot.savefig(picname, bbox_inches='tight')

pyplot.close()
# pyplot.show()
