'''Script to plot a figure showing a 2 dimensional collision between two particles'''

from matplotlib import rc
rc('font',**{'size':12})

import numpy
import math
import matplotlib.pyplot as pyplot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib import patches
from mpl_toolkits.mplot3d import proj3d
import inspect,os

# use LaTeX fonts in the plot
pyplot.rc('text', usetex=True)
pyplot.rc('font', family='serif')

def line(p,v,t):
    r=p+(v*t)
    return r
def arrow_12(p1,p2,c,a):
    ax.arrow(p1[0],p1[1],p2[0]-p1[0],p2[1]-p1[1],length_includes_head='True',color=c,alpha=a,head_width=0.015,head_length=0.025)
def line3D(ax,r1,r2,ls,c):
    ax.plot([r1[0],r2[0]],[r1[1],r2[1]],[r1[2],r2[2]],ls,color=c)
def arrow(ax,r1,r2,c,arrs,ms,cs,ls):
    arr=Arrow3D([r1[0],r2[0]],[r1[1],r2[1]],[r1[2],r2[2]],connectionstyle=cs,mutation_scale=ms,lw=1, arrowstyle=arrs, color=c,linestyle=ls)
    ax.add_artist(arr)
def label(ax1,r1,r2,s,**kwargs):
    p=r1+r2
    ax1.text(p[0],p[1],p[2],s)

def label2D(ax1,r1,r2,s,**kwargs):
    p=r1+r2
    ax1.text(p[0],p[1],s,**kwargs)

def norm_vec(r1,r2):
    v=(r2-r1)/numpy.linalg.norm(r2-r1)
    return v
def half_line(r1,r2):
    p=line(r1,norm_vec(r1,r2),0.5*numpy.linalg.norm(r2-r1))
    return p
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs
    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)
def cm2inch(value):
    return value/2.54
def arrow2D(ax,p1,p2,lbl,arrs,cs):
    ax.annotate(lbl,xy=(p1[0], p1[1]), xycoords='data',xytext=(p2[0], p2[1]), textcoords='data',arrowprops=dict(arrowstyle=arrs,connectionstyle=cs))

def drawCirc(ax,radius,centX,centY,angle_,theta2_,color_='black',**kwargs):
    #========Line
    arc = patches.Arc([centX,centY],radius,radius,angle=angle_,
          theta1=0,theta2=theta2_,capstyle='round',linestyle='-',lw=1,color=color_,**kwargs)
    ax.add_patch(arc)


    #========Create the arrow head
    endX=centX+(radius/2)*numpy.cos(numpy.radians(theta2_+angle_)) #Do trig to determine end position
    endY=centY+(radius/2)*numpy.sin(numpy.radians(theta2_+angle_))

    ax.add_patch(                    #Create triangle as arrow head
        patches.RegularPolygon(
            (endX, endY),            # (x,y)
            3,                       # number of vertices
            radius/9,                # radius
            numpy.radians(angle_+theta2_),     # orientation
            color=color_
        )
    )

cos = numpy.cos
arccos = numpy.arccos
sin = numpy.sin
arcsin = numpy.arcsin
tan = numpy.tan
arctan = numpy.arctan
pi = numpy.pi
atan2=math.atan2

_0=numpy.array([0.0,0.0,0.0])
x=numpy.array([1.0,0.0,0.0])
y=numpy.array([0.0,1.0,0.0])
z=numpy.array([0.0,0.0,1.0])

p1=numpy.array([0.2,0.2,0.0])
p2=numpy.array([0.6,0.6,0.0])
p3=numpy.array([0.0,0.2,0.0])+p2

R1=0.05
R2=0.02
R3=0.25
R4=numpy.linalg.norm(p2-p1)
R_shift=0.05
R5=0.2
R_Om=0.15

v1=norm_vec(p1,p2)
p4=line(p2,v1,0.2)
v2=numpy.cross(z,v1)
p5=line(p2,v2,0.2)

y_rot=norm_vec(p5,p2)
x_rot=norm_vec(p4,p2)
shear_fac=-0.9

p6=line(p1,x,0.5)
p7=line(p1,y,0.5)

p8=line(p1,x,0.1)
p9=line(p1,v1,0.1)

tht=numpy.radians(-15)
p10=half_line(p1,p2)
p11=numpy.array([p10[0]*cos(tht)-p10[1]*sin(tht),p10[0]*sin(tht)+p10[1]*cos(tht),p10[2]])

p12=half_line(p8,p9)
p13=half_line(p10,p11)
angle_rot=numpy.radians(20)
print x
Rc_rot=numpy.array([(x[0]*numpy.cos(angle_rot))+(x[1]*numpy.sin(angle_rot)),(-x[0]*numpy.sin(angle_rot))+(x[1]*numpy.cos(angle_rot)),0])
print "Rc_rot = {}".format(Rc_rot)
p14=line(p2,Rc_rot,R3)

points=[p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13]

shear_points=[p2,
p2+(R_shift*(p2/numpy.linalg.norm(p2))),
p2+(2.0*R_shift*(p2/numpy.linalg.norm(p2))),
p2-(R_shift*(p2/numpy.linalg.norm(p2))),
p2-(2.0*R_shift*(p2/numpy.linalg.norm(p2)))]

#define circle/line objects
c1 = pyplot.Circle((p1[0],p1[1]),R1, color='yellow')
c2 = pyplot.Circle((p3[0],p3[1]),R2, color='green')
c3 = pyplot.Circle((p2[0],p2[1]),R3, color='red',alpha=0.5,zorder=0)
# c4 = pyplot.Circle((p1[0],p1[1]),R4, color='blue',fill=False,linestyle=":",zorder=1,clip_on=False)
# c5 = pyplot.Circle((p1[0],p1[1]),R4+R_shift, color='blue',fill=False,linestyle=":",zorder=1,clip_on=False)
# c6 = pyplot.Circle((p1[0],p1[1]),R4+(2.0*R_shift), color='blue',fill=False,linestyle=":",zorder=1,clip_on=False)
# c7 = pyplot.Circle((p1[0],p1[1]),R4-R_shift, color='blue',fill=False,linestyle=":",zorder=1,clip_on=False)
# c8 = pyplot.Circle((p1[0],p1[1]),R4-(2.0*R_shift), color='blue',fill=False,linestyle=":",zorder=1,clip_on=False)
c4 = pyplot.Circle((p1[0],p1[1]),R4, color='blue',fill=False,linestyle=":",zorder=1)
c5 = pyplot.Circle((p1[0],p1[1]),R4+R_shift, color='blue',fill=False,linestyle=":",zorder=1)
c6 = pyplot.Circle((p1[0],p1[1]),R4+(2.0*R_shift), color='blue',fill=False,linestyle=":",zorder=1)
c7 = pyplot.Circle((p1[0],p1[1]),R4-R_shift, color='blue',fill=False,linestyle=":",zorder=1)
c8 = pyplot.Circle((p1[0],p1[1]),R4-(2.0*R_shift), color='blue',fill=False,linestyle=":",zorder=1)

circles=[c1,c2,c3,c4,c5,c6,c7,c8]

#create figure
fig, ax = pyplot.subplots() # note we must use pyplot.subplots, not pyplot.subplot

pc_tex=0.16605 # latex pc in inches
text_width=43.0*pc_tex
s_x=1.0
s_y=1.0
x_len=((text_width/2.0)+(1.5*pc_tex))*s_x
y_len=(x_len)*s_y
print "size: {}x{} inches".format(x_len,y_len)
fig.set_size_inches(x_len,y_len)

ax.set_aspect("equal")

# for i,p in enumerate(points):
#     ax.text(p[0],p[1],"{}".format(i+1),color="r",zorder=99)

for c in circles:
    ax.add_artist(c)

for i,sp in enumerate(shear_points):
    print sp
    ax.scatter(sp[0],sp[1],color="b",s=10,zorder=1)

    if i>0:
        p=line(sp,y_rot,-(shear_fac)*numpy.dot(p2-sp,x_rot))
        # ax.scatter(p[0],p[1])
        arrow_12(sp,p,"b",0.5)

ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
col='k'
alph=1.0
arrow_12(p2,p4,col,alph)
arrow_12(p2,p5,col,alph)
arrow_12(p1,p6,col,alph)
arrow_12(p1,p7,col,alph)

alph=0.5
arrow_12(p1,p2,col,alph)
arrow_12(p2,p3,col,alph)
arrow_12(p2,p14,col,alph)

angle=numpy.arccos(numpy.dot(p9-p1,p8-p1)/(numpy.linalg.norm(p9-p1)*numpy.linalg.norm(p8-p1)))
print angle,numpy.degrees(angle)
# arrow2D(ax,p9,p8,"","->","arc3,rad={}".format(angle))
drawCirc(ax,numpy.linalg.norm(p8-p1)*2.0,p1[0],p1[1],0,numpy.degrees(angle)-13)
drawCirc(ax,R_Om,p2[0],p2[1],0,300,color_="r",zorder=3)
# arrow2D(ax,p10,p11,"","->","arc3,rad={}".format(numpy.radians(15)))

label2D(ax,p6,0.01*x,'$X$')
label2D(ax,p7,_0,'$Y$')
# label2D(ax,p12,_0,'$\phi$')
label2D(ax,p4,0.01*x,'$x$')
label2D(ax,p5,0.015*y,'$y$')
label2D(ax,p14,0.025*y,'$R_\\mathrm{{c}}$',horizontalalignment='center')
label2D(ax,p3,0.025*x,'$\\mathbf{{r}}$')
label2D(ax,p2-(0.12*y),_0,'$\\Omega$',color="r",horizontalalignment='center')

ax.set_xlim([0.1,0.9])
ax.set_ylim([0.1,0.9])

# pyplot.tight_layout()
pyplot.axis('off')

base=inspect.getfile(inspect.currentframe()) # script filename (usually with path)
base=os.path.splitext(base)[0]
picname="{}.png".format(base)
print picname
pyplot.savefig(picname, bbox_inches='tight')

picname="{}.pdf".format(base)
print picname
pyplot.savefig(picname,bbox_inches='tight')

pyplot.close()
# pyplot.show()
