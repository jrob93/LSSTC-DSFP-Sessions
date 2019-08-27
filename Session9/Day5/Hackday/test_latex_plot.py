'''
'''

from matplotlib import rc
# rc('font',**{'size':9})

import matplotlib
# matplotlib.rcParams['mathtext.fontset'] = 'stix'
# matplotlib.rcParams['font.family'] = 'STIXGeneral'

# # Say, "the default sans-serif font is COMIC SANS"
# matplotlib.rcParams['font.sans-serif'] = "Comic Sans MS"
# # Then, "ALWAYS use sans-serif fonts"
# matplotlib.rcParams['font.family'] = "sans-serif"

import numpy
import matplotlib.pyplot as pyplot
import matplotlib.gridspec as gridspec
import matplotlib.colors
import os

# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern Sans serif']})
# rc('font',**{'family':'monospace','monospace':['Computer Modern Typewriter']})
# rc('font',**{'family':'monospace','monospace':['Courier']})

# use LaTeX fonts in the plot
pyplot.rc('text', usetex=True)
pyplot.rc('font', family='serif')

# create figure
fig = pyplot.figure()

gs = gridspec.GridSpec(1,1)
ax1 = pyplot.subplot(gs[0,0])

x=numpy.linspace(0,1)
y1=x
y2=x**2

Mtot=[6.54e19,1.77e21]

# ax1.scatter(x,y1,label="M_c={:.2e}kg".format(Mtot[0]))
# ax1.scatter(x,y2,label="M_c={:.2e}kg".format(Mtot[0]))
# ax1.set_xlabel("m_2/m_1")
# ax1.set_ylabel("(m_2+m_1)/M_c")

# ax1.scatter(x,y1,label="$M_{{\\mathrm{{c}}}}={:.2e} \/ \\mathrm{{kg}}$".format(Mtot[0]))
# ax1.scatter(x,y2,label="$M_{{\\mathrm{{c}}}}={:.2e} \/ \\mathrm{{kg}}$".format(Mtot[0]))
# ax1.set_xlabel("$m_2/m_1$")
# ax1.set_ylabel("$(m_2+m_1)/M_{{\\mathrm{{c}}}}$")

# # pretty similar to latex
# ax1.scatter(x,y1,label=r"$M_{\mathrm{c}}=%.2e ~ \mathrm{kg}$" % (Mtot[0]))
# ax1.scatter(x,y2,label=r"$M_{\mathrm{c}}=%.2e ~ \mathrm{kg}$" % (Mtot[0]))
# ax1.set_xlabel(r"$m_2/m_1$")
# ax1.set_ylabel(r"$(m_2+m_1)/M_{\mathrm{c}}$")

# # Use typewriter
# ax1.scatter(x,y1,label=r"$\mathtt{M_{\mathrm{c}}=%.2e \/ \mathrm{kg}}$" % (Mtot[0]))
# ax1.scatter(x,y2,label=r"$\mathtt{M_{\mathrm{c}}=%.2e \/ \mathrm{kg}}$" % (Mtot[0]))
# ax1.set_xlabel(r"$\mathtt{m_2/m_1}$")
# ax1.set_ylabel(r"$\mathtt{(m_2+m_1)/M_{\mathrm{c}}}$")

ax1.scatter(x,y1,label="$M_\\mathrm{{c}}=$~{:.2e}$~\\mathrm{{kg}}$".format(Mtot[0]))
ax1.scatter(x,y2,label="$M_\\mathrm{{c}}=$~{:.2e}$~\\mathrm{{kg}}$".format(Mtot[0]))
ax1.set_xlabel("$m_2/m_1$")
ax1.set_ylabel("$(m_2+m_1)/M_\\mathrm{{c}}$")

ax1.legend()

print "save {}.pdf".format(os.path.basename(__file__).split('.')[0])
pyplot.savefig("{}.pdf".format(os.path.basename(__file__).split('.')[0]), bbox_inches='tight')

pyplot.show()
