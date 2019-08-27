'''
Performs operations such as opening, running and copying the .py and .pdf files used in the grav collapse paper
'''

import numpy
import os
import subprocess

dest="/Users/jrobinson/qub_phd_writing/grav_cloud_paper_MNRAS/figures/test_figs"

fig_list=["read_real_binaries_period_hack",
"2d_Hill_fig_hack",
"summary_plot_027_fig_simple_hack",
"mean_velocity_dispersion2_hack",
"collision_velocity_stepsize_hack",
"plot_system_multiplicity",
"plot_particle_mass_hist_hack2",
"plot_binary_mass_ratios_hack_norm",
"plot_binary_mass_ratios_hack_log_classes_fig",
"plot_binary_mass_ratios_by_f_hack",
"plot_binary_magnitude_ratios_hack",
"plot_binary_absolute_magnitude_ratios_hack",
"plot_binary_orbits_fig_hack",
"read_real_binaries_polar_hack",
"plot_binary_orbits_fig_hack2",
"find_clusters_hack",
"mean_collision_velocity_hack"]

py_files=["{}.py".format(p) for p in fig_list]
print py_files

# # Find files
# for f in fig_list:
#     fig="{}.py".format(f)
#     subprocess.Popen("ls {}*.pdf".format(f),shell=True).wait()

pdf_list=["read_real_binaries_period_hack.pdf",
"2d_Hill_fig_hack.pdf",
"summary_plot_027_fig_simple_hack_027_cloud_order_kelvin_fix.pdf",
"mean_velocity_dispersion2_hack.pdf",
"collision_velocity_stepsize_hack_cloud_runs_fix_022_cloud_order_kelvin_fix.pdf",
"plot_system_multiplicity.pdf",
"plot_particle_mass_hist_hack2_df_plot_100_all_stable.pdf",
"plot_binary_mass_ratios_hack_norm_df_plot_100_all_stable.pdf",
"plot_binary_mass_ratios_hack_log_classes_fig_df_plot_100_all_stable.pdf",
"plot_binary_mass_ratios_by_f_hack_df_plot_100_all_stable.pdf",
"plot_binary_magnitude_ratios_hack_df_plot_100_all_stable.pdf",
"plot_binary_absolute_magnitude_ratios_hack_df_plot_100_all_stable.pdf",
"plot_binary_orbits_fig_hack_df_plot_100_all_stable.pdf",
"read_real_binaries_polar_hack.pdf",
"plot_binary_orbits_fig_hack2_df_plot_100_all_stable.pdf",
"find_clusters_hack_0000202.pdf",
"mean_collision_velocity_hack.pdf"]

print len(fig_list)
print len(pdf_list)

# # open py scripts in atom
# for p in py_files:
#     atom_cmd="atom {}".format(p)
#     print atom_cmd
#     subprocess.Popen(atom_cmd,shell=True).wait()
#     # break

# # run all py scripts
# for p in py_files:
#     py_cmd="python {}".format(p)
#     print py_cmd
#     subprocess.Popen(py_cmd,shell=True).wait()
#     # break

# git add py scripts
for p in py_files:
    py_cmd="git add {}".format(p)
    print py_cmd
    subprocess.Popen(py_cmd,shell=True).wait()
    # break

# # open all pdfs
# for i in range(len(pdf_list)):
#     open_cmd="open {} {}".format(pdf_list[i])
#     print open_cmd
#     subprocess.Popen(open_cmd,shell=True).wait()
#     # break

# # copy all pdfs to destination
# for i in range(len(pdf_list)):
#     cp_cmd="cp {} {}".format(pdf_list[i],dest)
#     print cp_cmd
#     subprocess.Popen(cp_cmd,shell=True).wait()
#     # break
