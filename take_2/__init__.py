"""
TAKE
Temperature Alteration Kinetic Elucidation is a method for analyzing the kinetics of reactions
performed using continuous addition of a species.
"""

# Imports
from take_2.take_prep_2 import read_data
from take_2.take_fitting_2 import sim_take
from take_2.take_fitting_2 import fit_take
from take_2.take_fitting_2 import fit_err_real
from take_2.take_plotting_2 import plot_conc_vs_time
from take_2.take_plotting_2 import plot_rate_vs_conc
from take_2.take_plotting_2 import plot_other_fits_2D
from take_2.take_plotting_2 import plot_other_fits_3D

# Handle versioneer
#from ._version import get_versions
#versions = get_versions()
#__version__ = versions['version']
#__git_revision__ = versions['full-revisionid']
#del get_versions, versions
#
#from ._version import get_versions
#__version__ = get_versions()['version']
#del get_versions