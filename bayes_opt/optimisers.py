import cma
import sys
import os
import scipy               as sp
import numpy               as np
import time                as timer
import operator            as op
import GPflow              as gpflow
import numpy.linalg        as la
import matplotlib.pyplot   as plt
import AcqusitionFunctions as af
from   DIRECT              import solve
from   os.path             import expanduser as eu


def optimiseAF(x0, gpModel, settings):
	"""
	This function optimises an acqusition function using various optimisation
	methods
	x0 - optimisation starting point (ND array)
	gpModel - trained GP Model (GPy)
	settings - dictinary with settings
	"""

	optimiser_choice = settings['acqf_optimiser']

	if optimiser_choice == 'cmaes':

	elif optimiser_choice == 'direct':
