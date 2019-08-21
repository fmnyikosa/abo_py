# This module contains Bayesian Optimisation (BO) utilities.
#
# NOTE: Depends on GPflow
#
# Copyright (c) Favour Mandanji Nyikosa <favour@nyikosa.com>  6-OCT-2017

import numpy as numpy
import numpy.linalg as linalg
import scipy as scipy
import operator as op
import time as timer
import sys
import matplotlib.pyplot as pyplot
import os
import gpflow as GPflow
