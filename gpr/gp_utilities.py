#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Gaussian Process Library
# Favour Mandanji Nyikosa
# May 19 2016
#
import numpy as np
# squared exponential kernel
# theta_1 * exp
exponential_kernel = lambda x, y, params: theta[0]**2 * \
    np.exp( -0.5 * (1/theta[1]**2) * np.sum((x - y)**2) )
