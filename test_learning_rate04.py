# -*- coding: utf-8 -*-
# Test script for ConvNets
#
# Copyright (c) Favour M. Nyikosa <favour@nyikosa.com> 11-MAR-2018

import           learning_rate_test as test
import           time

num_epochs_cifar = 10
num_epochs_mnist = 1

tic_             = time.time()

tic              = time.time()
test.manager(    'softmax', 'adam', 'mnist', num_epochs_cifar )
toc              = time.time()
print(           'Experiment took ' + str( toc - tic ) + ' Seconds')

tic              = time.time()
test.manager(    'softmax', 'nesterov', 'mnist', num_epochs_cifar )
toc              = time.time()
print(           'Experiment took ' + str( toc - tic ) + ' Seconds')

tic              = time.time()
test.manager(    'softmax', 'sgd', 'mnist', num_epochs_cifar )
toc              = time.time()
print(           'Experiment took ' + str( toc - tic ) + ' Seconds')


toc_             = time.time()
print(           'All experiments took ' + str( toc_ - tic_ ) + ' Seconds')
