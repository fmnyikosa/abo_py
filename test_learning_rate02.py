# -*- coding: utf-8 -*-
# Test script for Nesterov
#
# Copyright (c) Favour M. Nyikosa <favour@nyikosa.com> 11-MAR-2018

import           learning_rate_test as test
import           time

num_epochs_cifar = 100
num_epochs_mnist = 20

tic_             = time.time()

tic              = time.time()
test.manager(    'softmax',       'nesterov', 'mnist', num_epochs_mnist )
toc              =   time.time()
print(           'Experiment took ' + str( toc - tic ) + ' Seconds')

tic              = time.time()
test.manager(    'softmax',       'nesterov', 'cifar10', num_epochs_cifar )
toc              =   time.time()
print(           'Experiment took ' + str( toc - tic ) + ' Seconds')

tic              = time.time()
test.manager(    'multilayer',    'nesterov', 'mnist', num_epochs_mnist )
toc              =   time.time()
print(          'Experiment took ' + str( toc - tic ) + ' Seconds')

tic              = time.time()
test.manager(    'multilayer',    'nesterov', 'cifar10', num_epochs_cifar )
toc              =   time.time()
print(           'Experiment took ' + str( toc - tic ) + ' Seconds')
tic              = time.time()

# tic = time.time()
# test.manager('convolutional', 'nesterov', 'cifar10', num_epochs_cifar )
# toc = time.time()
# print(  'Experiment took ' + str( toc - tic ) + ' Seconds')

toc_             = time.time()
print(           'All experiments took ' + str( toc_ - tic_ ) + ' Seconds')
