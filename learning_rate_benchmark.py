# -*- coding: utf-8 -*-
# Module with components for learning rate adaptation benchmark experiments with ABO. It
# contains componets for the neural network models, datasets (cifar10 and minst).
#
# Classes: (defined only for neural network models)
#   LogisticRegression
#   ConvNet
#
# Functions: (for data management and auxillary functionality)
#       get_minst_data()
#       get_cifar10_data()
#       build_multilayer_model()
#       manager( model_, optimizer_, data_ )
#
# Copyright (c) Favour M. Nyikosa <favour@nyikosa.com> 11-MAR-2018

import time
import torch
import torch.nn               as nn
import numpy                  as np
import scipy.io               as sio
import torch.nn.functional    as F

import torchvision
import torchvision.datasets   as dsets
import torchvision.transforms as transforms

from   datetime               import datetime
from   torch.autograd         import Variable
from   nn_models              import *


##########################################################################################
#                               NEURAL NETWORK MODELS
#-----------------------------------------------------------------------------------------

# Logistic Regression Model (also called SoftMax Regression)
class LogisticRegression( nn.Module ):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward( self, x ):
        x
        out = self.linear(x)
        return out

# Convolutional Neural network model
class ConvNet( nn.Module ):
    def __init__( self, input_size, num_classes ):
        super( ConvNet , self).__init__()
        # 3 input image channels, 6 output channels, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        # 2x2 max pool layer
        self.pool  = nn.MaxPool2d(2, 2)
        # 6 input image channels, 16 output channels, 5x5 square convolution kernel
        self.conv2 = nn.Conv2d(6, 16, 5)
        # affine operations: y = Wx + b
        self.fc1   = nn.Linear(16 * 5 * 5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, num_classes)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = self.pool( F.relu( self.conv1(x) ) )
        x = self.pool( F.relu( self.conv2(x) ) )
        x = x.view( -1, 16 * 5 * 5 )
        x = F.relu( self.fc1(x) )
        x = F.relu( self.fc2(x) )
        x = self.fc3( x )
        return x

    def num_flat_features(self, x):
        # all dimensions except the batch dimension
        size         = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def build_multilayer_model_2(input_dim, output_dim):
    hidden  = 1000
    model   = torch.nn.Sequential()
    model.add_module("linear_1",   torch.nn.Linear(input_dim, hidden,     bias=False) )
    model.add_module("relu_1",     torch.nn.ReLU() )
    model.add_module("dropout_1",  torch.nn.Dropout(0.2))
    model.add_module("linear_2",   torch.nn.Linear(hidden,    hidden,     bias=False) )
    model.add_module("relu_2",     torch.nn.ReLU())
    model.add_module("dropout_2",  torch.nn.Dropout(0.2))
    model.add_module("linear_3",   torch.nn.Linear(hidden,    output_dim, bias=False) )
    return model

def build_multilayer_model_1(input_dim, output_dim):
    hidden  = 1000
    model   = torch.nn.Sequential()
    model.add_module("linear_1",   torch.nn.Linear(input_dim, hidden,     bias=False) )
    model.add_module("relu_1",     torch.nn.ReLU() )
    model.add_module("dropout_1",  torch.nn.Dropout(0.2))
    model.add_module("linear_2",   torch.nn.Linear(hidden,    output_dim,     bias=False) )
    return model

def build_vgg_model(input_dim, output_dim):
    hidden  = 1000
    model   = torch.nn.Sequential()
    model.add_module("linear_1",   torch.nn.Linear(input_dim, hidden, bias=False) )
    model.add_module("relu_1",     torch.nn.ReLU() )
    model.add_module("dropout_1",  torch.nn.Dropout(0.2))
    model.add_module("linear_2",   torch.nn.Linear(hidden, hidden, bias=False))
    model.add_module("relu_2",     torch.nn.ReLU())
    model.add_module("dropout_2",  torch.nn.Dropout(0.2))
    model.add_module("linear_3",   torch.nn.Linear(hidden, output_dim, bias=False))
    return model


##########################################################################################
#                                   DATA LOADING
#-----------------------------------------------------------------------------------------
def get_minst_data(): # 28 * 28
    batch_size = 128
    transform = transforms.Compose(
                                [transforms.ToTensor(),
                                 transforms.Resize( (784,1) )])
    # MNIST Dataset (Images and Labels)
    train_dataset = dsets.MNIST(root='./data',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)
    test_dataset = dsets.MNIST( root='./data',
                                train=False,
                                transform=transforms.ToTensor())
    # Dataset Loader (Input Pipline)
    trainloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    testloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)
    print('Number of training images: ' + str( train_dataset.__len__() ) )
    print('Number of test images: '     + str( test_dataset.__len__()  ) )
    return trainloader , testloader

def get_cifar10_data(): # 32 * 32
    batch_size = 128

    # transform = transforms.Compose(
    #                             [transforms.ToTensor(),
    #                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5) ) ])
    # trainset = torchvision.datasets.CIFAR10( root='./data', train=True,
    #                                           download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
    #                                           shuffle=True, num_workers=2)


    # testset = torchvision.datasets.CIFAR10( root='./data', train=False,
    #                                           download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader( testset, batch_size=batch_size,
    #                                           shuffle=False, num_workers=2)

    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


    print('Number of training images: ' + str( trainset.__len__() ) )
    print('Number of test images: '     + str( testset.__len__()  ) )
    return trainloader , testloader


##########################################################################################
#                                EXPERIMENT MANAGEMENT
#-----------------------------------------------------------------------------------------

# get gradient vectors for hyper-gradient calculations
def get_grad_list(optimizer):
    gradients = []
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            d_p = p.grad.data
            gradients.append(d_p)
    return gradients

# resizes images if model is not convolutional
def resize_images(images, height):
    return images.view(-1, height*height)


# manages experiments
def manager( model_ , optimizer_ , data_, n_epochs = 1 ):

    # manage benchmark training rates
    base_bechmark_mnist0         = [    1e-1 ,   1e-2 ,   1e-3 ,   1e-4, 1e-5      ]
    base_bechmark_mnist1         = [  1*1e-5 , 1*1e-4 , 1*1e-3 , 1*1e-2, 1*1e-2    ]

    benchmark_etas_minst         = [  2*1e-2 , 1*1e-2 , 1*1e-3 , 1*1e-4, 1*1e-5,  \
                                      1*1e-4 , 1*1e-3 , 1*1e-2 , 2*1e-2, 3*1e-2,  \
                                      1*1e-3 , 1*1e-4 , 1*1e-5 , 1*1e-4, 1*1e-3,  \
                                      1*1e-2 , 2*1e-2 , 3*1e-2 , 1*1e-3, 1*1e-4,  \
                                      1*1e-5 , 1*1e-4 , 1*1e-3 , 1*1e-2, 2*1e-2    ]

    base_bechmark_cifar0         = [  1*1e-1 , 1*1e-2 , 1*1e-3 , 1*1e-4, 1*1e-5  ]
    base_bechmark_cifar1         = [  1*1e-5 , 1*1e-4 , 1*1e-3 , 1*1e-2, 1*1e-1  ]

    benchmark_etas_cifar         = [  2*1e-2 , 1*1e-2 , 1*1e-3 , 1*1e-4, 1*1e-5,  \
                                      1*1e-4 , 1*1e-3 , 1*1e-2 , 1*1e-2, 2*1e-2,  \
                                      1*1e-3 , 1*1e-4 , 1*1e-2 , 1*1e-4, 1*1e-3,  \
                                      2*1e-2 , 3*1e-2 , 1*1e-2 , 1*1e-3, 1*1e-4,  \
                                      1*1e-5 , 1*1e-4 , 1*1e-3 , 1*1e-2, 2*1e-2,  \
                                      3*1e-2 , 1*1e-3 , 1*1e-4 , 1*1e-5, 1*1e-4,  \
                                      1*1e-3 , 1*1e-2 , 2*1e-2 , 3*1e-2, 1*1e-3,  \
                                      1*1e-4 , 1*1e-5 , 1*1e-4 , 1*1e-3, 1*1e-2,  \
                                      2*1e-2 , 3*1e-2 , 1*1e-3 , 1*1e-4, 1*1e-5,  \
                                      1*1e-2 , 1*1e-3 , 1*1e-5 , 1*1e-2, 2*1e-2,  \
                                      1*1e-3 , 1*1e-4 , 1*1e-5 , 1*1e-4, 1*1e-3,  \
                                      1*1e-2 , 2*1e-2 , 3*1e-2 , 1*1e-3, 1*1e-4,  \
                                      1*1e-5 , 1*1e-4 , 1*1e-3 , 1*1e-2, 2*1e-2,  \
                                      3*1e-2 , 1*1e-3 , 1*1e-4 , 1*1e-5, 1*1e-4,  \
                                      1*1e-3 , 1*1e-2 , 1*1e-4 , 1*1e-2, 1*1e-3,  \
                                      1*1e-4 , 1*1e-5 , 1*1e-4 , 1*1e-3, 1*1e-2,  \
                                      2*1e-2 , 3*1e-2 , 1*1e-3 , 1*1e-4, 1*1e-5,  \
                                      1*1e-4 , 1*1e-3 , 1*1e-2 , 2*1e-2, 3*1e-2,  \
                                      1*1e-3 , 1*1e-4 , 1*1e-5 , 1*1e-4, 1*1e-3,  \
                                      1*1e-2 , 2*1e-2 , 3*1e-2 , 1*1e-3, 1*1e-4    ]

    print('\n------------------------------------------')
    print('Running ' + model_ + ' on ' + data_ + ' dataset with ' + optimizer_ )

    # catch any errors in inputs
    if  model_   == 'softmax' and data_ == 'cifar10':
        raise ValueError('Unfortunately, I cannot test Softmax on CIFAR10 yet.')
    elif model_   == 'multilayer1' and data_ == 'cifar10':
        raise ValueError('Unfortunately, I cannot test Multilayer1 on CIFAR10 yet.')
    elif model_   == 'multilayer2' and data_ == 'cifar10':
        raise ValueError('Unfortunately, I cannot test Multilayer2 on CIFAR10 yet.')
    elif model_   == 'convolutional' and data_ == 'mnist':
        raise ValueError('Unfortunately, I cannot test LeNet on MNIST yet.')

    # initialize data based on passed option data_
    print('==> Preparing data..')
    if  data_   == 'mnist':
        train_data, test_data = get_minst_data()
        input_dim  = 784
        output_dim = 10
        benchmarks = benchmark_etas_minst
    elif data_  == 'cifar10':
        train_data, test_data = get_cifar10_data()
        input_dim  = 3072
        output_dim = 10
        benchmarks = benchmark_etas_cifar
    else:
        raise ValueError(
        'You passed an unidentified dataset string descriptor to the experiment manager.')

    # initialize model based on passed option model_
    print('==> Building model..')
    if  model_ ==  'softmax':         # softmax or logistic regression model
        model = LogisticRegression( input_dim , output_dim )
    elif model_ == 'multilayer1':     # multi-layer neural network  - 1 hidden layer
        model = build_multilayer_model_1( input_dim , output_dim )
    elif model_ == 'multilayer2':     # multi-layer neural network  - 2 hidden layers
        model = build_multilayer_model_2( input_dim , output_dim )
    elif model_ == 'convolutional':   # convolutional neural network - LeNet
        model = ConvNet( input_dim , output_dim )
    elif model_ == 'vgg':             # convolutional neural network
        model = build_vgg_model( input_dim , output_dim )
    else:
        raise ValueError(
        'You passed an unidentified model string descriptor to the experiment manager.')

    # initialize learning rate and related variaables
    learning_rate = 0.001
    momentum_     = 0.900
    weight_decay  = 1e-4   # regularization

    # initialize optimizer based on passed option optimizer_
    if  optimizer_   == 'sgd':
        optimizer = torch.optim.SGD(  model.parameters(), lr=learning_rate,
                                        weight_decay=weight_decay )
    elif optimizer_  == 'nesterov':
        optimizer = torch.optim.SGD(  model.parameters(), lr=learning_rate,
                                        weight_decay=weight_decay,
                                               momentum=momentum_, nesterov=1 )
    elif optimizer_  == 'adam':
        optimizer = torch.optim.Adam( model.parameters(), lr=learning_rate,
                                        weight_decay=weight_decay )
    else:
        raise ValueError(
        'You passed an unidentified optimizer string descriptor to the experiment manager.')

    # initialize loss / ciriterion
    criterion                    = nn.CrossEntropyLoss()

    # pre-allocate key output variables
    eta_epochs, eta_iterations   =  [] , []
    loss_epochs, loss_iterations =  [] , []
    loss_epochs_                 =  [] # store non summed losses
    gradients_epochs             =  []
    gradients_epochs_            =  [] # store non summed gradients
    accuracy_epoch               =  []
    last_loss                    =  0
    last_eta                     =  0
    current_eta                  =  2*1e-2
    current_loss                 =  0
    iterations                   =  1
    current_gradient             =  0.0
    previous_gradient1           =  0.0
    previous_gradient2           =  0.0
    last_iter_gradient           =  0.0
    this_iter_gradient           =  0.0
    last_epoch_gradient          =  0.0
    this_epoch_gradient          =  0.0

    number_burn_in               =  3

    # training
    print('==> Training model..')
    for epoch in range(n_epochs):

        running_loss     = 0.0
        running_gradient = 0.0

        # set learning rate
        last_eta         = current_eta
        current_eta      = benchmarks[ epoch ]

        # batch training loop    
        for i , (images, labels) in enumerate( train_data , 0 ):

            if model_ != 'convolutional':
                if data_ == 'mnist':
                    resized_images = images.view( -1, 28*28  ) # resize images
                else:
                    resized_images = images.view( -1, 32*32*3) # resize images
            else:
                resized_images     = images

            # wrap them in Variable
            inputs           = Variable( resized_images )
            labels           = Variable( labels         )

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs          = model( inputs )
            loss             = criterion( outputs , labels )

            loss.backward()
            optimizer.step_abo( current_eta )

            last_loss        = current_loss
            current_loss     = loss.item()

            # update output lists
            loss_iterations.append( current_loss  )
            eta_iterations.append(  current_eta   )

            # per-batch gradient management
            previous_gradient2  = previous_gradient1
            previous_gradient1  = current_gradient
            current_gradient    = get_grad_list( optimizer )[ 0 ]

            # calculate hypergradient according as done in Almeida (1998)
            if i > 2:
                gradients_i         = previous_gradient2.view( -1  , previous_gradient2.numel() )
                gradients_j         = previous_gradient1.view(  previous_gradient1.numel() , -1 )
                last_iter_gradient  = this_iter_gradient
                this_iter_gradient  = torch.mm( gradients_i , gradients_j )

            # update iteration count
            iterations         += 1

            # update running loss
            running_loss       += current_loss

            # calculate running loss
            current_average_loss= running_loss/iterations

            # update running gradient
            if i > 2:
                running_gradient+= this_iter_gradient.item()

            # print statistics
            num_mini_batches    = 80
            if i % num_mini_batches == num_mini_batches - 1: # print every 2000 mini-batches
                print(       '[%d, %5d] loss: %.3f' %
                      (      epoch + 1, i + 1, current_average_loss ))

        # update output lists
        # print 'iterations:   ' + str( iterations           )
        # print 'running loss: ' + str( current_average_loss )
        loss_epochs.append(   current_average_loss  )
        eta_epochs.append(    current_eta   )
        loss_epochs_.append(  loss_iterations[ -2 ]   )

        # calculate hypergradient according as done in Almeida (1998)
        gradients_i          = previous_gradient2.view( -1  , previous_gradient2.numel() )
        gradients_j          = previous_gradient1.view(  previous_gradient1.numel() , -1 )
        this_epoch_gradient  = torch.mm( gradients_i , gradients_j )

        current_average_grad = running_gradient/iterations
        gradients_epochs.append(    current_average_grad   )
        gradients_epochs_.append(   this_epoch_gradient    )

        print 'Epoch                 : ' + str( epoch     )
        print 'Proposed learning rate: ' + str( last_eta     )
        print 'Associated loss       : ' + str( current_average_loss )

    print ' '
    print('Finished benchmark training of training Data...')
    print ' '

    mat_name = './logs/' + 'bench-' +  model_ + '-' + optimizer_ + '-' + data_ + '-' + \
    datetime.now().strftime('%d-%B-%Y-%H-%M-%S-') + str(n_epochs) + '-epochs' + '.mat'

    sio.savemat( mat_name , {
        'iterations':           iterations,
        'eta_epochs':           eta_epochs,
        'loss_epochs':          loss_epochs,
        'loss_epochs_':         loss_epochs_,
        'gradients_epochs':     gradients_epochs,
        'gradients_epochs_':    gradients_epochs_,
        'eta_iterations':       eta_iterations,
        'loss_iterations':      loss_iterations } )

##########################################################################################
#                                   DEFAULT RUN
#-----------------------------------------------------------------------------------------
if __name__ == "__main__":

    tic_ = time.time()

    num_epochs_cifar = 100
    num_epochs_mnist = 25

    # Nesterov
    tic = time.time()
    manager('convolutional', 'nesterov', 'cifar10', num_epochs_cifar )
    toc =   time.time()
    print(  'Experiment took ' + str( toc - tic ) + ' Seconds')

    tic = time.time()
    manager('softmax',       'nesterov', 'mnist', num_epochs_mnist  )
    toc =   time.time()
    print(  'Experiment took ' + str( toc - tic ) + ' Seconds')

    tic = time.time()
    manager('multilayer1',    'nesterov', 'mnist', num_epochs_mnist  )
    toc =   time.time()
    print(  'Experiment took ' + str( toc - tic ) + ' Seconds')

    tic = time.time()
    manager('multilayer2',    'nesterov', 'mnist', num_epochs_mnist   )
    toc =   time.time()
    print(  'Experiment took ' + str( toc - tic ) + ' Seconds')
    tic = time.time()


    # Adam
    tic = time.time()
    manager('convolutional', 'adam', 'cifar10', num_epochs_cifar )
    toc =   time.time()
    print(  'Experiment took ' + str( toc - tic ) + ' Seconds')

    tic = time.time()
    manager('softmax',       'adam', 'mnist', num_epochs_mnist  )
    toc =   time.time()
    print(  'Experiment took ' + str( toc - tic ) + ' Seconds')

    tic = time.time()
    manager('multilayer1',    'adam', 'mnist', num_epochs_mnist  )
    toc =   time.time()
    print(  'Experiment took ' + str( toc - tic ) + ' Seconds')

    tic = time.time()
    manager('multilayer2',    'adam', 'mnist', num_epochs_mnist )
    toc =   time.time()
    print(  'Experiment took ' + str( toc - tic ) + ' Seconds')
    tic = time.time()

    # SGD
    tic = time.time()
    manager('convolutional', 'sgd', 'cifar10', num_epochs_cifar )
    toc =   time.time()
    print(  'Experiment took ' + str( toc - tic ) + ' Seconds')

    tic = time.time()
    manager('softmax',       'sgd', 'mnist', num_epochs_mnist  )
    toc =   time.time()
    print(  'Experiment took ' + str( toc - tic ) + ' Seconds')

    tic = time.time()
    manager('multilayer1',    'sgd', 'mnist', num_epochs_mnist  )
    toc =   time.time()
    print(  'Experiment took ' + str( toc - tic ) + ' Seconds')

    tic = time.time()
    manager('multilayer2',    'sgd', 'mnist', num_epochs_mnist )
    toc =   time.time()
    print(  'Experiment took ' + str( toc - tic ) + ' Seconds')


    toc_ =   time.time()
    print('------------------------------------------')
    print(  '    '  )
    print(  'All the experiments took ' + str(  toc_ - tic_ ) +             ' Seconds')
    print(  'All the experiments took ' + str( (toc_ - tic_ )/60        ) + ' Minutes')
    print(  'All the experiments took ' + str( (toc_ - tic_ )/(60 * 40) ) + ' Hours'  )
    print(  '    '  )
    print('------------------------------------------')
    print(  '    '  )
    print(  'Experiments completed at: ' + datetime.now().strftime('%H:%M:%S')  )
    print(  '    '  )

