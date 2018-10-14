# -*- coding: utf-8 -*-
# Module with tools for learning rate adaptation experiments with ABO
#
# Copyright (c) Favour M. Nyikosa <favour@nyikosa.com> 11-MAR-2018

# delegates learning rate adaptation ABO experiments
# models    - 'logistic' or 'multi' or 'conv'
# optimizer - 'adam'or 'nesterov' or 'sgd'
# data      - 'cifar-10' or 'mnist'
def manager( model, optimizer, data ):
    # check inputs match choices

    # set up model with specifications

    # train model


    # return results


def plot_comparison( standard_results, abo_results, save_to_file=0 ):
    # verify inputs

    # assign values

    # prepare figures

    # save to file if necesary





if __name__ == "__main__":
    manager('', 'sgd', 'minst')
