import os
import utils


class Config:
    proj_dir = '/data1/wurundi/CycleGAN_motion/'
    exp_name = os.getcwd().split('/')[-1]
    exp_dir = os.path.join(proj_dir, exp_name)
    log_dir = os.path.join(exp_dir, 'log/')
    model_dir = os.path.join(exp_dir, 'model/')

    stat_path = os.path.join(proj_dir, 'statistic.csv')

    device = None
    isTrain = None

    # data
    base_id = {"h": 0, "nh": 0}

    # model parameters
    input_nc = 63
    output_nc = 45
    ngf = 64
    ndf = 64
    n_layers_G = 3
    n_layers_D = 2
    G_en_ks = 8
    G_de_ks = 7
    D_ks = 8

    direction = 'AtoB'

    # training parameters
    niter = 50 # 100
    niter_decay = 50 # 100
    beta1 = 0.5
    lr = 0.0002
    gan_mode = 'lsgan'
    pool_size = 50
    lr_policy = 'linear'
    lr_decay_iters = 50

    # loss weight
    lambda_A = 10.0 # weight for cycle loss (A -> B -> A)
    lambda_B = 10.0 # weight for cycle loss (B -> A -> B)

    nr_epochs = niter + niter_decay
    batch_size = 64
    save_frequency = 10
    val_frequency = 50
    visualize_frequency = 200

    utils.ensure_dirs([proj_dir, log_dir, exp_dir, model_dir])


config = Config()
