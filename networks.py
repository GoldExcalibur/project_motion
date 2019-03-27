import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import random


# networks
##############################################################################
class Encoder(nn.Module):
    def __init__(self, channels, kernel_size=8, activation='lrelu', convpool=False):
        super(Encoder, self).__init__()

        model = []
        if activation == 'lrelu':
            acti = nn.LeakyReLU(0.2)
        elif activation == 'relu':
            acti = nn.ReLU()
        elif activation == 'tanh':
            acti = nn.Tanh()
        else:
            raise NameError

        for i in range(len(channels) - 1):
            if not convpool:
                pad = (kernel_size - 2) // 2
                model.append(nn.ReflectionPad1d(pad))
                model.append(nn.Conv1d(channels[i], channels[i+1],
                                   kernel_size=kernel_size, stride=2))
                model.append(acti)
            else:
                pad = (kernel_size - 1) // 2
                model.append(nn.ReflectionPad1d(pad))
                model.append(nn.Conv1d(channels[i], channels[i+1],
                                       kernel_size=kernel_size, stride=1))
                model.append(acti)
                model.append(nn.MaxPool1d(kernel_size=2, stride=2))
            #odel.append(nn.Dropout(p=0.2))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x


class Decoder(nn.Module):
    def __init__(self, channels, kernel_size=7, activation='lrelu'):
        super(Decoder, self).__init__()

        model = []
        pad = (kernel_size - 1) // 2
        if activation == 'lrelu':
            acti = nn.LeakyReLU(0.2)
        elif activation == 'relu':
            acti = nn.ReLU()
        elif activation == 'tanh':
            acti = nn.Tanh()
        else:
            raise NameError

        for i in range(len(channels) - 1):
            model.append(nn.Upsample(scale_factor=2, mode='nearest'))
            model.append(nn.ReflectionPad1d(pad))
            model.append(nn.Conv1d(channels[i], channels[i + 1],
                                            kernel_size=kernel_size, stride=1))
            if i == 0 or i == 1:
                model.append(nn.Dropout(p=0.2))
            if not i == len(channels) - 2:
                model.append(acti)          # whether to add tanh a last?
                #model.append(nn.Dropout(p=0.2))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf, n_layers, en_ks=8, en_acti='lrelu', de_ks=7, de_acti='lrelu'):
        super(Generator, self).__init__()
        en_channels = [input_nc] + [ngf * (2 ** i) for i in range(n_layers)]
        de_channels = en_channels[:0:-1] + [output_nc]
        self.encoder = Encoder(en_channels, kernel_size=en_ks, activation=en_acti)
        self.decoder = Decoder(de_channels, kernel_size=de_ks, activation=de_acti)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, kw = 8, norm="BatchNorm"):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if norm == "BatchNorm":
            norm_layer = nn.BatchNorm1d
            use_bias = False
        else:
            norm_layer = None
            use_bias = True

        kw = kw # 4
        padw = (kw - 2) // 2 # 1
        sequence = [nn.Conv1d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv1d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
            ]
            if norm_layer is not None:
                sequence += [norm_layer(ndf * nf_mult)]
            sequence += [nn.LeakyReLU(0.2, True)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)

        sequence += [
            nn.Conv1d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw - 1, stride=1, padding=padw, bias=use_bias),
        ]
        if norm_layer is not None:
            sequence += [norm_layer(ndf * nf_mult)]
        sequence += [nn.LeakyReLU(0.2, True)]

        sequence += [nn.Conv1d(ndf * nf_mult, 1, kernel_size=kw - 1, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class MotionPool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if pool_size > 0:
            self.pool = []
            self.num_motions = 0

    def query(self, motions):
        if self.pool_size == 0:
            return motions
        return_motions = []
        for motion in motions:
            motion_data = torch.unsqueeze(motion.data, 0)
            if self.num_motions < self.pool_size:
                self.pool.append(motion_data)
                self.num_motions += 1
                return_motions.append(motion_data)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.pool[random_id].clone()
                    self.pool[random_id] = motion_data
                    return_motions.append(tmp)
                else:
                    return_motions.append(motion_data)
        return torch.cat(return_motions, 0)


# Functions
##############################################################################
def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler
