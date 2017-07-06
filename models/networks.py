import torch
import torch.nn as nn
<<<<<<< HEAD
from torch.nn import init
=======
>>>>>>> update models
import functools
from torch.autograd import Variable
import numpy as np
###############################################################################
# Functions
###############################################################################


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm)
    return norm_layer

def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, gpu_ids=[], input_size=None, latent_nc=None, latent_z=None):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    if latent_nc is not None and which_model_netG == 'resnet_5blocks':
        netG_to_latent = LatentResnetGenerator(input_nc, latent_nc, [input_nc, input_size, input_size], ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=5, latent_z=latent_z, gpu_ids=gpu_ids)
        netG_from_latent = ResnetGenerator(latent_nc+latent_z, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=5, gpu_ids=gpu_ids)
        netG = (netG_to_latent, netG_from_latent)
    elif latent_nc is not None and which_model_netG == 'resnet_3blocks':
        netG_to_latent = LatentResnetGenerator(input_nc, latent_nc, [input_nc, input_size, input_size], ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3, latent_z=latent_z, gpu_ids=gpu_ids)
        netG_from_latent = ResnetGenerator(latent_nc+latent_z, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3, gpu_ids=gpu_ids)
        netG = (netG_to_latent, netG_from_latent)
    elif which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    else:
        print('Generator model name [%s] is not recognized' % which_model_netG)
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)

    if type(netG) == tuple:
        for net in netG:
            if len(gpu_ids) > 0:
                net.cuda(device_id=gpu_ids[0])
            net.apply(weights_init)
    else:
        if len(gpu_ids) > 0:
            netG.cuda(device_id=gpu_ids[0])
        netG.apply(weights_init)
    return netG


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    if use_gpu:
        netD.cuda(device_id=gpu_ids[0])
    netD.apply(weights_init)
    return netD

def define_D_1d(input_size, which_model_netD, n_layers_D=3, use_sigmoid=False,
                gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator1d(input_size, n_layers=3, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator1d(input_size, n_layers=n_layers_D, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    else:
        print('Discriminator model name [%s] is not recognized' %
              which_model_netD)
    if use_gpu:
        netD.cuda(device_id=gpu_ids[0])
    netD.apply(weights_init)
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.label_vars = {}
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        var = self.label_vars.get(target_is_real)
        if (var is None) or (var.numel() != input.numel()):
            if target_is_real is True:
                label = self.real_label
            elif target_is_real is False:
                label = self.fake_label
            else:
                label = target_is_real * self.real_label + (1 - target_is_real) * self.fake_label
            tensor = self.Tensor(input.size()).fill_(label)
            self.label_vars[target_is_real] = var = Variable(tensor, requires_grad=False)
        return var

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

class LatentResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, input_shape, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=3, latent_z=8, gpu_ids=[]):
        assert(n_blocks >= 0)
        assert(latent_z >= 0)
        super().__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids

        pre_model = [nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3),
                     norm_layer(ngf, affine=True),
                     nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            pre_model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1),
                          norm_layer(ngf * mult * 2, affine=True),
                          nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            pre_model += [ResnetBlock(ngf * mult, 'zero', norm_layer=norm_layer, use_dropout=use_dropout)]

        self.pre_model = nn.Sequential(*pre_model)

        latent_model = []
        conv_z_model = []
        linear_z_model = []

        # upsample latent path
        for i in range(n_downsampling):
            mult_latent = 2**(n_downsampling - i)
            latent_model += [nn.ConvTranspose2d(ngf * mult_latent, int(ngf * mult_latent / 2),
                                       kernel_size=3, stride=2,
                                       padding=1, output_padding=1),
                              norm_layer(int(ngf * mult_latent / 2), affine=True),
                              nn.ReLU(True)]

        latent_model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3)]
        latent_model += [nn.Tanh()]
        self.latent_model = nn.Sequential(*latent_model)

        # downsample z path more to size ~2
        pre_shape = self._get_output_shape(self.pre_model, input_shape)
        for i in range(int(np.floor(np.log(pre_shape[1] / 2) / np.log(2)))):
            conv_z_model += [nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3,
                                  stride=2, padding=1),
                             norm_layer(ngf * mult, affine=True),
                             nn.ReLU(True)]

        self.conv_z_model = nn.Sequential(*conv_z_model)

        z_path_shape = self._get_output_shape(self.conv_z_model, pre_shape)
        z_path_size = self.linear_z_input_size = int(np.prod(z_path_shape))
        # FC linear z path
        for i in range(2):
            next_z_path_size = max(latent_z, int(np.ceil(z_path_size / 2)))
            linear_z_model += [nn.Linear(z_path_size, next_z_path_size),
                           nn.BatchNorm1d(next_z_path_size, affine=True),
                           nn.ReLU(True)]
            z_path_size = next_z_path_size

        linear_z_model += [nn.Linear(z_path_size, latent_z)]
        self.linear_z_model = nn.Sequential(*linear_z_model)


    def _get_output_shape(self, model, input_shape):
        input = Variable(torch.zeros(1, *input_shape))
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            output =  nn.parallel.data_parallel(model, input, self.gpu_ids)
        else:
            output = model(input)
        shape = output.data.size()[1:]
        del input
        del output
        return shape

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            pre = nn.parallel.data_parallel(self.pre_model, input, self.gpu_ids)
            latent = nn.parallel.data_parallel(self.latent_model, pre, self.gpu_ids)
            conv_z = nn.parallel.data_parallel(self.conv_z_model, pre, self.gpu_ids)
            linear_z_input = conv_z.view(conv_z.size(0), self.linear_z_input_size)
            z = nn.parallel.data_parallel(self.linear_z_model, linear_z_input, self.gpu_ids)
        else:
            pre = self.pre_model(input)
            latent = self.latent_model(pre)
            conv_z = self.conv_z_model(pre)
            linear_z_input = conv_z.view(conv_z.size(0), self.linear_z_input_size)
            z = self.linear_z_model(linear_z_input)
        return latent, z


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # currently support only input_nc == output_nc
        assert(input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost

        downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([self.model(x), x], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


class NLayerDiscriminator1d(nn.Module):
    def __init__(self, input_size, n_layers=3, use_sigmoid=False, gpu_ids=[]):
        super().__init__()
        self.gpu_ids = gpu_ids

        sequence = [
            nn.Linear(input_size, input_size * 2),
            nn.BatchNorm1d(input_size * 2, affine=True),
            nn.LeakyReLU(0.2, True),
        ]

        input_size = input_size * 2

        for n in range(1, n_layers):
            new_input_size = max(12, int(np.ceil(input_size / 2)))
            sequence += [
                nn.Linear(input_size, new_input_size),
                nn.BatchNorm1d(new_input_size, affine=True),
                nn.LeakyReLU(0.2, True),
            ]
            input_size = new_input_size

        new_input_size = max(6, int(np.ceil(input_size / 2)))
        sequence += [
            nn.Linear(input_size, new_input_size),
            nn.BatchNorm1d(new_input_size, affine=True),
            nn.LeakyReLU(0.2, True),
        ]
        input_size = new_input_size

        sequence += [nn.Linear(input_size, 1)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids)  and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)
