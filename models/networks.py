import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
###############################################################################
# Functions
###############################################################################


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1 or  classname.find('InstanceNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type):
    if norm_type == 'batch':
        norm_layer = nn.BatchNorm2d
    elif norm_type == 'instance':
        norm_layer = nn.InstanceNorm2d
    else:
        print('normalization layer [%s] is not found' % norm)
    return norm_layer

def define_G(input_nc, output_nc, latent_resnet_block, ngf, which_model_netG, norm='batch', use_dropout=False, gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    if latent_resnet_block >= 0:
        assert which_model_netG.find('resnet') != -1

    if which_model_netG == 'resnet_9blocks':
        if latent_resnet_block >= 0:
            # netG = ResnetLatentGenerator(input_nc, output_nc, ngf, norm_layer, use_dropout=use_dropout, n_blocks=9, latent_resnet_block=latent_resnet_block, gpu_ids=gpu_ids)
            netG = LatentNet(input_nc, output_nc, norm_layer, gpu_ids=gpu_ids)
        else:
            netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_6blocks':
        if latent_resnet_block >= 0:
            # netG = ResnetLatentGenerator(input_nc, output_nc, ngf, norm_layer, use_dropout=use_dropout, n_blocks=6, latent_resnet_block=latent_resnet_block, gpu_ids=gpu_ids)
            netG = LatentNet(input_nc, output_nc, norm_layer, gpu_ids=gpu_ids)
        else:
            netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    else:
        print('Generator model name [%s] is not recognized' % which_model_netG)
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
        print('Discriminator model name [%s] is not recognized' %
              which_model_netD)
    if use_gpu:
        netD.cuda(device_id=gpu_ids[0])
    netD.apply(weights_init)
    return netD


def define_cycle_featurizer(netD, k = 1, gpu_ids=[]):
    if k == 0:
        return nn.Identity()
    else:
        return KthLayerDiscriminatorFeatureExtractor(netD, k, gpu_ids)


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
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class LatentNet(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, gpu_ids=[]):
        super(LatentNet, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.gpu_ids = gpu_ids

        n_layers = 2
        # resnet_before_layer = 1
        n_resblock_each_side = 4


        ngf = 64
        kw = 4
        padw = int(np.floor((kw-1)/2))
        to_latent_sequence = [
            nn.Conv2d(input_nc, ngf, kernel_size=kw, padding=padw),
            norm_layer(ngf, affine=True),
            nn.ReLU(True),
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers + 1):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            to_latent_sequence += [
                nn.Conv2d(ngf * nf_mult_prev, ngf * nf_mult,
                                kernel_size=kw, stride=2, padding=padw),
                # TODO: use InstanceNorm
                norm_layer(ngf * nf_mult, affine=True),
                nn.ReLU(True)
            ]

        for _ in range(n_resblock_each_side):
            to_latent_sequence += [
                ResnetBlock(ngf * nf_mult, 'zero', norm_layer=norm_layer, use_dropout=False)
            ]

        to_latent_sequence += [
            nn.Conv2d(ngf * nf_mult, int(np.ceil(ngf * nf_mult / 2)), kernel_size=kw, padding=padw, stride = 2),
            norm_layer(int(np.ceil(ngf * nf_mult / 2)), affine=True),
            nn.ReLU(True)
        ]
        to_latent_sequence += [
            nn.Conv2d(int(np.ceil(ngf * nf_mult / 2)), int(np.ceil(ngf * nf_mult / 4)), kernel_size=3, stride = 3),
            nn.ReLU(True)
        ]


        from_latent_sequence = []
        for module in to_latent_sequence[3:][::-1]:
            if module.__class__.__name__.find('Conv2d') != -1:
                if type(module.stride) == tuple:
                    output_padding = tuple(s - 1 for s in module.stride)
                else:
                    output_padding = module.stride - 1
                from_latent_sequence += [
                    nn.ConvTranspose2d(module.out_channels, module.in_channels,
                                    kernel_size=module.kernel_size, stride=module.stride, padding=module.padding, output_padding=output_padding),
                    # TODO: use InstanceNorm
                    norm_layer(module.in_channels, affine=True),
                    nn.ReLU(True)
                ]
            elif module.__class__.__name__.find('ResnetBlock') != -1:
                from_latent_sequence += [
                    ResnetBlock(module.dim, 'zero', norm_layer=norm_layer, use_dropout=False)
                ]

        from_latent_sequence += [nn.Conv2d(ngf, output_nc, kernel_size=kw, padding=2)]
        from_latent_sequence += [nn.Tanh()]

        self.model = nn.Sequential(*(to_latent_sequence + from_latent_sequence))
        self.to_latent_model = nn.Sequential(*to_latent_sequence)
        self.from_latent_model = nn.Sequential(*from_latent_sequence)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

    def forward_to_latent(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.to_latent_model, input, self.gpu_ids)
        else:
            return self.to_latent_model(input)

    def forward_from_latent(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.from_latent_model, input, self.gpu_ids)
        else:
            return self.from_latent_model(input)

    def forward_latent(self, input):
        latent = self.forward_to_latent(input)
        print(latent.size())
        return latent, self.forward_from_latent(latent)


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[]):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids

        model = [nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3),
                 norm_layer(ngf, affine=True),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1),
                      norm_layer(ngf * mult * 2, affine=True),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, 'zero', norm_layer=norm_layer, use_dropout=use_dropout)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1),
                      norm_layer(int(ngf * mult / 2), affine=True),
                      nn.ReLU(True)]

        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

class ResnetLatentGenerator(ResnetGenerator):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, latent_resnet_block=3, gpu_ids=[]):
        super(ResnetLatentGenerator, self).__init__(input_nc, output_nc, ngf, norm_layer, use_dropout, n_blocks, gpu_ids)

        modules = list(self.model)

        i = 0
        while latent_resnet_block >= 0:
            if modules[i].__class__.__name__.find('ResnetBlock') != -1:
                latent_resnet_block -= 1
            i += 1

        self.latent = None
        # modules[i - 1].register_forward_hook(lambda m, i, o: self.__setattr__('latent', o))
        self.to_latent_model = nn.Sequential(*modules[:i])
        self.from_latent_model = nn.Sequential(*modules[i:])

    def forward_to_latent(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.to_latent_model, input, self.gpu_ids)
        else:
            return self.to_latent_model(input)

    def forward_from_latent(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.from_latent_model, input, self.gpu_ids)
        else:
            return self.from_latent_model(input)

    def forward_latent(self, input):
        latent = self.forward_to_latent(input)
        return latent, self.forward_from_latent(latent)

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout):
        super(ResnetBlock, self).__init__()
        self.dim = dim
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout):
        conv_block = []
        p = 0
        # TODO: support padding types
        assert(padding_type == 'zero')
        p = 1

        # TODO: InstanceNorm
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim, affine=True),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim, affine=True)]

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
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if  self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
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
        downnorm = norm_layer(inner_nc, affine=True)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc, affine=True)

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
                # TODO: use InstanceNorm
                norm_layer(ndf * nf_mult, affine=True),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                            kernel_size=kw, stride=1, padding=padw),
            # TODO: use InstanceNorm
            norm_layer(ndf * nf_mult, affine=True),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids)  and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

# Defines the PatchGAN discriminator with the specified arguments.
class KthLayerDiscriminatorFeatureExtractor(nn.Module):
    def __init__(self, netD, k = 1, gpu_ids = []):
        super(KthLayerDiscriminatorFeatureExtractor, self).__init__()
        self.gpu_ids = gpu_ids

        D_modules = netD.model

        modules = []
        i = 0
        while True:
            if D_modules[i].__class__.__name__.find('Conv') != -1:
                k -= 1
                if k < 0:
                    break
            modules.append(D_modules[i])
            i += 1

        self.model = nn.Sequential(*modules)

    def forward(self, input):
        if len(self.gpu_ids)  and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

