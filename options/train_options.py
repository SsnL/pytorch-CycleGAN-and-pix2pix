from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at specified learning rate')
        self.parser.add_argument('--niter_warmup', type=int, default=0, help='# of iter to warmup (lienarly increasing learning rate)')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--initial_lr', type=float, default=0.0, help='initial learning rate for adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='learning rate after warmup for adam')
        self.parser.add_argument('--lambdas', nargs='+', type=float, default=[10,10], help='[multi_cycle_gan, multi_cycle_gan_hub] cycle loss weights from reconstruction for each dataset')
        self.parser.add_argument('--non_hub_multiplier', type=float, default=0.5, help='[multi_cycle_gan_hub] multiplier for cycle loss from reconstruction from non-hub vertices')
        self.parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.isTrain = True
