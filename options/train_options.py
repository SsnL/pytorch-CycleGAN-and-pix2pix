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
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--cycle_feature_level', type=int, default=2, help='feature level of D used for cycle consistency, 0 is pixel level')
        self.parser.add_argument('--weight_cycle_niter', type=int, default = 0, help='weight cycle loss using discriminator output for certain iters at beginning')
        self.parser.add_argument('--lambda_A', type=float, default=10, help='weight for cycle loss in pixel space (A -> B -> A)')
        self.parser.add_argument('--lambda_B', type=float, default=10, help='weight for cycle loss in pixel space (B -> A -> B)')
        # self.parser.add_argument('--lambda_feature_A', type=float, default=10.0, help='weight for cycle loss in feature space (A -> B -> A)')
        # self.parser.add_argument('--lambda_feature_B', type=float, default=10.0, help='weight for cycle loss in feature space (B -> A -> B)')
        self.parser.add_argument('--feature_shift_range', nargs='+', type=int, default=[1, 201], help='iter range to linearly shift cycle consistency from pixel space to feature space')
        self.parser.add_argument('--feature_shift_to', type=float, default=0, help='value to which cycle consistency linearly shift from pixel space to feature space, <=1, 0 means no shift')
        self.parser.add_argument('--non_gan_decay_range', nargs='+', type=int, default=[101, 251], help='iter range to linearly decay cycle and identity loss to cycle_identity_decay_to')
        self.parser.add_argument('--non_gan_decay_to', type=float, default=1, help='value to which cycle and identity loss linearly decay, >=0, 1 means no decay')
        self.parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--stochasticity_replicate', type=int, default=1, help='replicate each data to ensure stochasticity output, only used when stochasticity is true')
        self.parser.add_argument('--no_flip'  , action='store_true', help='if specified, do not flip the images for data argumentation')

        # NOT-IMPLEMENTED self.parser.add_argument('--preprocessing', type=str, default='resize_and_crop', help='resizing/cropping strategy')
        self.isTrain = True
