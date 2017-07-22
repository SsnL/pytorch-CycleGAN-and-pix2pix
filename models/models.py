
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'cycle_gan':
        assert(opt.dataset_mode == 'unaligned')
        from .cycle_gan_model import CycleGANModel
        model = CycleGANModel()
    elif opt.model == 'weighted_cycle_gan':
        assert(opt.dataset_mode == 'unaligned')
        from .weighted_cycle_gan_model import WeightedCycleGANModel
        model = WeightedCycleGANModel()
    elif opt.model == 'encoder_cycle_gan':
        assert(opt.dataset_mode == 'unaligned')
        from .encoder_cycle_gan_model import EncoderCycleGANModel
        model = EncoderCycleGANModel()
    elif opt.model == 'cycle_gan_z':
        assert(opt.dataset_mode == 'unaligned')
        from .cycle_gan_z_model import CycleGANZModel
        model = CycleGANZModel()
    elif opt.model == 'dual_gan':
        assert(opt.dataset_mode == 'unaligned')
        from .dual_gan_model import DualGANModel
        model = DualGANModel()
    elif opt.model == 'latent_cycle_gan':
        assert(opt.dataset_mode == 'unaligned')
        from .latent_cycle_gan_model import LatentCycleGANModel
        model = LatentCycleGANModel()
    elif opt.model == 'multi_cycle_gan_hub':
        assert(opt.dataset_mode == 'unaligned')
        from .multi_cycle_gan_with_hub_model import MultiCycleGANWithHubModel
        model = MultiCycleGANWithHubModel()
    elif opt.model == 'multi_cycle_gan':
        assert(opt.dataset_mode == 'unaligned')
        from .multi_cycle_gan_model import MultiCycleGANModel
        model = MultiCycleGANModel()
    elif opt.model == 'pix2pix':
        assert(opt.dataset_mode == 'aligned')
        from .pix2pix_model import Pix2PixModel
        model = Pix2PixModel()
    elif opt.model == 'test':
        assert(opt.dataset_mode == 'single')
        from .test_model import TestModel
        model = TestModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
