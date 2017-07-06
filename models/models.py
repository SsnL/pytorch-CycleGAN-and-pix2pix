
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'cycle_gan':
        assert(opt.dataset_mode == 'unaligned')
        from .cycle_gan_model import CycleGANModel
        model = CycleGANModel()
    elif opt.model == 'dual_gan':
        from .dual_gan_model import DualGANModel
        #assert(opt.align_data == False)
        model = DualGANModel()
    elif opt.model == 'latent_cycle_gan':
        from .latent_cycle_gan_model import LatentCycleGANModel
        #assert(opt.align_data == False)
        model = LatentCycleGANModel()
    elif opt.model == 'multi_cycle_gan_hub':
        from .multi_cycle_gan_with_hub_model import MultiCycleGANWithHubModel
        #assert(opt.align_data == False)
        model = MultiCycleGANWithHubModel()
    elif opt.model == 'multi_cycle_gan':
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
