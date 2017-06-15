
def CreateDataLoader(opt):
    from data.custom_dataset_data_loader import CustomDatasetDataLoader
    data_loader = CustomDatasetDataLoader()
    # data_loader = None
    # if opt.align_data > 0:
    #     from data.aligned_data_loader import AlignedDataLoader
    #     data_loader = AlignedDataLoader()
    # else:
    #     if opt.model == 'multi_cycle_gan':
    #         from data.unaligned_multi_data_loader import UnalignedMultiDataLoader
    #         data_loader = UnalignedMultiDataLoader()
    #     else:
    #         from data.unaligned_data_loader import UnalignedDataLoader
    #         data_loader = UnalignedDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader
