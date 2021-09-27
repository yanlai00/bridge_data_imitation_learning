def get_dataset_class(name):
    if name == 'FixLenVideoDataset':
        from semiparametrictransfer.data_sets.data_loader import FixLenVideoDataset
        return FixLenVideoDataset
    if name == 'MultiDatasetLoader':
        from semiparametrictransfer.data_sets.multi_dataset_loader import MultiDatasetLoader
        return MultiDatasetLoader
