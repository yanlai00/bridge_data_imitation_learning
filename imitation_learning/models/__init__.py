def get_model_class(name):
    if name == 'GCBCImages':
        from imitation_learning.models.gcbc_images import GCBCImages
        return GCBCImages
    if name == 'GCBCTransfer':
        from imitation_learning.models.gcbc_transfer import GCBCTransfer
        return GCBCTransfer
    if name == 'GCBCImagesContext':
        from imitation_learning.models.gcbc_images_context import GCBCImagesContext
        return GCBCImagesContext
    else:
        raise ValueError("modelname not found!")
