def get_model_class(name):
    if name == 'GCBCImages':
        from bridgedata.models.gcbc_images import GCBCImages
        return GCBCImages
    if name == 'GCBCTransfer':
        from bridgedata.models.gcbc_transfer import GCBCTransfer
        return GCBCTransfer
    if name == 'GCBCImagesContext':
        from bridgedata.models.gcbc_images_context import GCBCImagesContext
        return GCBCImagesContext
    else:
        raise ValueError("modelname not found!")
