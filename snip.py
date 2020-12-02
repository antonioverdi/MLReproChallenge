def snip_prune(model, batch, labels):
	masks = create_masks(model)

	


def create_masks(model):
    masks = []
    for layer in model:
        if isinstance(layer, torch.nn.Linear) or isinstance(layer, torch.nn.Conv2d):
            masks.append(torch.ones_like(layer.weight))
    return masks