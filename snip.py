import torch
import torch.nn as nn
import torch.nn.functional as F
import types

def snip_forward_conv2d(self, x):
		return F.conv2d(x, self.weight * self.weight_mask, self.bias,
						self.stride, self.padding, self.dilation, self.groups)

def snip_forward_linear(self, x):
		return F.linear(x, self.weight * self.weight_mask, self.bias)

def snip_mask(model, batch, labels, compression):
	
	for layer in model.modules():
		#create pruning masks manually
		if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
			layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight)) 
#             layer.weight.requires_grad = False #computing gradient of mask not weights

		#monkey-patch forward methods
		if isinstance(layer, nn.Conv2d):
			layer.forward = types.MethodType(snip_forward_conv2d, layer)
			
		if isinstance(layer, nn.Linear):
			layer.forward = types.MethodType(snip_forward_linear, layer)
			
	#compute gradients of weight_mask (connections)
	model.zero_grad()
	out = model.forward(batch)
	loss_fn = nn.CrossEntropyLoss().cuda()
	loss = loss_fn(out, labels)
	loss.backward()
	
	absolute_saliency = []
	for layer in model.modules():
		if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
			absolute_saliency.append(torch.abs(layer.weight_mask.grad))
			
	saliency_scores = torch.cat([torch.flatten(x) for x in absolute_saliency])
	denominator = torch.sum(saliency_scores)
	saliency_scores.div_(denominator)
	
	kappa = int(len(saliency_scores) * compression)
	sorted_scores, indices = torch.topk(saliency_scores, kappa, sorted=True)
	threshold = sorted_scores[-1]
	
	connection_masks = []
	for c in absolute_saliency:
		connection_masks.append(((c / denominator) >= threshold).float())
	
	return connection_masks

def apply_snip(model, connection_masks):
	prunable_layers = filter(lambda layer: isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear), model.modules())

	for layer, mask in zip(prunable_layers, connection_masks):

		def hook_factory(keep_mask):
			"""
			The hook function can't be defined directly here because of Python's
			late binding which would result in all hooks getting the very last
			mask! Getting it through another function forces early binding.
			source: https://github.com/mil-ad/snip/blob/master/train.py
			"""
			def hook(grads):
				return grads * keep_mask

			return hook

		# Set the masked weights to zero (biases are ignored)
		layer.weight.data[mask == 0.] = 0.
		# Make sure their gradients remain zero. Register_hook gets called whenever a gradient is collected
		layer.weight.register_hook(hook_factory(mask))