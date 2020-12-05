'''Testing CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchsummary import summary
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torchvision
import torchvision.transforms as transforms

import os
import json
import argparse

import resnet

parser = argparse.ArgumentParser(description='ResNet56 pruning experiment testing properties')
parser.add_argument('--model_dir', type=str,  default='trained_models', help='directory of trained models. Should all be of same model type')
parser.add_argument('--arch', type=str, default="resnet56", help="model type to load pretrained weights into")
parser.add_argument('--log_dir', type=str, default="accuracy_logs.json", help='directory to save accuracy logs from pretrained models')
args = parser.parse_args()

def main():
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	print(device)

	transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

	test_loader = torch.utils.data.DataLoader(
		datasets.CIFAR10(root='./data', train=False, transform=transform_test, download=True),
		batch_size=128, shuffle=False,
		num_workers=0)

	
	##Not sure why this bit of code didn't create a ResNet model but no time examine. Let the hardcoding begin :D
	# model = None
	# if args.model_dir == "resnet56":
	# 	model = resnet.resnet56()
	model = resnet.resnet56()
	model = model.to(device)

	#save files need the format <pruning style><compression rate in 3 numbers>.pth for example SNIP010.pth for SNIP style pruning to 10% weight retention
	model_names = []
	with os.scandir(args.model_dir) as folder:
		for file in folder:
			model_names = file.name[:-3]

	#collect accuracies
	accuracies = []
	for prune_style in model_names:
		
		filename = args.model_dir + os.sep + prune_style + ".th"
		print("Testing Model: {} from {}".format(model, filename))
		pretrained = torch.load(filename)
		model.load_state_dict(pretrained, strict=False)
		model.eval()
		test_loss = 0
		correct = 0
		total = 0
		with torch.no_grad():
			for batch_idx, (inputs, targets) in enumerate(testloader):
				inputs, targets = inputs.to(device), targets.to(device)
				outputs = net(inputs)
				criterion = nn.CrossEntropyLoss()
				loss = criterion(outputs, targets)

				test_loss += loss.item()
				_, predicted = outputs.max(1)
				total += targets.size(0) 
				correct += predicted.eq(targets).sum().item()
		
		acc = 100.*correct/total
		accuracies.append(acc)
		print("    Accuracy: {}%".format(acc))

	#write to json file.
	output_json = {}
	for i,model_name in enumarate(model_names):
		prune_style = model_name[:-3]
		if not (prune_style in output_json):
			output_json[prune_style] = {}
		output_json[prune_style]['compression' + model_name[-3:]] = {'accuracy': accuracies[i]}

	with open(args.log_dir, 'w') as output:
		json.dump(output_json, output, indent=1)


if __name__ == '__main__':
	main()