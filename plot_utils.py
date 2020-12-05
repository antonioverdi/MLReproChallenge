import json
import matplotlib.pyplot as plt
import numpy as np

def create_plots(filepath, compression_list):
	with open(filepath, 'r') as logs:
		data = json.load(logs)
		int_compressions = list(map(_to_int, compression_list))
		x_ticks = list(map(_add_percent, int_compressions))

		
		for i,k in enumerate(data.keys()):
			accuracies = []
			for compression in compression_list:
				accuracies.append(data[k]['compression' + compression]['accuracy'])
			x = np.arange(len(accuracies))
			change = _get_acc_change(accuracies)
			plt.figure(figsize=(16,4))
			plt.suptitle(k + " Compression")
			plt.subplot(121)
			plt.ylim(0,100)
			plt.plot(x, accuracies)
			plt.xticks(x, x_ticks)
			plt.xlabel("Percent of Weights Retained")
			plt.ylabel("Accuracy on Test Set")
			plt.subplot(122)
			plt.grid(True)
			plt.plot(x, change, 'ob--')
			plt.xticks(x, x_ticks)
			plt.ylim(-3,1)
			plt.yticks([-3, -2, -1, 0, 1], ["-3%", "-2%", "-1%", "0%", "1%"])
			plt.xlabel("Percent of Weights Retained")
			plt.ylabel("Change in Accuracy")
			

def _to_int(n):
    return int(n)

def _add_percent(n):
	return str(n) + "%"

def _get_acc_change(accuracies):
	unpruned_acc = accuracies[0]
	acc_change = [x - unpruned_acc for x in accuracies]
	return acc_change