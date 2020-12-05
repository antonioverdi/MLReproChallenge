import json
import matplotlib.pyplot as plt
import numpy as np

def create_plots(filepath, compression_list):
	with open(filepath, 'r') as logs:
		data = json.load(logs)
		int_compressions = list(map(_to_int, compression_list))

		
		for i,k in enumerate(data.keys()):
			accuracies = []
			for compression in compression_list:
				accuracies.append(data[k]['compression' + compression]['accuracy'])
			x = np.arange(len(accuracies))
			plt.figure(figsize=(8,5))
			plt.title(k + " Compression")
			plt.ylim(0,100)
			plt.plot(x, accuracies)
			plt.xticks(x, int_compressions)
			plt.xlabel("Percent of Weights Retained")
			plt.ylabel("Accuracy on Test Set")
			

def _to_int(n):
    return int(n)