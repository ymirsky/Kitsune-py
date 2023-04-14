import numpy as np
import matplotlib.pyplot as plt
import csv 

dataset = "Fuzzing_dataset.csv"
threshold = 0.4  # anomaly threshold
col = 3  # column index

# Get list of all values in column i of the dataset 
def get_column(i):
    return [float(row[i]) for row in data_list]

if __name__ == "__main__":
    # Load data.
    anomalous_indices = np.load("fuzz_anomaly_indices.npy")
    anomalous_rmses = np.load("fuzz_anomaly_rmses.npy")

    with open(dataset, 'r') as f:
        reader = csv.reader(f)
        data_list = list(reader)

    # Split indices into those with rmse less than and greater than threshold
    column = get_column(col)
    column_less = []
    column_greater = []
    for i in range(anomalous_indices.shape[0]):
        if anomalous_rmses[i] < threshold:
            column_less.append(column[anomalous_indices[i] - 1])
        else:
            column_greater.append(column[anomalous_indices[i] - 1])

    max_i = max(np.amax(column_less), np.amax(column_greater))
    x_max = max_i
    plt.figure(figsize=(8,5))
    plt.xlim(0, x_max)

    bin_list = [x_max/200.0 * i for i in range(201)]

    n, bins, patches = plt.hist(column_greater, bins=bin_list, facecolor='g', label="malicious packets w/ rmse > " + str(threshold), log=True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    for c, p in zip(bin_centers, patches):
        plt.setp(p, color='#a32632', alpha=0.7)

    n, bins, patches = plt.hist(column_less, bins=bin_list, facecolor='g', label="malicious packets w/ rmse < " + str(threshold), log=True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    for c, p in zip(bin_centers, patches):
        plt.setp(p, color='#305da1', alpha=0.7)

    plt.xlabel("Feature " + str(col), fontsize=15)
    plt.ylabel("Frequency", fontsize=15)
    plt.title("Fuzzing Feature " + str(col), fontsize=15)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(reversed(handles), reversed(labels), loc="upper right", fontsize=18)
    plt.show()
    #plt.savefig("fuzzing_feature.pdf", format="pdf", bbox_inches="tight")