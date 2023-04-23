import numpy as np
import matplotlib.pyplot as plt

from parse_args import * 

if __name__ == "__main__":

    args = parse_args()
    dataset = args.dataset
    desc = args.job_description
    
    # Load rmses (get rmses from running example.py)
    normal_rmses = np.load(f"results/{dataset}_{desc}_normal_rmses.npy")
    anomalous_rmses = np.load(f"results/{dataset}_{desc}_anomaly_rmses.npy")

    # Count number of malicious packets with rmse less than threshold
    threshold = args.threshold
    count = 0 
    for i in range(anomalous_rmses.shape[0]):
        if anomalous_rmses[i] < threshold:
            count += 1
    print("Count: {}".format(count))

    # Plot histogram.
    plt.figure(figsize=(11,5))
    max_rmse = max(np.amax(normal_rmses), np.amax(anomalous_rmses))
    print("Max rmse: {}".format(max_rmse))
    x_max = 20  # MODIFY depending on dataset
    plt.xlim(0, x_max)
    bin_list = [x_max/200.0 * i for i in range(201)]

    n, bins, patches = plt.hist(anomalous_rmses, bins=bin_list, facecolor='g', label="rmse of anomalous packets", log=True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    for c, p in zip(bin_centers, patches):
        plt.setp(p, color='#a32632', alpha=0.7)

    n, bins, patches = plt.hist(normal_rmses, bins=bin_list, facecolor='g', label="rmse of normal packets", log=True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    for c, p in zip(bin_centers, patches):
        plt.setp(p, color='#1f9156', alpha=0.7)

    plt.xlabel("RMSE", fontsize=15)
    plt.ylabel("Frequency", fontsize=15)
    plt.title("Attack RMSEs", fontsize=15)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(reversed(handles), reversed(labels), loc="upper right", fontsize=18)
    plt.savefig(f"results/{dataset}_{desc}_rmse.pdf", format="pdf", bbox_inches="tight")
