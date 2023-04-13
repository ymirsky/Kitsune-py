import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Load data.
    normal_indices = np.load("fuzz_normal_indices.npy")
    anomalous_indices = np.load("fuzz_anomaly_indices.npy")
    normal_rmses = np.load("fuzz_normal_rmses.npy")
    anomalous_rmses = np.load("fuzz_anomaly_rmses.npy")
    count = 0 
    for i in range(anomalous_rmses.shape[0]):
        if anomalous_rmses[i] < 0.6:
            #print("Anomalous index: {}".format(anomalous_indices[i]))
            count += 1
    print("Count: {}".format(count))
    print(normal_rmses.shape[0])

    # for threshold in range(150,250):
    #     correct_under_thershold = (correct < threshold).sum()
    #     incorrect_under_thershold = (incorrect < threshold).sum()
    #     accuracy = correct_under_thershold/(correct_under_thershold + incorrect_under_thershold)
    #     coverage = (correct_under_thershold + incorrect_under_thershold)/(correct.size + incorrect.size)
    #     print("Threshold: {}, Accuracy: {}, Coverage: {}".format(threshold, accuracy, coverage))

    # max_rmse = max(np.amax(normal_rmses), np.amax(anomalous_rmses))
    # print("Max rmse: {}".format(max_rmse))

    # Plot histogram.
    plt.figure(figsize=(11,5))
    x_max = 15
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
    plt.title("Fuzzing attack RMSEs", fontsize=15)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(reversed(handles), reversed(labels), loc="upper right", fontsize=18)
    plt.savefig("fuzz_rmse.pdf", format="pdf", bbox_inches="tight")
