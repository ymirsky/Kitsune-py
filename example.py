from Kitsune import Kitsune
import numpy as np
import time
import csv
import pickle

##############################################################################
# Kitsune a lightweight online network intrusion detection system based on an ensemble of autoencoders (kitNET).
# For more information and citation, please see our NDSS'18 paper: Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection

# This script demonstrates Kitsune's ability to incrementally learn, and detect anomalies in recorded a pcap of the Mirai Malware.
# The demo involves an m-by-n dataset with n=115 dimensions (features), and m=100,000 observations.
# Each observation is a snapshot of the network's state in terms of incremental damped statistics (see the NDSS paper for more details)

#The runtimes presented in the paper, are based on the C++ implimentation (roughly 100x faster than the python implimentation)
###################  Last Tested with Anaconda 3.6.3   #######################

# Load Mirai pcap (a recording of the Mirai botnet malware being activated)
# The first 70,000 observations are clean...
# print("Unzipping Sample Capture...")
# import zipfile
# with zipfile.ZipFile("mirai.zip","r") as zip_ref:
#     zip_ref.extractall()

from parse_args import * 

args = parse_args()
dataset = args.dataset
desc = args.job_description


# File location
path = f"{dataset}_pcap.pcap" #the pcap, pcapng, or tsv file to process.
packet_limit = np.Inf #the number of packets to process

# Get labels
labels = f"{dataset}_labels.csv"  #the labels for the pcap packet data
with open(labels, 'r') as f:
    reader = csv.reader(f)
    labels_list = list(reader)
labels_list = [int(sublist[-1]) for sublist in labels_list] #flatten list of labels
if dataset != 'mirai':
    if len(labels_list[0]) == 2:
        for i in range(len(labels_list)):
            labels_list[i] = labels_list[i][1]
    if '0' not in labels_list[0]: 
        labels_list.pop(0)
    benign_packets = 0
    for i in range(len(labels_list)):
        if labels_list[i] == '0':
            benign_packets += 1
        else:
            break
else:
    benign_packets = 0
    for i in range(len(labels_list)):
        if int(labels_list[i]) == 0:
            benign_packets += 1
        else:
            break

# KitNET params:
# maxAE = 10 #maximum size for any autoencoder in the ensemble layer
# FMgrace = 10000 #the number of instances taken to learn the feature mapping (the ensemble's architecture)
# ADgrace = 100000 #the number of instances used to train the anomaly detector (ensemble itself)
maxAE = 10
FMgrace = int(0.2*benign_packets)
ADgrace = int(0.6*benign_packets)

# Build Kitsune
K = Kitsune(path,packet_limit,maxAE,FMgrace,ADgrace)

print("Running Kitsune:")
RMSEs = []
i = 0
start = time.time()

normal_rmses = []
normal_indices = []
anomaly_rmses = []
anomaly_indices = []

# Here we process (train/execute) each individual packet.
# In this way, each observation is discarded after performing process() method.
all_vec = []
while True:
    i+=1
    if i % 1000 == 0:
        print("packet ", i, " rmse ", RMSEs[-1])
    rmse, vec = K.proc_next_packet()
    if vec is not None:
        all_vec.append(vec)
    if rmse == -1:
        break
    if labels_list[i-1] == 0:
        normal_rmses.append(rmse)
        normal_indices.append(i)
    else:
        anomaly_rmses.append(rmse)
        anomaly_indices.append(i)
    RMSEs.append(rmse)
stop = time.time()
print("Complete. Time elapsed: "+ str(stop - start))

normal_rmses = np.array(normal_rmses)
normal_indices = np.array(normal_indices)
anomaly_rmses = np.array(anomaly_rmses)
anomaly_indices = np.array(anomaly_indices)
all_vec = np.array(all_vec)
print("Normal rmse mean: ", np.mean(normal_rmses))
print("Normal rmse std: ", np.std(normal_rmses))
print("Normal rmses: ", np.sort(normal_rmses))
print("Anomaly rmse mean: ", np.mean(anomaly_rmses))
print("Anomaly rmse std: ", np.std(anomaly_rmses))
print("Anomaly rmses: ", np.sort(anomaly_rmses))
np.save(f"results/{dataset}_{desc}_anomaly_rmses.npy", anomaly_rmses)
np.save(f"results/{dataset}_{desc}_anomaly_indices.npy", anomaly_indices)
np.savetxt(f"results/{dataset}_{desc}_all_vec.csv", all_vec)
# with open(f"results/{dataset}_{desc}_all_vec.pickle", 'wb') as handle:
#     pickle.dump(all_vec, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('All vectors saved, shape', all_vec.shape)

# Here we demonstrate how one can fit the RMSE scores to a log-normal distribution (useful for finding/setting a cutoff threshold \phi)
from scipy.stats import norm
benignSample = np.log(RMSEs[FMgrace+ADgrace+1:100000])
logProbs = norm.logsf(np.log(RMSEs), np.mean(benignSample), np.std(benignSample))

# plot the RMSE anomaly scores
print("Plotting results")
from matplotlib import pyplot as plt
from matplotlib import cm
plt.figure(figsize=(10,5))
fig = plt.scatter(range(FMgrace+ADgrace+1,len(RMSEs)),RMSEs[FMgrace+ADgrace+1:],s=0.1,c=logProbs[FMgrace+ADgrace+1:],cmap='RdYlGn')
plt.yscale("log")
plt.title("Anomaly Scores from Kitsune's Execution Phase")
plt.ylabel("RMSE (log scaled)")
plt.xlabel("Time elapsed [min]")
figbar=plt.colorbar()
figbar.ax.set_ylabel('Log Probability\n ', rotation=270)
plt.savefig(f'results/{dataset}_{desc}_rmse_plot.png')

