import csv
import math
from math import floor

import numpy as np
from scapy.utils import rdpcap
from random import sample

from KitPlugin import KitPlugin

inputs = {
    "mirai_malicious" : {
        "input_path" : "input_data/mirai.pcap",
        "input_path_test" : "input_data/mirai.pcap",
        "packet_limit" : 200000,
        "maxAE" : 10,
        "FMgrace" : 5000,
        "ADgrace" : 50000,
        "training_min" : 0,
        "training_max" : 60000,
        "testing_min" : 140330,
        "testing_max" : 140355
    },
    "mirai_benign": {
        "input_path" : "input_data/mirai.pcap",
        "input_path_test" : "input_data/mirai.pcap",
        "packet_limit": 200000,
        "maxAE": 10,
        "FMgrace": 5000,
        "ADgrace": 50000,
        "training_min": 0,
        "training_max": 60000,
        "testing_min": 70000,
        "testing_max": 70025
    }
}
#KitPlugin = KitPlugin()
#KitPlugin.hyper_opt("input_data/Monday-WorkingHours_10_percent_random.pcap", 100, 1000000)

# Run series of statistics
#KitPlugin.run_series_stats(inputs)
# Get feature list from pickle file
#KitPlugin.feature_loader()
# Train Kitsune on the training data
#KitPlugin.kit_trainer(training_min, training_max)
# Calculate SHAP-values
#KitPlugin.shap_values_builder(training_min, training_max, testing_min, testing_max)
# Pickle SHAP-values
#KitPlugin.shap_values_pickle()
# Load SHAP-values
#KitPlugin.shap_values_loader()
# Calculate shap summary statistics
#KitPlugin.shap_stats_summary_builder(testing_min, testing_max)
#KitPlugin.shap_stats_excel_export()

# Calculate EER and AUC values
#KitPlugin = KitPlugin(input_path="input_data/mirai.pcap", packet_limit=200000, num_autenc=10, FMgrace=5000, ADgrace=50000, learning_rate=0.1, hidden_ratio=0.75)
#KitPlugin.feature_loader()
#KitPlugin.kit_trainer(0, 60000)
#KitPlugin.model_pickle()
# ONLY run this on a mixed batch of benign/malicious samples
#RMSEs = KitPlugin.kit_runner(120000, 122000, normalize=True)
# Labels
# Create an array of zeros with 1622 entries
#zeros = np.zeros(1622)
# Create an array of ones with 378 entries
#ones = np.ones(378)
# Concatenate the two arrays to get the final array
#labels = np.concatenate((zeros, ones))
#KitPlugin.calc_auc_eer(RMSEs, labels)

#KitPlugin = KitPlugin()
# Random sample of 500000 packets, ordered by timestamp
#KitPlugin.random_sample_pcap("input_data/Monday-WorkingHours.pcap", "input_data/Monday-WorkingHours_500k.pcap", 500000)
#KitPlugin.random_sample_pcap("input_data/Monday-WorkingHours.pcap", "input_data/Monday-WorkingHours_1M.pcap", 1000000)

# Out of every 1000 packets, it only keeps the first 100; so, only 10% of packets is kept. It will then do the same for the next 1000 packets
#KitPlugin.interval_sample_pcap("input_data/Monday-WorkingHours.pcap", "input_data/Monday-WorkingHours_10_percent.pcap", 10)

# Sample 10 percent of conversations
#SampleKitPlugin = KitPlugin()
##conversations = SampleKitPlugin.sample_percentage_conversations(10, "input_data/Monday_Split/17_01-18_01.pcapng", "input_data/Monday_Split/17_01-18_01-sample-10.pcap")
#conversations = SampleKitPlugin.conversations_loader('pickles/17_01-18_01_sample_10_conv')
#packets = rdpcap('input_data/Monday_Split/17_01-18_01-sample-10.pcap')
#features = SampleKitPlugin.load_pcap_to_features('input_data/Monday_Split/17_01-18_01-sample-10.pcap')

#print('conversations: '+str(len(conversations)))
#print('packets: '+str(len(packets)))
#print('feature lists: '+str(len(features)))

#del SampleKitPlugin

#NewKitPlugin = KitPlugin('input_data/Monday_Split/17_01-18_01-sample-10.pcap', num_autenc=6, FMgrace=int(0.05*len(features)), int(0.95*len(ADgrace)))

# Train Kitsune
#NewKitPlugin.kit_trainer_supplied_features(features)

# Open TSV file and extract features
#KitPlugin = KitPlugin(input_path='input_data/Monday-WorkingHours.pcap.tsv', packet_limit=np.Inf, num_autenc=6, FMgrace=11000000, ADgrace=12000000, learning_rate=0.1, hidden_ratio=0.75)
#features = KitPlugin.feature_builder('input_data/features.csv')
#KitPlugin.feature_pickle()

KitPlugin = KitPlugin()
#labels = KitPlugin.read_label_file('input_data/monday_labels_cleaned.csv')
#iter = 0
#for label in labels:
#    iter += 1
#    label.append(str(labels.index(label)-1))
# We sample 10 percent of labels
#labels = sample(labels, int(0.1*len(labels)))

#with open('input_data/Monday-WorkingHours.pcap.tsv') as csvfile:
#    packetreader = csv.reader(csvfile)
#    counter = 0
#    for row in packetreader:
#        if row:
#            counter +=1
#        if counter % 10000 == 0:
#            print(counter)

#KitPlugin.sample_packets_by_conversation('input_data/Monday-WorkingHours.pcap.tsv', 'input_data/Monday-WorkingHours_nohash.pcap.tsv', labels)
# Map samples to features of an existing featureList
#KitPlugin.map_packets_to_features('input_data/Monday-WorkingHours_nohash.pcap.tsv', 'input_data/features.csv', 'input_data/sampled_features.csv')

# Total cutoff should be length of the sampled features file, training size is 0.7 times the total size
# 50 test runs
KitPlugin.hyper_opt_KitNET("input_data/sampled_features.csv", int(0.7*338105), 338105, 50)
