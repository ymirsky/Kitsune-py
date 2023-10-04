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
#KitPlugin = KitPlugin(input_path='input_data/Wednesday-WorkingHours.pcap', packet_limit=np.Inf, num_autenc=6, FMgrace=11000000, ADgrace=12000000, learning_rate=0.1, hidden_ratio=0.75)
#features = KitPlugin.feature_builder('input_data/wednesday_features.csv')
#KitPlugin.feature_pickle()

#KitPlugin = KitPlugin()
#print('reading labels file')
#labels = KitPlugin.read_label_file('input_data/tuesday_labels_cleaned.csv')
#iter = 0
#for label in labels:
#    iter += 1
#    if iter % 10000 == 0:
#        print(iter)
#    label.append(str(labels.index(label)-1))
#We sample all malicious labels
#print(len(labels))
#train_labels = [lst for lst in labels if len(lst) >= 5 and lst[4] == 'BENIGN']
#test_labels = [lst for lst in labels if len(lst) >= 5 and lst[4] != 'BENIGN']

#print('training set: '+str(len(train_labels)))
#print('training set: '+str(len(test_labels)))
# Sample 10 percent of the training set
#train_labels = sample(train_labels, int(0.1*len(train_labels)))

#with open('input_data/Monday-WorkingHours.pcap.tsv') as csvfile:
#    packetreader = csv.reader(csvfile)
#    counter = 0
#    for row in packetreader:
#        if row:
#            counter +=1
#        if counter % 10000 == 0:
#            print(counter)

#print('sampling training set')
#KitPlugin.sample_packets_by_conversation('input_data/Tuesday-WorkingHours.pcap.tsv', 'input_data/Tuesday-WorkingHours_benign.pcap.tsv', train_labels)
#print('sampling testing set')
#KitPlugin.sample_packets_by_conversation('input_data/Tuesday-WorkingHours.pcap.tsv', 'input_data/Tuesday-WorkingHours_malicious.pcap.tsv', test_labels)

# Map samples to features of an existing featureList
KitPlugin = KitPlugin()
#print('mapping training set')
#KitPlugin.map_packets_to_features('input_data/Tuesday-WorkingHours_benign.pcap.tsv', 'input_data/tuesday_features.csv', 'input_data/sampled_tuesday_features_benign.csv')
#print('mapping testing set')
#KitPlugin.map_packets_to_features('input_data/Tuesday-WorkingHours_malicious.pcap.tsv', 'input_data/tuesday_features.csv', 'input_data/sampled_tuesday_features_malicious.csv')

# Total cutoff should be length of the sampled features file, training size is 0.7 times the total size
# 50 test runs
#KitPlugin.hyper_opt_KitNET("input_data/sampled_tuesday_features_benign.csv", int(0.7*338105), 338105, 50)

#KitPlugin = KitPlugin()
#KitPlugin.hyper_opt("input_data/Monday-WorkingHours_10_percent_random.pcap", 100, 1000000)

#KitPlugin = KitPlugin()
#shap_values = KitPlugin.shap_values_builder_from_csv('input_data/sampled_features.csv', floor(0.9*395789), 395789, 7, 0.132533, 0.509961)
#shap_values = KitPlugin.shap_values_builder_separate_train_test_csv('input_data/sampled_features_monday.csv', 'input_data/sampled_tuesday_features_malicious.csv', 395789, 891634, 6, 0.1312, 0.5050)
#KitPlugin.shap_values_pickle()
#KitPlugin.shap_stats_excel_export("output_data/shap_report_malicious.xlsx")
