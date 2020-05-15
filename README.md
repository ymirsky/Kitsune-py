# Overview
In this repository you will find a Python implementation of Kitsune; an online network intrusion detection system, based on an ensemble of autoencoders. From,

*Yisroel Mirsky, Tomer Doitshman, Yuval Elovici, and Asaf Shabtai, "Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection", Network and Distributed System Security Symposium 2018 (NDSS'18)*

# What is Kitsune?

Neural networks have become an increasingly popular solution for network intrusion detection systems (NIDS). Their capability of learning complex patterns and behaviors make them a suitable solution for differentiating between normal traffic and network attacks. However, a drawback of neural networks is the amount of resources needed to train them. Many network gateways and routers devices, which could potentially host an NIDS, simply do not have the memory or processing power to train and sometimes even execute such models. More importantly, the existing neural network solutions are trained in a supervised manner. Meaning that an expert must label the network traffic and update the model manually from time to time.

Kitsune is a novel ANN-based NIDS which is online, unsupervised, and efficient. A Kitsune, in Japanese folklore, is a mythical fox-like creature that has a number of tails, can mimic different forms, and whose strength increases with experience. Similarly, Kitsune has an ensemble of small neural networks (autoencoders), which are trained to mimic (reconstruct) network traffic patterns, and whose performance incrementally improves overtime. 
	
The architecture of Kitsune is illustrated in the figure below:
* First, a feature extraction framework called *AfterImage* efficiently tracks the patterns of every network channel using damped incremental statisitcs, and extracts a feature vector for each packet. The vector captures the temporal context of the packet's channel and sender. 
* Next, the features are mapped to the visible neurons of an ensemble of autoenoders (*KitNET* https://github.com/ymirsky/KitNET-py). 
* Then, each autoencoder attempts to reconstruct the instance's features, and computes the reconstruction error in terms of root mean squared errors (RMSE). 
* Finally, the RMSEs are forwarded to an output autoencoder, which acts as a non-linear voting mechanism for the ensemble. 

We note that while training \textbf{Kitsune}, no more than one instance is stored in memory at a time. Kitsune has one main parameter, which is the maximum number of inputs for any given autoencoder in the ensemble. This parameter is used to increase the algorithm's speed with a modest trade off in detection performance.
	
![An illustration of Kitsune's architecture](https://raw.githubusercontent.com/ymirsky/Kitsune-py/master/Kitsune_fig.png)

	
Some points about KitNET:
* It is completely plug-and-play.
* It is based on an unsupervised machine learning algorithm (it does not need label, just train it on *normal* data!)
* Its efficiency can be scaled with its input parameter m: the maximal size of any autoencoder in the ensemble layer (smaller autoencoders are exponentially cheaper to train and execute)

# Implimentation Notes: 

* This python implimentation of Kitsune is **is not optimal** in terms of speed. To make Kitsune run as fast as described in the paper, the entire project must be cythonized, or implimented in C++
* For an experimental AfterImage version, change the import line in netStat.py to use AfterImage_extrapolate.py, and change line 5 of FeatureExtractor.py to True (uses cython). This version uses Lagrange-based Polynomial extrapolation to assit in computing the correlation based features.
* We also require the scapy library for parsing (tshark [Wireshark] is default).
* The source code has been tested with Anaconda 3.6.3 on a Windows 10 64bit machine.

To install scapy, run in the terminal:
```
pip install scapy
```
 


# Using The Code
Here is a simple example of how to make a Kitsune object:
```
from Kitsune import *


# KitNET params:
maxAE = 10 #maximum size for any autoencoder in the ensemble layer
FMgrace = 5000 #the number of instances taken to learn the feature mapping (the ensemble's architecture)
ADgrace = 50000 #the number of instances used to train the anomaly detector (ensemble itself)
packet_limit = np.Inf #the number of packets from the input file to process
path = "../../captured.pcap" #the pcap, pcapng, or tsv file which you wish to process.

# Build Kitsune
K = Kitsune(path,packet_limit,maxAE,FMgrace,ADgrace)
```

You can also configure the learning rate and hidden layer's neuron ratio via Kitsune's contructor.

The input file can be any pcap network capture. When the object is created, the code check whether or not you have tshark (Wireshark) installed. If you do, then it uses tshark to parse the pcap into a tsv file which is saved to disk locally. This file is then later used when running Kitnet. You can also load this tsv file instead of the origional pcap to save time. Note that we currently only look for tshark in the Windows directory "C:\Program Files\Wireshark\tshark.exe"

If tshark is not found, then the scapy packet parsing library is used. Scapy is significatly slower than using wireshark/tsv...

To use the Kitsune object, simply tell Kitsune to process the next packet. After processing a packet, Kitsune returns the RMSE value of the packet (zero during the FM featuremapping and AD grace periods).

Here is an example usage of the Kitsune object:
```
while True: 
    rmse = K.proc_next_packet() #will train during the grace periods, then execute on all the rest.
    if rmse == -1:
        break
    print(rmse)
```


# Demo Code
As a quick start, a demo script is provided in example.py. In the demo, we run Kitsune on a network capture of the Mirai malware. You can either run it directly or enter the following into your python console
```
import example.py
```


The code was written and with the Python environment Anaconda: https://anaconda.org/anaconda/python
For significant speedups, as shown in our paper, you must implement Kitsune in C++, or entirely using cython.

# Full Datasets
The full datasets used in our NDSS paper can be found by following this google drive link:
https://goo.gl/iShM7E

# License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details


# Citations
If you use the source code, the datasets, or implement KitNET, please cite the following paper:

*Yisroel Mirsky, Tomer Doitshman, Yuval Elovici, and Asaf Shabtai, "Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection", Network and Distributed System Security Symposium 2018 (NDSS'18)*

Yisroel Mirsky
yisroel@post.bgu.ac.il

