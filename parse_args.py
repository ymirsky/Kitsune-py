import argparse

def parse_args():

    parser = argparse.ArgumentParser(description='Kitsune project')
    parser.add_argument('--dataset',
						help='dataset name: [mirai, SYN_DoS, SSDP_Flood].',
						type=str,
						default='mirai')
    parser.add_argument('--job_description',
						help='Any verbal description to distinguish between results.',
						type=str,
						default='job')
    parser.add_argument('--threshold',
						help='Anomaly threshold.',
						type=float,
						default=0.4)
    
    args = parser.parse_args()
    
    return args