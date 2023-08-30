from KitPlugin import KitPlugin

training_min = 0
training_max = 60000
testing_min = 140329
testing_max = 140333

# Instantiate KitPlugin with appropriate parameters
packet_limit = 200000
maxAE = 10
FMgrace = 5000
ADgrace = 50000
inputs = {
    "mirai_malicious" : {
        "input_path" : "input_data/mirai.pcap",
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
        "input_path": "input_data/mirai.pcap",
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
KitPlugin = KitPlugin()
KitPlugin.run_series_stats(inputs)
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