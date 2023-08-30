import sklearn.metrics

from KitPlugin import KitPlugin
import sklearn
import optuna
import numpy as np
from scipy.stats import norm

def objective(trial):
    numAE = trial.suggest_int('numAE', 1, 10)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.5)
    hidden_ratio = trial.suggest_float('hidden_ratio', 0.5, 0.8)

    kit = KitPlugin("input_data/mirai.pcap", 200000, numAE, 5000, 50000, learning_rate, hidden_ratio)
    # Load the feature list beforehand to save time
    kit.feature_loader()
    kit.kit_trainer(0, 60000)

    y_test = np.zeros((200, 1))
    y_pred = kit.kit_runner(121550, 121750)

    # Do small test run with benign sample to find normalization
    print("Calculating normalization sample")
    benignSample = np.log(kit.kit_runner(70000, 80000))
    logProbs = norm.logsf(np.log(y_pred), np.mean(benignSample), np.std(benignSample))

    error = sklearn.metrics.mean_squared_error(y_test, logProbs)
    return error

study = optuna.create_study()
study.optimize(objective, n_trials=100)