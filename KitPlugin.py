import openpyxl

from Kitsune import Kitsune
import shap
import numpy as np
import pickle
from openpyxl import Workbook
from openpyxl.styles import PatternFill
from datetime import datetime
import sklearn
import optuna
from scipy.stats import norm

# Class that provides a callable interface for Kitsune components.
# Note that this approach nullifies the "incremental" aspect of Kitsune and significantly slows it down.
class KitPlugin:
    # Function used by SHAP as callback to test instances of features
    def kitsune_model(self, input_data):
        prediction = self.K.feed_batch(input_data)
        return prediction

    # Builds a Kitsune instance. Does not train KitNET yet.
    def __init__(self, input_path=None, packet_limit=None, num_autenc=None, FMgrace=None, ADgrace=None, learning_rate=0.1, hidden_ratio=0.75):
        # This code will be removed when batch running Kitsune has been finalized
        if input_path != None and num_autenc != None:
            self.features_list = None
            self.explainer = None
            self.shap_values = None
            self.K = Kitsune(input_path, packet_limit, num_autenc, FMgrace, ADgrace, learning_rate, hidden_ratio)
            self.metadata = {
                "filename" : input_path,
                "packet_limit" : packet_limit,
                "num_autenc" : num_autenc,
                "FMgrace": FMgrace,
                "ADgrace" : ADgrace,
                "timestamp" : datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            }
        self.testFeatures = None

    # Calls Kitsune's get_feature_list function to build the list of features
    def feature_builder(self):
        print("Building features")
        # Dummy-running Kitsune to get a list of features
        self.features_list = self.K.get_feature_list()

    # Loads Kitsune's feature list from a pickle file
    def feature_loader(self):
        print("Loading features from file")
        with open('pickles/featureList.pkl', 'rb') as f:
            features_list = pickle.load(f)
        self.features_list = features_list

    # Writes Kitsune's feature list to a pickle file
    def feature_pickle(self):
        print("Writing features to file")
        with open('pickles/featureList.pkl', 'wb') as f:
            pickle.dump(self.features_list, f)

    # Trains KitNET, using the specified index range of this class' feature list
    def kit_trainer(self, min_index, max_index):
        print("Training")
        self.K.feed_batch(self.features_list[min_index:max_index])
        print("Training finished")

    # Runs KitNET, using specified index range of this class' feature list
    def kit_runner(self, min_index, max_index, normalize=False):
        print("Running")
        return self.K.feed_batch(self.features_list[min_index:max_index])

    # Calculates KitNET's SHAP-values for the specified indexes
    def shap_values_builder(self, min_train, max_train, min_test, max_test):
        self.metadata['min_train'] = min_train
        self.metadata['max_train'] = max_train
        self.metadata['min_test'] = min_test
        self.metadata['max_test'] = max_test
        print("Building SHAP explainer")
        self.explainer = shap.Explainer(self.kitsune_model, np.array(self.features_list[min_train:max_train]))
        print("Calculating SHAP values")
        if self.testFeatures != None:
            self.shap_values = self.explainer.shap_values(np.array(self.testFeatures[min_test:max_test]))
        else:
            self.shap_values = self.explainer.shap_values(np.array(self.features_list[min_test:max_test]))

    # Writes the SHAP-values to a pickle-file
    def shap_values_pickle(self):
        with open('pickles/shap_values.pkl', 'wb') as f:
            pickle.dump(self.shap_values, f)

    # Gets the SHAP-values from a pickle-file
    def shap_values_loader(self):
        with open('pickles/shap_values.pkl', 'rb') as f:
            self.shap_values = pickle.load(f)

    # Calculates summary statistics of SHAP-values
    def shap_stats_summary_builder(self, min_index, max_index, plot_type="dot"):
        return shap.summary_plot(self.shap_values, np.array(self.features_list[min_index:max_index]), plot_type=plot_type)

    # Creates an Excel-file containing summary statistics for each feature
    def shap_stats_excel_export(self):
        self.workbook = openpyxl.load_workbook('input_data/template_statistics_file.xlsx')
        self.create_sheet("mirai_60k_4asdf")
        excel_file = "summary_statistics_test.xlsx"
        self.workbook.save(excel_file)
        print('done')

    # Prints the three best and worst values for all statistics
    def get_high_low_indices(self):
        shap_transposed = self.shap_values.T
        # List of statistics functions
        stat_functions = {
            'mean': np.mean,
            'median': np.median,
            'std_dev': np.std,
            'variance': np.var,
            'minimum': np.min,
            'maximum': np.max,
            'total_sum': np.sum
        }

        # Dictionary to store results
        result_dict = {}

        # Loop over statistics
        for stat_name, stat_func in stat_functions.items():
            # Calculate the statistic for each list
            stat_values = stat_func(shap_transposed, axis=1)

            # Calculate the indices of the highest and lowest values
            sorted_indices = np.argsort(stat_values)
            highest_indices = sorted_indices[-3:]
            lowest_indices = sorted_indices[:3]
            # Store the indices in the result dictionary
            result_dict[stat_name] = {
                'highest_indices': highest_indices,
                'lowest_indices': lowest_indices
            }
        return result_dict

    # Creates an Excel sheet with relevant statistics
    def create_sheet(self, sheet_title):
        sheet = self.workbook.copy_worksheet(self.workbook.active)
        sheet.title = sheet_title
        headers = ['Mean', 'Median', 'Standard Deviation', 'Variance', 'Minimum', 'Maximum', 'Sum', 'Metadata']
        header_row = headers
        for col, value in enumerate(header_row):
            cell = sheet.cell(row=1, column=6 + col)
            cell.value = value
        for idx, num_list in enumerate(self.shap_values.T):
            mean = np.mean(num_list)
            median = np.median(num_list)
            std_dev = np.std(num_list)
            variance = np.var(num_list)
            minimum = np.min(num_list)
            maximum = np.max(num_list)
            total_sum = np.sum(num_list)

            row_data = [mean, median, std_dev, variance, minimum, maximum, total_sum]

            for col, value in enumerate(row_data):
                cell = sheet.cell(row=idx + 2, column=6 + col)
                cell.value = value

        color_indices = self.get_high_low_indices()
        stat_columns = {
            'mean': "F",
            'median': "G",
            'std_dev': "H",
            'variance': "I",
            'minimum': "J",
            'maximum': "K",
            'total_sum': "L"
        }
        for stat in color_indices:
            for index in color_indices[stat]["highest_indices"]:
                cell_index = stat_columns[stat] + str(index + 2)
                if stat == "std_dev" or stat == "variance":
                    # Make largest three standard deviation and variance values blue
                    sheet[cell_index].fill = PatternFill(start_color="ADD8E6", end_color="ADD8E6", fill_type="solid")
                elif stat == "minimum":
                    # Make largest three cells minimum red
                    sheet[cell_index].fill = PatternFill(start_color="FF0000", end_color="FF0000", fill_type="solid")
                else:
                    # In all other cases, make largest three cells green
                    sheet[cell_index].fill = PatternFill(start_color="00FF00", end_color="00FF00", fill_type="solid")
            for index in color_indices[stat]["lowest_indices"]:
                cell_index = stat_columns[stat] + str(index + 2)
                if stat == "minimum":
                    # Make largest three cells minimum red
                    sheet[cell_index].fill = PatternFill(start_color="FF0000", end_color="FF0000", fill_type="solid")
                else:
                    # In all other cases, make smallest three cells red
                    sheet[cell_index].fill = PatternFill(start_color="FF0000", end_color="FF0000", fill_type="solid")
        # Fill in metadata
        start_row = 2
        start_column_keys = 'M'
        start_column_values = 'N'

        # Loop over the dictionary and write capitalized keys and values to cells
        for idx, (key, value) in enumerate(self.metadata.items()):
            capitalized_key = key[0].upper() + key[1:]
            key_cell = f"{start_column_keys}{start_row + idx}"
            value_cell = f"{start_column_values}{start_row + idx}"
            sheet[key_cell] = capitalized_key
            sheet[value_cell] = value
        return sheet

    # Runs a series of Kitsune models and calculates statistics for each run.
    def run_series_stats(self, inputs):
        self.workbook = openpyxl.load_workbook('input_data/template_statistics_file.xlsx')
        # Loop over the different Kitsune configs we are going to make
        for session in inputs:
            self.features_list = None
            self.explainer = None
            self.shap_values = None
            self.K = Kitsune(inputs[session]["input_path"], inputs[session]["packet_limit"], inputs[session]["maxAE"], inputs[session]["FMgrace"], inputs[session]["ADgrace"])
            self.metadata = {
                "filename": inputs[session]["input_path"],
                "packet_limit": inputs[session]["packet_limit"],
                "maxAE": inputs[session]["maxAE"],
                "FMgrace": inputs[session]["FMgrace"],
                "ADgrace": inputs[session]["ADgrace"],
                "timestamp": datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            }
            self.feature_builder()
            self.kit_trainer(inputs[session]["training_min"], inputs[session]["training_max"])
            if inputs[session]["input_path"] != inputs[session]["input_path_test"]:
                self.testKit = Kitsune(inputs[session]["input_path_test"], inputs[session]["packet_limit"], inputs[session]["maxAE"], inputs[session]["FMgrace"], inputs[session]["ADgrace"])
                self.testFeatures = self.testKit.get_feature_list()
            self.shap_values_builder(inputs[session]["training_min"], inputs[session]["training_max"], inputs[session]["testing_min"], inputs[session]["testing_max"])
            self.create_sheet(session)
        excel_file = "summary_statistics_" + datetime.datetime.now().strftime('%d-%m-%Y_%H-%M') + ".xlsx"
        self.workbook.save(excel_file)

    # Runs a hyperparameter optimization on the supplied dataset, constrained by packet limit and number of runs
    def hyper_opt(self, input_path, packet_limit, runs):
        def objective(trial):
            numAE = trial.suggest_int('numAE', 1, 10)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.5)
            hidden_ratio = trial.suggest_float('hidden_ratio', 0.5, 0.8)

            self.K = Kitsune(input_path, packet_limit, numAE, 5000, 50000, learning_rate, hidden_ratio)
            # Load the feature list beforehand to save time
            self.feature_loader()
            self.kit_trainer(0, 60000)

            y_test = np.zeros((200, 1))
            y_pred = self.kit_runner(121550, 121750)

            # Do small test run with benign sample to find normalization
            print("Calculating normalization sample")
            benignSample = np.log(self.kit_runner(70000, 80000))
            logProbs = norm.logsf(np.log(y_pred), np.mean(benignSample), np.std(benignSample))

            error = sklearn.metrics.mean_squared_error(y_test, logProbs)
            return error

        study = optuna.create_study()
        study.optimize(objective, n_trials=runs)

        # Create a new workbook and select the active worksheet
        wb = Workbook()
        ws = wb.active

        # Write header row
        header = ["Trial Number", "numAE", "learning_rate", "hidden_ratio"]
        ws.append(header)

        # Write trial information
        best_value = float("inf")
        best_row_idx = None  # Track the index of the best row
        for idx, trial in enumerate(study.trials, start=2):  # Start from row 2 to leave room for the header
            trial_params = trial.params
            trial_row = [trial.number, trial_params["numAE"], trial_params["learning_rate"], trial_params["hidden_ratio"], trial.value]
            ws.append(trial_row)

            if trial.value < best_value:
                best_value = trial.value
                best_row_idx = idx

        # Set fill color for the best value row
        green_fill = PatternFill(start_color="00FF00", end_color="00FF00", fill_type="solid")
        if best_row_idx is not None:
            for cell in ws[best_row_idx]:
                cell.fill = green_fill

        # Save the workbook to a file

        # Save the workbook to a file
        excel_file_path = "output_data/hyperparameter_optimization_results_" + datetime.datetime.now().strftime('%d-%m-%Y_%H-%M') + ".xlsx"
        wb.save(excel_file_path)

        print("Results exported to", excel_file_path)
        return study.best_trial

    def calc_eer(self, RMSEs, expected):
        fpr, tpr, threshold = sklearn.metrics.roc_curve(RMSEs, expected, pos_label=1)
        fnr = 1-tpr
        #eer_threshold = threshold[np.nanargmin(np.absolute((fnr-fpr)))]
        EER = fpr[np.nanargmin(np.absolute((fnr-fpr)))]
        return EER