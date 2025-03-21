import os
import numpy as np
import xarray as xr
from scipy.stats import pearsonr
from scipy.stats import wasserstein_distance
from scipy.ndimage import zoom
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
#from multiprocessing import Pool

# Define function to calculate RMSE
def calculate_rmse(reference, comparison):
    return np.sqrt(np.mean((reference - comparison)**2, axis=0))

# Define function to calculate Mean Bias
def calculate_mean_bias(reference, comparison):
    return np.mean(comparison - reference, axis=0)

# Define function to calculate Ratio of Variance
def calculate_ratio_of_variance(reference, comparison):
    reference_variance = np.var(reference, axis=0)
    comparison_variance = np.var(comparison, axis=0)
    return (comparison_variance / reference_variance) * 100

# Define function to calculate Pearson Correlation Coefficient
def calculate_correlation(reference, comparison):
    cor_map = np.zeros(reference.shape[1:])
    for i in range(reference.shape[1]):
        for j in range(reference.shape[2]):
            cor_map[i, j], _ = pearsonr(reference[:, i, j], comparison[:, i, j])
    return cor_map

# Define function to calculate Wasserstein Distance
def calculate_wasserstein_distance_abs(reference, comparison):
    wass_map = np.zeros(reference.shape[1:])
    for i in range(reference.shape[1]):
        for j in range(reference.shape[2]):
            wass_map[i, j] = wasserstein_distance(reference[:, i, j], comparison[:, i, j])
    return wass_map


# Define function to calculate normalized Wasserstein Distance
def calculate_wasserstein_distance_rel(reference, comparison):
    wass_map = np.zeros(reference.shape[1:])
    
    for i in range(reference.shape[1]):
        for j in range(reference.shape[2]):
            ref_series = reference[:, i, j]
            comp_series = comparison[:, i, j]

            # Normalize the data to [0, 1] range
            ref_min, ref_max = np.min(ref_series), np.max(ref_series)
            comp_min, comp_max = np.min(comp_series), np.max(comp_series)

            if ref_max > ref_min and comp_max > comp_min:  # Avoid division by zero
                ref_series = (ref_series - ref_min) / (ref_max - ref_min)
                comp_series = (comp_series - comp_min) / (comp_max - comp_min)
                
                wass_map[i, j] = wasserstein_distance(ref_series, comp_series)
            else:
                wass_map[i, j] = np.nan  # Assign NaN if normalization fails (constant values)
    
    return wass_map

# Define function to calculate 99th Percentile
def calculate_99th_percentile(data):
    return np.percentile(data, 99, axis=0)

# Define function to calculate Mean Value
def calculate_mean_value(data):
    return np.mean(data, axis=0)

# Define function to calculate Absolute Value
def calculate_abs_value(data):
    time_step = 108  # Fixed time step
    return data[time_step, :, :]

# Upscale function
def upsample_2d_array(low_res_array, upscale_factor):
    # Initialize high-resolution array
    time_records = low_res_array.shape[0]
    high_res_array = np.empty((time_records, 
                               low_res_array.shape[1] * upscale_factor, 
                               low_res_array.shape[2] * upscale_factor))
    
    # Apply zoom for each time record
    for t in range(time_records):
        high_res_array[t] = zoom(low_res_array[t], zoom=(upscale_factor, upscale_factor), order=1)
    
    return high_res_array


def compute_metrics(ground_truth, predictions_list, threshold=0.5):
    """
    Compute metrics per (latitude, longitude) for each experiment in predictions_list.
    
    Args:
        ground_truth: 3D numpy array (time, latitude, longitude) of binary ground truth (0 or 1).
        predictions_list: List of 3D numpy arrays (time, latitude, longitude) of predicted probabilities.
        threshold: Threshold to convert probabilities to binary predictions.
    
    Returns:
        A dictionary where each metric is a list of 2D arrays (latitude, longitude),
        one for each experiment.
    """
    # Initialize the metrics
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    results = {metric: [] for metric in metrics}

    # Iterate through each experiment in the predictions list
    for predictions in predictions_list:
        # Ensure shapes match
        assert ground_truth.shape == predictions.shape, "Shape mismatch between ground truth and predictions."

        # Get dimensions
        _, lat, lon = ground_truth.shape

        # Initialize 2D arrays for metrics
        accuracy_map = np.zeros((lat, lon))
        precision_map = np.zeros((lat, lon))
        recall_map = np.zeros((lat, lon))
        f1_map = np.zeros((lat, lon))
        roc_auc_map = np.zeros((lat, lon))

        # Loop over each grid point
        for i in range(lat):
            for j in range(lon):
                # Extract time-series data for the grid point
                y_true = ground_truth[:, i, j]
                y_pred_probs = predictions[:, i, j]

                # Only calculate metrics if there are positive cases in the ground truth
                if np.sum(y_true) > 0:
                    # Convert probabilities to binary predictions using the threshold
                    y_pred = (y_pred_probs >= threshold).astype(int)

                    # Compute metrics
                    accuracy_map[i, j] = accuracy_score(y_true, y_pred)
                    precision_map[i, j] = precision_score(y_true, y_pred, zero_division=0)
                    recall_map[i, j] = recall_score(y_true, y_pred, zero_division=0)
                    f1_map[i, j] = f1_score(y_true, y_pred, zero_division=0)
                    roc_auc_map[i, j] = roc_auc_score(y_true, y_pred_probs)
                else:
                    # If no positive cases, metrics are set to NaN (not applicable)
                    accuracy_map[i, j] = np.nan
                    precision_map[i, j] = np.nan
                    recall_map[i, j] = np.nan
                    f1_map[i, j] = np.nan
                    roc_auc_map[i, j] = np.nan

        # Append the results for this experiment
        results["accuracy"].append(accuracy_map)
        results["precision"].append(precision_map)
        results["recall"].append(recall_map)
        results["f1"].append(f1_map)
        results["roc_auc"].append(roc_auc_map)

    return results

def compute_metrics_vectorized(ground_truth, predictions, threshold):

    # Initialize the metrics
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    results = {metric: [] for metric in metrics}

    # Flatten time axis for ROC-AUC computation (time, lat, lon -> (time, lat*lon))
    ground_truth_flat = ground_truth.reshape(ground_truth.shape[0], -1)
    
    for prediction in predictions:
        # Convert probabilities to binary predictions using the threshold
        prediction = (prediction >= threshold).astype(int)
        prediction_flat = prediction.reshape(prediction.shape[0], -1)

        # Vectorized true/false positive/negative calculations
        tp = (prediction == 1) & (ground_truth == 1)
        tn = (prediction == 0) & (ground_truth == 0)
        fp = (prediction == 1) & (ground_truth == 0)
        fn = (prediction == 0) & (ground_truth == 1)

        # Use np.sum to quickly calculate counts
        tp_sum = np.sum(tp, axis=0)
        tn_sum = np.sum(tn, axis=0)
        fp_sum = np.sum(fp, axis=0)
        fn_sum = np.sum(fn, axis=0)

        # Compute metrics
        precision = tp_sum / (tp_sum + fp_sum + 1e-8)
        recall = tp_sum / (tp_sum + fn_sum + 1e-8)
        accuracy = (tp_sum + tn_sum) / (tp_sum + tn_sum + fp_sum + fn_sum + 1e-8)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

        # Flatten the arrays for ROC-AUC computation
        #ground_truth_flat = ground_truth.ravel()
        #prediction_flat = prediction.ravel()

        # ROC-AUC calculation
        roc_auc = np.zeros((ground_truth.shape[1], ground_truth.shape[2]))
        for i in range(ground_truth.shape[1]):  # Latitude
            for j in range(ground_truth.shape[2]):  # Longitude
                try:
                    roc_auc[i, j] = roc_auc_score(ground_truth_flat[:, i * ground_truth.shape[2] + j],
                                                  prediction_flat[:, i * prediction.shape[2] + j])
                except ValueError:
                    roc_auc[i, j] = np.nan  # Handle case where only one class is present

        """
        # Compute ROC-AUC (check if ground_truth contains both classes)
        if len(np.unique(ground_truth_flat)) > 1:  # Avoid errors when there's only one class
            roc_auc = roc_auc_score(ground_truth_flat, prediction_flat)
        else:
            roc_auc = np.nan  # Not computable
        """

        # Append the results for this experiment
        results["accuracy"].append(accuracy)
        results["precision"].append(precision)
        results["recall"].append(recall)
        results["f1"].append(f1_score)
        results["roc_auc"].append(roc_auc)

    return results



def compute_single_metrics(prediction, ground_truth):
    tp = (prediction == 1) & (ground_truth == 1)
    tn = (prediction == 0) & (ground_truth == 0)
    fp = (prediction == 1) & (ground_truth == 0)
    fn = (prediction == 0) & (ground_truth == 1)

    tp_sum = np.sum(tp, axis=0)
    tn_sum = np.sum(tn, axis=0)
    fp_sum = np.sum(fp, axis=0)
    fn_sum = np.sum(fn, axis=0)

    precision = tp_sum / (tp_sum + fp_sum + 1e-8)
    recall = tp_sum / (tp_sum + fn_sum + 1e-8)
    accuracy = (tp_sum + tn_sum) / (tp_sum + tn_sum + fp_sum + fn_sum + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1_score}


