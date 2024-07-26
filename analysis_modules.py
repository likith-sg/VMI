# analysis_modules.py
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow.keras import layers, models
from scipy import stats
import logging
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(filename='behavior_analysis.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_memory_features(memory_data):
    """
    Extract advanced features from memory data, including statistical moments and entropy.
    """
    memory_array = np.array(memory_data)
    features = {
        'mean': np.mean(memory_array),
        'variance': np.var(memory_array),
        'skewness': stats.skew(memory_array),
        'kurtosis': stats.kurtosis(memory_array),
        'entropy': stats.entropy(np.histogram(memory_array, bins='auto')[0]),
        'median': np.median(memory_array),
        'max': np.max(memory_array),
        'min': np.min(memory_array),
    }
    return features

def extract_process_features(process_list):
    """
    Extract features from process data, including statistical measures and distributions.
    """
    process_series = pd.Series(process_list)
    process_counts = process_series.value_counts()
    features = {
        'process_counts': process_counts.to_dict(),
        'unique_processes': len(process_counts),
        'process_mean_count': process_series.mean(),
        'process_stddev_count': process_series.std(),
        'process_entropy': stats.entropy(process_counts.values)
    }
    return features

def extract_syscall_features(syscall_list):
    """
    Extract features from syscall data, including frequency, statistical distributions, and time series analysis.
    """
    syscall_series = pd.Series(syscall_list)
    syscall_counts = syscall_series.value_counts()
    features = {
        'syscall_counts': syscall_counts.to_dict(),
        'unique_syscalls': len(syscall_counts),
        'syscall_mean_count': syscall_series.mean(),
        'syscall_stddev_count': syscall_series.std(),
        'syscall_entropy': stats.entropy(syscall_counts.values)
    }
    return features

def extract_file_features(file_events):
    """
    Extract features from file operations, focusing on event types, frequencies, and time series analysis.
    """
    file_series = pd.Series(file_events)
    file_counts = file_series.value_counts()
    features = {
        'file_event_counts': file_counts.to_dict(),
        'unique_file_events': len(file_counts),
        'file_event_mean_count': file_series.mean(),
        'file_event_stddev_count': file_series.std(),
        'file_event_entropy': stats.entropy(file_counts.values)
    }
    return features

def extract_network_features(network_activity):
    """
    Extract features from network activity, including traffic patterns, connections, and time series analysis.
    """
    network_series = pd.Series(network_activity)
    network_counts = network_series.value_counts()
    features = {
        'network_activity_counts': network_counts.to_dict(),
        'unique_network_events': len(network_counts),
        'network_activity_mean_count': network_series.mean(),
        'network_activity_stddev_count': network_series.std(),
        'network_activity_entropy': stats.entropy(network_counts.values)
    }
    return features

def scale_features(features, method='standard'):
    """
    Standardize or normalize features to improve the performance of anomaly detection.
    """
    scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
    features_array = np.array(list(features.values())).reshape(-1, 1)
    scaled_features = scaler.fit_transform(features_array)
    return scaled_features

def dimensionality_reduction(features):
    """
    Apply dimensionality reduction to simplify the feature space.
    """
    pca = PCA(n_components=min(len(features), 10))  # Reduce to 10 components or fewer
    reduced_features = pca.fit_transform(features)
    return reduced_features

def build_autoencoder(input_shape):
    """
    Build and compile an autoencoder for anomaly detection.
    """
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(input_shape[0], activation='sigmoid'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def monitor_memory(vmi_instance):
    """
    Monitor VM memory and extract features.
    """
    memory_data = vmi_instance.get_memory()
    features = extract_memory_features(memory_data)
    return features

def monitor_processes(vmi_instance):
    """
    Monitor VM processes and extract features.
    """
    process_list = vmi_instance.get_process_list()
    features = extract_process_features(process_list)
    return features

def monitor_syscalls(vmi_instance):
    """
    Monitor system calls and extract features.
    """
    syscall_list = vmi_instance.get_syscalls()
    features = extract_syscall_features(syscall_list)
    return features

def monitor_files(vmi_instance):
    """
    Monitor file operations and extract features.
    """
    file_events = vmi_instance.get_file_events()
    features = extract_file_features(file_events)
    return features

def monitor_network(vmi_instance):
    """
    Monitor network activity and extract features.
    """
    network_activity = vmi_instance.get_network_activity()
    features = extract_network_features(network_activity)
    return features

def collect_data(vmi_instance):
    """
    Collect and integrate data from all monitoring functions.
    """
    data = {
        'memory': monitor_memory(vmi_instance),
        'processes': monitor_processes(vmi_instance),
        'syscalls': monitor_syscalls(vmi_instance),
        'files': monitor_files(vmi_instance),
        'network': monitor_network(vmi_instance)
    }
    return data

def analyze_behavior(data):
    """
    Analyze collected data using statistical methods and machine learning.
    """
    combined_features = {}
    combined_features.update(data['memory'])
    combined_features.update(data['processes'])
    combined_features.update(data['syscalls'])
    combined_features.update(data['files'])
    combined_features.update(data['network'])
    
    features_df = pd.DataFrame([combined_features])
    scaled_features = scale_features(combined_features)
    reduced_features = dimensionality_reduction(scaled_features)

    return features_df, reduced_features

def detect_anomalies(analysis_results):
    """
    Detect anomalies using Isolation Forest, One-Class SVM, and Autoencoders.
    """
    results_df, reduced_features = analyze_behavior(analysis_results)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(reduced_features)

    # Isolation Forest
    iso_forest = IsolationForest(contamination=0.01)
    iso_forest_preds = iso_forest.fit_predict(scaled_features)
    iso_forest_anomalies = np.where(iso_forest_preds == -1)[0]
    
    # One-Class SVM
    oc_svm = OneClassSVM(nu=0.01, kernel='rbf', gamma='scale')
    oc_svm_preds = oc_svm.fit_predict(scaled_features)
    oc_svm_anomalies = np.where(oc_svm_preds == -1)[0]
    
    # Autoencoder
    autoencoder = build_autoencoder(input_shape=(scaled_features.shape[1],))
    autoencoder.fit(scaled_features, scaled_features, epochs=10, batch_size=32, verbose=0)
    reconstructions = autoencoder.predict(scaled_features)
    mse = np.mean(np.power(scaled_features - reconstructions, 2), axis=1)
    autoencoder_anomalies = np.where(mse > np.percentile(mse, 99))[0]
    
    # Combine results
    anomalies = {
        'iso_forest_anomalies': iso_forest_anomalies,
        'oc_svm_anomalies': oc_svm_anomalies,
        'autoencoder_anomalies': autoencoder_anomalies
    }
    
    # Logging
    for model, indices in anomalies.items():
        if len(indices) > 0:
            logging.info(f"Detected anomalies using {model}: {indices}")
        else:
            logging.info(f"No anomalies detected using {model}.")
    
    return anomalies

def visualize_anomalies(data, anomalies):
    """
    Visualize detected anomalies for better understanding.
    """
    plt.figure(figsize=(10, 6))
    
    # Example visualization
    plt.subplot(2, 2, 1)
    plt.hist(data['memory'], bins=50, alpha=0.7, label='Memory')
    plt.title('Memory Data Distribution')
    
    plt.subplot(2, 2, 2)
    plt.hist(data['processes'], bins=50, alpha=0.7, label='Processes')
    plt.title('Processes Data Distribution')
    
    plt.subplot(2, 2, 3)
    plt.hist(data['syscalls'], bins=50, alpha=0.7, label='Syscalls')
    plt.title('Syscalls Data Distribution')
    
    plt.subplot(2, 2, 4)
    plt.hist(data['files'], bins=50, alpha=0.7, label='Files')
    plt.title('Files Data Distribution')
    
    plt.tight_layout()
    plt.savefig('anomalies_visualization.png')
    plt.show()

