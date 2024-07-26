import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import plotly.graph_objects as go
import plotly.express as px
import os
import logging
import traceback

# Set up logging
logging.basicConfig(filename='reporting_module.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_plot_as_image(fig, filename, dpi=300):
    """
    Save a Matplotlib figure as an image file with specified DPI.
    """
    fig.savefig(filename, dpi=dpi)
    plt.close(fig)

def save_html_plot(fig, filename):
    """
    Save a Plotly figure as an HTML file.
    """
    fig.write_html(filename)

def plot_feature_distributions(features_df, output_dir):
    """
    Plot distribution of features with KDE using Seaborn and Plotly.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Seaborn plots
    plt.figure(figsize=(14, 10))
    num_features = len(features_df.columns)
    num_rows = (num_features + 2) // 3
    for i, feature in enumerate(features_df.columns):
        plt.subplot(num_rows, 3, i + 1)
        sns.histplot(features_df[feature], kde=True, color='blue', bins=30)
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
    save_plot_as_image(plt.gcf(), os.path.join(output_dir, 'feature_distributions.png'))
    
    # Plotly interactive plots
    for feature in features_df.columns:
        fig = px.histogram(features_df, x=feature, marginal="box", title=f'Distribution of {feature}')
        fig.update_layout(title_text=f'Distribution of {feature}', xaxis_title=feature, yaxis_title='Frequency')
        save_html_plot(fig, os.path.join(output_dir, f'{feature}_distribution.html'))

def plot_anomalies_heatmap(features_df, anomalies, model_name, output_dir):
    """
    Plot a heatmap of anomalies using Plotly.
    """
    if model_name not in anomalies:
        logging.warning(f"No anomalies found for model: {model_name}")
        return

    anomaly_indices = anomalies[model_name]
    heatmap_data = np.zeros(features_df.shape)
    heatmap_data[anomaly_indices, :] = 1

    fig = go.Figure(data=go.Heatmap(z=heatmap_data, colorscale='Reds', colorbar=dict(title='Anomaly Intensity')))
    fig.update_layout(title=f'Anomaly Heatmap for {model_name}', xaxis_title='Feature Index', yaxis_title='Sample Index')
    save_html_plot(fig, os.path.join(output_dir, f'{model_name}_heatmap.html'))

def generate_pdf_report(features_df, anomalies, output_dir, file_name='anomaly_report.pdf'):
    """
    Generate a detailed PDF report of anomalies and visualizations.
    """
    try:
        with PdfPages(os.path.join(output_dir, file_name)) as pdf:
            # Plot feature distributions
            plt.figure(figsize=(14, 10))
            num_features = len(features_df.columns)
            num_rows = (num_features + 2) // 3
            for i, feature in enumerate(features_df.columns):
                plt.subplot(num_rows, 3, i + 1)
                sns.histplot(features_df[feature], kde=True)
                plt.title(f'Distribution of {feature}')
                plt.xlabel(feature)
                plt.ylabel('Frequency')
            pdf.savefig()
            plt.close()

            # Plot pairwise feature relationships
            plt.figure(figsize=(12, 10))
            sns.pairplot(features_df, diag_kind='kde')
            plt.suptitle('Pairwise Feature Relationships', y=1.02)
            pdf.savefig()
            plt.close()

            # Plot anomaly heatmaps and distributions
            for model_name in anomalies:
                # Save heatmap as HTML and include in PDF
                plot_anomalies_heatmap(features_df, anomalies, model_name, output_dir)
                pdf.savefig(os.path.join(output_dir, f'{model_name}_heatmap.html'))
                
                # Anomaly scores plot
                plt.figure()
                anomaly_scores = np.zeros(features_df.shape[0])
                if model_name in anomalies:
                    anomaly_scores[anomalies[model_name]] = 1
                plt.scatter(range(len(anomaly_scores)), anomaly_scores, c='red', label='Anomalies')
                plt.title(f'Anomaly Scores for {model_name}')
                plt.xlabel('Index')
                plt.ylabel('Anomaly Score')
                plt.legend()
                save_plot_as_image(plt.gcf(), os.path.join(output_dir, f'{model_name}_anomalies_plot.png'))
                pdf.savefig(os.path.join(output_dir, f'{model_name}_anomalies_plot.png'))
                
    except Exception as e:
        logging.error(f"Error generating PDF report: {e}")
        logging.debug(traceback.format_exc())

def visualize_anomalies(data, anomalies, output_dir='output'):
    """
    Visualize detected anomalies and generate detailed reports.
    """
    try:
        # Convert the data dictionary to a DataFrame
        features_df = pd.DataFrame(data)

        # Create output directory if not exists
        os.makedirs(output_dir, exist_ok=True)

        # Generate PDF report
        generate_pdf_report(features_df, anomalies, output_dir)
        
        # Save interactive HTML plots
        plot_feature_distributions(features_df, output_dir)
        
        # Optional: Save individual plots
        for model_name, anomaly_indices in anomalies.items():
            plt.figure()
            plt.scatter(range(len(anomaly_indices)), anomaly_indices, c='red', label='Anomalies')
            plt.title(f'Anomalies Detected by {model_name}')
            plt.xlabel('Index')
            plt.ylabel('Anomaly Score')
            plt.legend()
            save_plot_as_image(plt.gcf(), os.path.join(output_dir, f'{model_name}_anomalies_plot.png'))
            plt.show()

    except Exception as e:
        logging.error(f"Error visualizing anomalies: {e}")
        logging.debug(traceback.format_exc())
