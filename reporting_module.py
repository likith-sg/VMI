# reporting_module.py
import matplotlib.pyplot as plt

def generate_report(anomalies):
    # Example function for generating a report
    if anomalies:
        plt.figure()
        # Plot anomalies (example)
        plt.plot(anomalies)
        plt.title("Detected Anomalies")
        plt.xlabel("Index")
        plt.ylabel("Anomaly Score")
        plt.show()
    else:
        print("No anomalies detected.")
