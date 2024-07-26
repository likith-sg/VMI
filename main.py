import asyncio
import logging
import traceback
import json
import yaml
from vmi_integration import VMIIntegration
from analysis_modules import (
    monitor_memory, monitor_processes, monitor_syscalls,
    monitor_files, monitor_network, analyze_behavior,
    detect_anomalies
)
from reporting_module import generate_report
from gui_module import AdvancedApp

# Load configuration
def load_config(file_path):
    """Load configuration from a JSON or YAML file."""
    with open(file_path, 'r') as file:
        if file_path.endswith('.json'):
            return json.load(file)
        elif file_path.endswith('.yaml') or file_path.endswith('.yml'):
            return yaml.safe_load(file)
        else:
            raise ValueError("Unsupported configuration file format")

config = load_config('config.yaml')

# Configure logging
logging.basicConfig(
    filename=config.get('log_file', 'main.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

async def collect_data(vmi_instance):
    """
    Collect and integrate data from all monitoring functions.
    """
    try:
        logging.info("Starting data collection.")
        # Create asynchronous tasks for each monitoring function
        tasks = {
            'memory': asyncio.create_task(monitor_memory(vmi_instance)),
            'processes': asyncio.create_task(monitor_processes(vmi_instance)),
            'syscalls': asyncio.create_task(monitor_syscalls(vmi_instance)),
            'files': asyncio.create_task(monitor_files(vmi_instance)),
            'network': asyncio.create_task(monitor_network(vmi_instance))
        }
        
        # Gather results from all tasks
        results = await asyncio.gather(*tasks.values())
        
        # Combine results into a dictionary
        data = {key: results[idx] for idx, key in enumerate(tasks)}
        logging.info("Data collection completed successfully.")
        return data
    
    except Exception as e:
        logging.error(f"Error during data collection: {e}")
        logging.debug(traceback.format_exc())
        return {}

async def analyze_and_report(data):
    """
    Analyze the collected data, detect anomalies, and generate a report.
    """
    try:
        logging.info("Starting data analysis.")
        if not data:
            raise ValueError("No data collected for analysis.")
        
        # Analyze Behavior
        analysis_results, reduced_features = analyze_behavior(data)
        
        # Detect Anomalies
        anomalies = detect_anomalies(analysis_results)
        
        # Log anomalies
        for model, indices in anomalies.items():
            if indices:
                logging.info(f"Detected anomalies using {model}: {indices}")
            else:
                logging.info(f"No anomalies detected using {model}.")
        
        # Generate and Save Report
        report_path = generate_report(anomalies)
        logging.info(f"Report generated at: {report_path}")

    except ValueError as ve:
        logging.warning(f"ValueError during analysis and reporting: {ve}")
    except Exception as e:
        logging.error(f"Error during analysis and reporting: {e}")
        logging.debug(traceback.format_exc())

async def initialize_vmi():
    """
    Initialize VMI integration.
    """
    try:
        async with VMIIntegration() as vmi_integration:
            return vmi_integration.get_vmi_instance()
    except Exception as e:
        logging.error(f"Error initializing VMI: {e}")
        logging.debug(traceback.format_exc())
        raise

async def main():
    """
    Main function to initialize VMI, collect data, analyze, detect anomalies, and start GUI.
    """
    try:
        vmi_instance = await initialize_vmi()
        data = await collect_data(vmi_instance)
        await analyze_and_report(data)
    
    except Exception as e:
        logging.error(f"Unhandled exception in main: {e}")
        logging.debug(traceback.format_exc())

    try:
        logging.info("Starting GUI.")
        app = AdvancedApp()
        app.mainloop()
    except Exception as e:
        logging.error(f"Error starting GUI: {e}")
        logging.debug(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main())
