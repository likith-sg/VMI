# main.py
from vmi_integration import initialize_vmi, get_vmi_instance
from analysis_modules import monitor_memory, monitor_processes, monitor_syscalls, monitor_files, monitor_network, analyze_behavior, detect_anomalies
from gui_module import create_gui
from reporting_module import generate_report

def main():
    # Initialize VMI
    vmi = initialize_vmi()
    vmi_instance = get_vmi_instance(vmi)

    # Collect Data
    data = {
        'memory': monitor_memory(vmi_instance),
        'processes': monitor_processes(vmi_instance),
        'syscalls': monitor_syscalls(vmi_instance),
        'files': monitor_files(vmi_instance),
        'network': monitor_network(vmi_instance)
    }

    # Analyze Behavior
    analysis_results = analyze_behavior(data)
    
    # Detect Anomalies
    anomalies = detect_anomalies(analysis_results)

    # Reporting
    generate_report(anomalies)

    # Start GUI
    create_gui()

if __name__ == "__main__":
    main()
