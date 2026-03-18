<div align="center">

# VMI

> Virtual Machine Introspection with interactive reports 

![Language](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![GitHub Stars](https://img.shields.io/github/stars/likith-sg/VMI?style=for-the-badge&color=yellow)
![GitHub Forks](https://img.shields.io/github/forks/likith-sg/VMI?style=for-the-badge)
![Drift Detected](https://img.shields.io/badge/docs-drift%20detected-orange?style=for-the-badge)

</div>

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Tech Stack](#️-tech-stack)
- [Configuration](#️-configuration)
- [Contributing](#-contributing)

---

## 🎯 Overview
The Virtual Machine Introspection (VMI) project is a Python-based tool that utilizes interactive reports to provide insights into virtual machine activity. It is designed for users who require in-depth monitoring and analysis of virtual machine behavior, such as system administrators and security professionals. The project's unique aspect lies in its ability to collect and integrate data from various monitoring functions, including memory, processes, syscalls, files, and network activity, to provide a comprehensive view of the virtual machine's behavior.

The VMI project leverages a range of dependencies, including libvmi, numpy, pandas, scikit-learn, and tensorflow, to analyze the collected data and detect anomalies. The project's configuration is loaded from a JSON or YAML file, allowing users to customize the logging settings, such as the log file and level. The project's logging system is configured to write logs to a file, providing a record of the project's activity. The use of asyncio enables the project to collect data from multiple monitoring functions concurrently, making it efficient and scalable.

## ✨ Features
* 🔥 **Monitor Memory** — collects data on virtual machine memory usage
* 📊 **Monitor Processes** — tracks and analyzes running processes within the virtual machine
* 📝 **Monitor Syscalls** — captures and examines system calls made by the virtual machine
* 📁 **Monitor Files** — monitors file system activity, including file access and modifications
* 📈 **Monitor Network** — analyzes network traffic and activity
* 🚨 **Detect Anomalies** — identifies unusual patterns and behavior in the collected data using machine learning algorithms, including IsolationForest, RandomForestClassifier, and OneClassSVM
* 📊 **Analyze Behavior** — provides in-depth analysis of the virtual machine's behavior, including data visualization using matplotlib and plotly
* 📄 **Generate Report** — creates interactive reports based on the collected and analyzed data, providing a comprehensive overview of the virtual machine's activity

---

## 🚀 Getting Started

### Prerequisites
To run the VMI project, you will need to have Python installed on your system, along with the necessary dependencies. The required dependencies include asyncio, logging, traceback, json, pyyaml, libvmi, sentry-sdk, tenacity, numpy, pandas, scikit-learn, tensorflow, matplotlib, scipy, plotly, tkinterdnd2, and Pillow.

### Installation
```bash
pip install -r requirements.txt
```

### Quick Start
```bash
python main.py
```

## 📖 Usage
The VMI project provides a comprehensive analysis and reporting tool. Here are a few examples of how to use it:
* Run the project using `python main.py` to launch the GUI and start analyzing data.
* Use the `analysis_modules.py` file to create custom analysis modules and integrate them into the project.
* Utilize the `reporting_module.py` file to generate reports based on the analysis results, which can be visualized using the `matplotlib` and `plotly` libraries.
* Integrate with the `libvmi` library to collect and analyze data from virtual machines, and use the `sentry-sdk` library to track and report errors.

---

## 📁 Project Structure
```
VMI/
# requirements.txt: lists all project dependencies
# main.py: the main entry point of the application
# analysis_modules.py: contains functions for monitoring and analyzing system activity
# gui_module.py: handles the graphical user interface
# reporting_module.py: generates reports based on collected data
# vmi_integration.py: integrates with the VMI system
```

## 🛠️ Tech Stack
| Technology | Version | Purpose |
|-----------|---------|---------|
| asyncio | latest | asynchronous operations |
| logging | latest | logging framework |
| traceback | latest | error handling |
| json | latest | data serialization |
| pyyaml | latest | configuration file parsing |
| libvmi | latest | VMI integration |
| sentry-sdk | latest | error tracking |
| tenacity | latest | retry mechanism |
| numpy | latest | numerical computations |
| pandas | latest | data manipulation |
| scikit-learn | latest | machine learning |
| tensorflow | latest | machine learning |
| matplotlib | latest | data visualization |
| scipy | latest | scientific computations |
| plotly | latest | data visualization |
| tkinterdnd2 | latest | drag-and-drop functionality |
| Pillow | latest | image processing |

## ⚙️ Configuration
| Variable | Description | Required |
|----------|-------------|----------|
| log_file | path to the log file | optional |
---

## ⚠️ Documentation Drift Detected

LiveDocAI detected that the documentation may be outdated based on recent code changes:

> The latest commit added a new requirements.txt file with a list of dependencies, but the existing README/docs do not reflect these changes, indicating a potential drift between the documentation and the actual code.

*This documentation was automatically regenerated to reflect the latest code.*

---

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is open source. See the repository for license details.

---

<div align="center">

**[⬆ Back to Top](#)**

*Documentation auto-generated by [LiveDocAI](https://github.com) — Production-Aware API Intelligence Tool*
*Commit: `eac8bfe`*

</div>