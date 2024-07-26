import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinterdnd2 import DND_FILES, TkinterDnD
import pandas as pd
import logging
import os
import threading
import asyncio
from reporting_module import visualize_anomalies
import plotly.express as px
from io import BytesIO

# Set up logging
logging.basicConfig(filename='gui_module.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AdvancedApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("Behavioral Analysis Tool")
        self.geometry("1200x900")
        self.create_widgets()
        self.data = None
        self.output_dir = "output_reports"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def create_widgets(self):
        # Notebook for tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Data Tab
        self.data_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.data_tab, text='Data Management')
        self.create_data_tab_widgets()

        # Analysis Tab
        self.analysis_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_tab, text='Data Analysis')
        self.create_analysis_tab_widgets()

    def create_data_tab_widgets(self):
        # Label
        tk.Label(self.data_tab, text="Drag and drop your data files here or use the button below:", font=("Helvetica", 14)).pack(pady=10)
        
        # Drop target
        self.drop_target = tk.Label(self.data_tab, text="Drop files here", bg="lightgrey", width=60, height=20)
        self.drop_target.pack(pady=20)
        self.drop_target.drop_target_register(DND_FILES)
        self.drop_target.dnd_bind('<<Drop>>', self.on_drop)

        # File Dialog Button
        tk.Button(self.data_tab, text="Select Files", command=self.load_data).pack(pady=10)
        
        # Data Summary
        self.summary_frame = ttk.Frame(self.data_tab)
        self.summary_frame.pack(pady=20)
        self.summary_label = tk.Label(self.summary_frame, text="No data loaded", font=("Helvetica", 12))
        self.summary_label.pack()

    def create_analysis_tab_widgets(self):
        # Status
        self.status_label = tk.Label(self.analysis_tab, text="Status: Ready", font=("Helvetica", 12))
        self.status_label.pack(pady=10)
        
        # Analyze Button
        tk.Button(self.analysis_tab, text="Analyze Data", command=self.start_analysis).pack(pady=20)
        
        # Progress Bar
        self.progress = ttk.Progressbar(self.analysis_tab, orient="horizontal", length=400, mode="determinate")
        self.progress.pack(pady=10)
        
        # Interactive Plot
        self.plot_frame = ttk.Frame(self.analysis_tab)
        self.plot_frame.pack(pady=20, fill=tk.BOTH, expand=True)

    def on_drop(self, event):
        files = self.tk.splitlist(event.data)
        if files:
            self.process_files(files)

    def load_data(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx")])
        if file_paths:
            self.process_files(file_paths)

    def process_files(self, file_paths):
        try:
            data_frames = []
            for file_path in file_paths:
                if file_path.endswith('.csv'):
                    data_frames.append(pd.read_csv(file_path))
                elif file_path.endswith('.xlsx'):
                    data_frames.append(pd.read_excel(file_path))
                else:
                    messagebox.showerror("File Format Error", f"Unsupported file format: {file_path}")
                    return
            self.data = pd.concat(data_frames, ignore_index=True)
            self.summary_label.config(text=f"Data Loaded: {len(self.data)} rows and {len(self.data.columns)} columns")
            self.status_label.config(text="Status: Data Loaded Successfully")
            logging.info("Data loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            messagebox.showerror("Error", f"An error occurred while loading the data: {e}")

    def start_analysis(self):
        if self.data is not None:
            self.status_label.config(text="Status: Analyzing data...")
            self.progress['value'] = 0
            asyncio.run(self.analyze_data())
        else:
            messagebox.showwarning("Warning", "No data loaded. Please load data before analysis.")

    async def analyze_data(self):
        try:
            # Example anomalies (replace with actual analysis)
            anomalies = {
                'iso_forest_anomalies': [1, 3, 5],
                'oc_svm_anomalies': [2, 4, 6],
                'autoencoder_anomalies': [0, 7, 8]
            }
            await asyncio.sleep(1)  # Simulate long computation
            
            visualize_anomalies(self.data, anomalies, self.output_dir)

            # Update progress and status
            self.status_label.config(text="Status: Analysis complete. Reports have been generated.")
            self.progress['value'] = 100
            logging.info("Analysis complete and reports generated.")
            self.update_plot()
        except Exception as e:
            logging.error(f"Error during analysis: {e}")
            messagebox.showerror("Error", f"An error occurred during the analysis: {e}")

    def update_plot(self):
        # Generate a plot using Plotly (example placeholder)
        fig = px.scatter(self.data, x=self.data.columns[0], y=self.data.columns[1], title="Data Visualization")
        buf = BytesIO()
        fig.write_image(buf, format='png')
        buf.seek(0)
        
        self.plot_img = tk.PhotoImage(data=buf.getvalue())
        self.plot_label = tk.Label(self.plot_frame, image=self.plot_img)
        self.plot_label.pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    app = AdvancedApp()
    app.mainloop()
