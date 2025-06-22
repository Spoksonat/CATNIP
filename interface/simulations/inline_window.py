import customtkinter as ctk
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import sys 
from utils.plots import Plots
import json
from interface.common_functions import *
from interface.execute_simulations import *

class InlineWindow(ctk.CTk):

    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.title("Inline Options")
        self.geometry("1000x750")  # Ensure enough space for all widgets

        self.plots = Plots()

        self.spectrum_mode = None
        self.energy_label = None
        self.energy_entry = None
        self.browse_btn = None
        self.show_btn = None
        self.grat = False
        self.DF = False
        self.CT = False

        # Back button (outside tabs)
        back_btn = ctk.CTkButton(self, text="← Back", width=80, command=self.on_back)
        back_btn.pack(anchor="nw", padx=20, pady=(10, 0))

        # Tab view
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(expand=True, fill="both", padx=20, pady=10)

        self.tab1 = self.tabview.add("Parameters")
        self.tab2 = self.tabview.add("Run Simulation")

        self.setup_tab1()
        self.setup_tab2()

        if(self.controller.tab1_by_default == True):
            self.tabview.set("Parameters")
        else:
            self.tabview.set("Run Simulation")

        # Redirect stdout to this textbox
        self.redirect = TextRedirector(self.output_textbox)
        sys.stdout = self.redirect

        # Help button (bottom-left, circular, no frame)
        help_btn = ctk.CTkButton(self, text="?", width=35, height=35, corner_radius=50, command=self.general_help)
        help_btn.place(relx=0.0, rely=1.0, x=20, y=-20, anchor="sw")

        self.update()
        self.minsize(self.winfo_width(), self.winfo_height())

        if(self.controller.run_ext_sim == True):
            self.load_parameters(ext_sim=True)
            self.run_ext_sim()

    def setup_tab1(self):
        # === Make tab1 scrollable ===
        self.scroll_canvas = tk.Canvas(self.tab1, borderwidth=0, highlightthickness=0)
        self.scroll_canvas.pack(side="left", fill="both", expand=True)
    
        self.scrollbar = ctk.CTkScrollbar(self.tab1, orientation="vertical", command=self.scroll_canvas.yview)
        self.scrollbar.pack(side="right", fill="y")
    
        self.scroll_canvas.configure(yscrollcommand=self.scrollbar.set)
    
        # Scrollable content frame
        self.scrollable_frame = ctk.CTkFrame(self.scroll_canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.scroll_canvas.configure(
                scrollregion=self.scroll_canvas.bbox("all")
            )
        )
    
        self.scroll_canvas_window = self.scroll_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
    
        # Resize canvas window with widget
        def _on_frame_configure(event):
            self.scroll_canvas.itemconfig(self.scroll_canvas_window, width=self.scroll_canvas.winfo_width())
        self.scroll_canvas.bind("<Configure>", _on_frame_configure)
    
        # Enable mouse scroll
        #self.scrollable_frame.bind_all("<MouseWheel>", lambda e: self.scroll_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
    
        # === Replace tab1 with scrollable_frame ===
        tab1 = self.scrollable_frame  # use this as your layout parent from here on
        #tab1 = self.tab1 # For no scrolling, delete all the content above
        
        self.n_cols = 3 # Number of columns of tab1 layout

        geom_params = ["Source-Detector distance (m)", "Source-Sample distance (m)"]

        self.add_source_section(parent=tab1)
        self.add_det_section(parent=tab1)
        self.add_sample_section(parent=tab1)
        self.add_geometry_section(parent=tab1, geom_params=geom_params)
    
        for frame in [self.source_frame, self.det_frame, self.samp_frame, self.geom_frame]:
            for c in range(6):
                frame.grid_columnconfigure(c, weight=1) # Column expansion with window expansion
    
        # === Load Config, Save Config, Update Config, Show Setup and Show Alignment Buttons ===
        btn_frame = ctk.CTkFrame(tab1)
        btn_frame.pack(pady=(10, 20))

        # Load Config button 
        load_btn = ctk.CTkButton(btn_frame, text="Load Parameters", command=self.load_parameters)
        load_btn.pack(side="left", padx=(0, 10))

        # Save Config button 
        save_btn = ctk.CTkButton(btn_frame, text="Save Parameters", command=self.save_parameters)
        save_btn.pack(side="left", padx=(0, 10))

        # Save button 
        update_btn = ctk.CTkButton(btn_frame, text="Update Parameters", command=self.update_parameters)
        update_btn.pack(side="left", padx=(0, 10))

        # Show setup button 
        showstp_btn = ctk.CTkButton(btn_frame, text="Show Setup", command=self.show_setup)
        showstp_btn.pack(side="left", padx=(0, 10))

        # === Progress Bar Frame ===
        pb_frame = ctk.CTkFrame(tab1)
        pb_frame.pack(pady=(10, 20))

        self.progress_bar_hist = ctk.CTkProgressBar(pb_frame, mode="indeterminate")
        self.progress_bar_hist.pack(pady=10)
        self.progress_bar_hist.stop()

    def setup_tab2(self):
        # Big textbox for output
        self.output_textbox = tk.Text(self.tab2, wrap="word", height=30)
        self.output_textbox.pack(fill="both", expand=True, padx=20, pady=(10, 5))
        self.output_textbox.configure(state="disabled")  # Make it read-only
    
        # Redirect stdout to the textbox
        self.stdout_redirector = TextRedirector(self.output_textbox)

        # === Run and Retrieval Buttons ===
        btn_run_frame = ctk.CTkFrame(self.tab2)
        btn_run_frame.pack(pady=(10, 20))
        
        # Run button
        run_button = ctk.CTkButton(btn_run_frame, text="Run Inline Simulation", command=self.run_Inline_sim)
        run_button.pack(side="left", padx= 10, pady=(0, 10))

        # Run button
        run2_button = ctk.CTkButton(btn_run_frame, text="Run Inline Simulation on Server", command=self.run_Inline_sim_server)
        run2_button.pack(side="left", padx= 10, pady=(0, 10))

        # Run button
        retrieve1_button = ctk.CTkButton(btn_run_frame, text="Transmission Retrieval", command=self.go_to_inline_retrieval)
        retrieve1_button.pack(side="left", padx= 10, pady=(0, 10))

        self.progress_bar = ctk.CTkProgressBar(self.tab2, mode="indeterminate")
        self.progress_bar.pack(pady=10)
        self.progress_bar.stop()

    def on_back(self):
        on_back_general(self)

    def general_help(self):
        general_help_general(self)

    def show_help(self, message):
        show_help_general(self, message)

    def define_param_dict(self):
        define_param_dict_general(self)

    def load_parameters(self, ext_sim=False):
        load_parameters_general(self, ext_sim)
    
    def save_parameters(self):
        save_parameters_general(self)

    def update_parameters(self):
        update_parameters_general(self)

    def show_setup(self):
        self.define_param_dict()
        fig = self.plots.plot_setup_inline(dict_params=self.dict_params)
        fig.canvas.manager.set_window_title("Setup")
        plt.show()

    def on_close(self):
        self.destroy()     # destroys the window
        sys.exit(0)        # exits the whole program

    def on_spectrum_type_change(self, choice):
        on_spectrum_type_change_general(self, choice)

    def on_samp_geometry_change(self, choice):
        on_samp_geometry_change_general(self, choice)

    def browse_spectrum_file(self):
        browse_spectrum_file_general(self)
    
    def show_spectrum_file(self):
        show_spectrum_file_general(self)

    def add_source_section(self, parent):
        add_source_section_general(self, parent)

    def add_det_section(self, parent):
        add_det_section_general(self, parent)

    def add_sample_section(self, parent):
        add_sample_section_general(self, parent)

    def add_geometry_section(self, parent, geom_params):
        add_geometry_section_general(self, parent, geom_params)

    def run_Inline_sim(self):
        run_Inline_sim(self)

    def run_Inline_sim_server(self):
        self.define_param_dict()
        with open("remote_simulations/temp/Param_Card.json", "w") as f:
            json.dump(self.dict_params, f)
        
        self.controller.flag_sim_to_ret = True
        self.destroy()
        self.controller.ext_script = "inline.py"
        self.controller.show_ct_inline_window_server()

    def run_ext_sim(self):
        run_ext_sim_general(self)

    def go_to_inline_retrieval(self):
        self.controller.flag_sim_to_ret = True
        self.controller.show_inline_retrieval_window()

class TextRedirector:
    def __init__(self, text_widget, tag="stdout"):
        self.text_widget = text_widget
        self.tag = tag

    def write(self, str):
        self.text_widget.configure(state="normal")
        self.text_widget.insert("end", str, (self.tag,))
        self.text_widget.see("end")
        self.text_widget.configure(state="disabled")

    def flush(self):
        pass
