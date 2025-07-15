import json
import numpy as np
from tkinter import filedialog, messagebox, Toplevel, Label
import customtkinter as ctk
import matplotlib.pyplot as plt
import threading
import sys
import paramiko
import os
import time

def define_param_dict_general(window):
    """
    Defines and populates the parameter dictionary for the simulation window from all input widgets.

    Args:
        window: The main window object containing all parameter widgets.

    Returns:
        None
    """
    window.dict_params = {}
    
    for key, widget in window.source_dropdowns:
        window.dict_params[key] = widget.get()
    for key, widget in window.source_textboxes:
        window.dict_params[key] = widget.get()
    
    for key, widget in window.det_dropdowns:
        window.dict_params[key] = widget.get()
    for key, widget in window.det_textboxes:
        window.dict_params[key] = widget.get()

    if(window.grat == True):
        for key, widget in window.grat_dropdowns:
            window.dict_params[key] = widget.get()
        for key, widget in window.grat_textboxes:
            window.dict_params[key] = widget.get()

    for key, widget in window.samp_dropdowns:
        window.dict_params[key] = widget.get()
    if(window.extra_sites > 0):
        for key, widget in window.samp_textboxes[:-window.extra_sites]:
            window.dict_params[key] = widget.get()
        for key, widget in window.samp_textboxes[-window.extra_sites:]:
            window.dict_params[key.cget("text")] = widget.get()
    else:
        for key, widget in window.samp_textboxes:
            window.dict_params[key] = widget.get()

    if(window.DF == True):
        for key, widget in window.DF_dropdowns:
            window.dict_params[key] = widget.get()
        for key, widget in window.DF_textboxes:
            window.dict_params[key] = widget.get()

    for key, widget in window.geom_dropdowns:
        window.dict_params[key] = widget.get()
    for key, widget in window.geom_textboxes:
        window.dict_params[key] = widget.get()

    if(window.CT == True):
        for key, widget in window.ct_dropdowns:
            window.dict_params[key] = widget.get()
        for key, widget in window.ct_textboxes:
            window.dict_params[key] = widget.get()

def load_parameters_general(window, ext_sim=False):
    """
    Loads simulation parameters from a file or remote JSON and populates the window widgets.

    Args:
        window: The main window object containing all parameter widgets.
        ext_sim (bool, optional): If True, loads parameters from remote simulation.

    Returns:
        None
    """
    if(ext_sim == False):
        file_path = filedialog.askopenfilename(
        title="Select NumPy File",
        filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")]
        )
        # Load the file (if a path was selected)
        if file_path:
            data = np.load(file_path, allow_pickle=True).item() # To read a dict inside an npy file
    else:
        with open("remote_simulations/temp/Param_Card.json") as f:
            data = json.load(f)
            
    for key, widget in window.source_dropdowns:
        widget.set(data[key])
    for key, widget in window.source_textboxes:
        widget.delete(0, 'end')     # Clear previous content
        widget.insert(0, data[key])  # Insert new text

    for key, widget in window.det_dropdowns:
        widget.set(data[key])
    for key, widget in window.det_textboxes:
        widget.delete(0, 'end')     # Clear previous content
        widget.insert(0, data[key])  # Insert new text

    if(window.grat == True):
        for key, widget in window.grat_dropdowns:
            widget.set(data[key])
        for key, widget in window.grat_textboxes:
            widget.delete(0, 'end')     # Clear previous content
            widget.insert(0, data[key])  # Insert new text

    for key, widget in window.samp_dropdowns:
        widget.set(data[key])
        window.on_samp_geometry_change(choice=data["Geometry"])
    if(window.extra_sites > 0):
        for key, widget in window.samp_textboxes[:-window.extra_sites]:
            widget.delete(0, 'end')     # Clear previous content
            widget.insert(0, data[key])  # Insert new text
        for key, widget in window.samp_textboxes[-window.extra_sites:]:
            widget.delete(0, 'end')     # Clear previous content
            widget.insert(0, data[key.cget("text")])  # Insert new text
    else:
        for key, widget in window.samp_textboxes:
            widget.delete(0, 'end')     # Clear previous content
            widget.insert(0, data[key])  # Insert new text

    if(window.DF == True):
        for key, widget in window.DF_dropdowns:
            widget.set(data[key])
        for key, widget in window.DF_textboxes:
            widget.delete(0, 'end')     # Clear previous content
            widget.insert(0, data[key])  # Insert new text

    for key, widget in window.geom_dropdowns:
        widget.set(data[key])
    for key, widget in window.geom_textboxes:
        widget.delete(0, 'end')     # Clear previous content
        widget.insert(0, data[key])  # Insert new text

    if(window.CT == True):
        for key, widget in window.ct_dropdowns:
            widget.set(data[key])
        for key, widget in window.ct_textboxes:
            widget.delete(0, 'end')     # Clear previous content
            widget.insert(0, data[key])  # Insert new text

    window.define_param_dict()

def on_back_general(window):
    """
    Handles the event for returning to the main simulation window.

    Args:
        window: The main window object.

    Returns:
        None
    """
    window.destroy()
    window.controller.show_simulation_window()

def on_back_ct_general(window):
    """
    Handles the event for returning to the CT simulation window.

    Args:
        window: The main window object.

    Returns:
        None
    """
    window.destroy()
    window.controller.show_ct_simulation_window()

def general_help_general(window):
    """
    Displays a help window with general instructions for the EI technique configuration.

    Args:
        window: The main window object.

    Returns:
        None
    """
    help_win = Toplevel(window)
    help_win.title("General Help")
    label = Label(help_win, text=(
        "This window allows you to configure parameters for EI technique.\n\n"
        "- Simulation Parameters: Configure simulation-specific options.\n"
        "- Detector Parameters: Configure detector-specific options.\n\n"
        "Use '?' buttons next to each element for detailed help."
    ), justify="left", padx=10, pady=10)
    label.pack()

def show_help_general(window, message):
    """
    Displays a help window with a custom message.

    Args:
        window: The main window object.
        message (str): Help message to display.

    Returns:
        None
    """
    help_win = Toplevel(window)
    help_win.title("Help")
    label = Label(help_win, text=message, justify="left", padx=10, pady=10)
    label.pack()

def save_parameters_general(window):
    """
    Saves the current parameter dictionary to a .npy file.

    Args:
        window: The main window object.

    Returns:
        None
    """
    file_path = filedialog.asksaveasfilename(
        defaultextension=".npy",
        filetypes=[("NumPy files", "*.npy")],
        title="Save Param Card"
    )

    if file_path:
        window.define_param_dict()
        np.save(file_path, window.dict_params)

def update_parameters_general(window):
    """
    Updates the parameter dictionary and displays a confirmation window.

    Args:
        window: The main window object.

    Returns:
        None
    """
    window.define_param_dict()

    update_win = Toplevel(window)
    update_win.title("Update")
    label = Label(update_win, text="Parameters were succesfully updated", justify="left", padx=10, pady=10)
    label.pack()

def on_spectrum_type_change_general(window, choice):
    """
    Handles changes in the spectrum type (Mono/Poly) and updates the UI accordingly.

    Args:
        window: The main window object.
        choice (str): Selected spectrum type.

    Returns:
        None
    """
    if choice == "Mono":
        window.energy_label.configure(text="Energy (keV)")
        if window.browse_btn:
            window.browse_btn.destroy()
            window.browse_btn = None
        if window.show_btn:
            window.show_btn.destroy()
            window.show_btn = None
        window.energy_entry.configure(state="normal")
    
    elif choice == "Poly":
        window.energy_label.configure(text="Spectrum file")
        window.energy_entry.delete(0, "end")
        window.energy_entry.configure(state="readonly")
    
        if not window.browse_btn:
            window.browse_btn = ctk.CTkButton(window.energy_label_frame, text="📂", width=30, height=25,
                                            command=window.browse_spectrum_file)
            window.browse_btn.pack(side="left", padx=(0, 5))
    
        if not window.show_btn:
            window.show_btn = ctk.CTkButton(window.energy_label_frame, text="Show", width=45, height=25,
                                              command=window.show_spectrum_file)
            window.show_btn.pack(side="left", padx=(0, 5))  

def on_samp_geometry_change_general(window, choice):
    """
    Handles changes in the sample geometry and updates the UI to show relevant extra parameters.

    Args:
        window: The main window object.
        choice (str): Selected sample geometry.

    Returns:
        None
    """
    n_rows = 4
    if(window.extra_sites > 0):
        last_textboxes = window.samp_textboxes[-window.extra_sites:]
        for label, textbox in last_textboxes:
            label.grid_forget()
            textbox.grid_forget()
        window.samp_textboxes = window.samp_textboxes[:-window.extra_sites]
            
    if choice == "Mammo Phantom":
        extra_params = ["Sample 2 material", "Sample 2 material density (g/cc)", "Diameter uC (μm)", "Radius pentagon (μm)", "Thickness wax (mm)"]
        window.extra_sites = len(extra_params)    
    elif choice == "block":
        extra_params =  ["Block thickness (mm)"]
        window.extra_sites = len(extra_params)
    elif choice == "angio tube":
        extra_params =  ["Sample 2 material", "Sample 2 material density (g/cc)", "External diameter (mm)", "Internal diameter (mm)"]
        window.extra_sites = len(extra_params)
    elif choice == "sphere":
        extra_params =  ["Sphere diameter (mm)"]
        window.extra_sites = len(extra_params)
    elif choice == "fibre":
        extra_params =  ["Fibre diameter (mm)"]
        window.extra_sites = len(extra_params)
    elif choice == "wedge":
        extra_params =  ["Wedge thickness (mm)"]
        window.extra_sites = len(extra_params)

    start_index = len(window.samp_textboxes) + len(window.samp_dropdowns)
        
    for i, param in enumerate(extra_params):
        col = (start_index + i) % window.n_cols
        row = ((start_index + i) // window.n_cols) * n_rows + 1
    
        label = ctk.CTkLabel(window.samp_frame, text=param)
        label.grid(row=row, column=col*2, sticky="w", padx=(0, 5))
    
        textbox = ctk.CTkEntry(window.samp_frame)
        textbox.grid(row=row+1, column=col*2, sticky="we", padx=(0, 5), pady=(0, 10))
    
        window.samp_textboxes.append((label, textbox))

def browse_spectrum_file_general(window):
    """
    Opens a file dialog to select a spectrum file and updates the energy entry widget.

    Args:
        window: The main window object.

    Returns:
        None
    """
    file_path = filedialog.askopenfilename(title="Select Spectrum File", filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
    if file_path:
        window.energy_entry.configure(state="normal")
        window.energy_entry.delete(0, "end")
        window.energy_entry.insert(0, file_path)
        window.energy_entry.configure(state="readonly")

def show_spectrum_file_general(window):
    """
    Loads and displays the selected spectrum file in a plot.

    Args:
        window: The main window object.

    Returns:
        None
    """
    path = window.energy_entry.get()
    if not path:
        messagebox.showinfo("No file", "No spectrum file selected.")
        return
    try:
        # Load the two-column data
        data = np.loadtxt(path)
    
        if data.ndim != 2 or data.shape[1] != 2:
            messagebox.showerror("Invalid format", "File must contain exactly two columns.")
            return
    
        x, y = data[:, 0], data[:, 1]/np.sum(data[:, 1])

        dict_plot_params = {"n_subplots": (1,1), "plot_size": (418.18, 418.18), "titles": ["Spectrum vs Energy"], "plots_space": (0.05, 0.05), "labels": [""], "axis_labels":("Normalized Intensity (a.u.)", "Energy (keV)")}

        fig = window.plots.layout_bars(xdatas=[x], ydatas=[y], dict_params=dict_plot_params)
        fig.canvas.manager.set_window_title("Spectrum")

        plt.show()

    except Exception as e:
        messagebox.showerror("Error", f"Could not read or plot file:\n{str(e)}")

def add_source_section_general(window, parent):
    """
    Adds the source parameters section to the UI.

    Args:
        window: The main window object.
        parent: The parent UI element to attach the section to.

    Returns:
        None
    """
    tab1 = parent
    # === Source Parameters Section ===
    window.source_frame = ctk.CTkFrame(tab1)
    window.source_frame.pack(pady=(10, 20), fill="x", padx=20)
    
    source_label = ctk.CTkLabel(window.source_frame, text="Source Parameters", font=ctk.CTkFont(size=16, weight="bold"))
    source_label.grid(row=0, column=0, columnspan=6, pady=(0, 10))
    
    window.source_params = ["Type of spectrum", "Energy (keV)", "Focal spot size (μm)", "Source geometry"]
    window.source_dropdowns = []
    window.source_textboxes = []
    
    for i in range(len(window.source_params)):
        n_rows = 2

        col = i % window.n_cols
        row = (i // window.n_cols) * n_rows + 1
    

        if(window.source_params[i] == "Type of spectrum" or window.source_params[i] == "Source geometry"):
            label = ctk.CTkLabel(window.source_frame, text=window.source_params[i])
            label.grid(row=row, column=col*2, sticky="w", padx=(0, 5))
            if(window.source_params[i] == "Type of spectrum"):
                dropdown = ctk.CTkComboBox(window.source_frame, values=["Mono", "Poly"])
                dropdown.set("Mono")
                dropdown.configure(command=window.on_spectrum_type_change)
                #window.source_dropdowns.append((source_params[i], dropdown))
                dropdown.grid(row=row+1, column=col*2, sticky="we", padx=(0, 5), pady=(0, 10))
            elif(window.source_params[i] == "Source geometry"):
                dropdown = ctk.CTkComboBox(window.source_frame, values=["Parallel", "Cone"])

            dropdown.grid(row=row+1, column=col*2, sticky="we", padx=(0, 5), pady=(0, 10))
            window.source_dropdowns.append((window.source_params[i], dropdown))

        elif(window.source_params[i] not in ["Type of spectrum", "Source geometry", "Focal spot size (μm)"]):
            window.energy_label_frame = ctk.CTkFrame(window.source_frame, fg_color="transparent")
            window.energy_label_frame.grid(row=row, column=col*2, columnspan=3, sticky="w")
            
            window.energy_label = ctk.CTkLabel(window.energy_label_frame, text="Energy (keV)")
            window.energy_label.pack(side="left", padx=(0, 5))

            window.energy_entry = ctk.CTkEntry(window.source_frame)
            window.energy_entry.grid(row=row+1, column=col*2, sticky="we", padx=(0, 5), pady=(0, 10))
            window.source_textboxes.append((window.source_params[i], window.energy_entry))
            
            window.browse_btn = None
            window.show_btn = None
        else:
            label = ctk.CTkLabel(window.source_frame, text=window.source_params[i])
            label.grid(row=row, column=col*2, sticky="w", padx=(0, 5))
    
            textbox = ctk.CTkEntry(window.source_frame)
            textbox.grid(row=row+1, column=col*2, sticky="we", padx=(0, 5), pady=(0, 10))
            window.source_textboxes.append((window.source_params[i], textbox))
    
        help_btn = ctk.CTkButton(window.source_frame, text="?", width=20, height=20, corner_radius=10,
                                     command=lambda p=window.source_params[i]: window.show_help(f"Dropdown for {p}"))
        help_btn.grid(row=row+1, column=col*2+1, sticky="w", padx=(0, 10), pady=(0, 10))

def add_det_section_general(window, parent):
    """
    Adds the detector parameters section to the UI.

    Args:
        window: The main window object.
        parent: The parent UI element to attach the section to.

    Returns:
        None
    """
    tab1 = parent
    # === Detector Parameters Section ===
    window.det_frame = ctk.CTkFrame(tab1)
    window.det_frame.pack(pady=(10, 20), fill="x", padx=20)
    
    det_label = ctk.CTkLabel(window.det_frame, text="Detector Parameters", font=ctk.CTkFont(size=16, weight="bold"))
    det_label.grid(row=0, column=0, columnspan=6, pady=(0, 10))
    
    det_params = ["Sim. pixel (μm)", "FOV (pix)", "Num. events per pixel", "Binning factor", "FWHM PSF (pix)"]
    window.det_dropdowns = []
    window.det_textboxes = []
    
    for i in range(len(det_params)):
        n_rows = 2

        col = i % window.n_cols
        row = (i // window.n_cols) * n_rows + 1
    
        label = ctk.CTkLabel(window.det_frame, text=det_params[i])
        label.grid(row=row, column=col*2, sticky="w", padx=(0, 5))
    
        textbox = ctk.CTkEntry(window.det_frame)
        textbox.grid(row=row+1, column=col*2, sticky="we", padx=(0, 5), pady=(0, 10))
        window.det_textboxes.append((det_params[i], textbox))
    
        help_btn = ctk.CTkButton(window.det_frame, text="?", width=20, height=20, corner_radius=10,
                                     command=lambda p=det_params[i]: window.show_help(f"Textbox for {p}"))
        help_btn.grid(row=row+1, column=col*2+1, sticky="w", padx=(0, 10), pady=(0, 10))

def add_grat_section_general(window, parent, grat_params):
    """
    Adds the grating parameters section to the UI.

    Args:
        window: The main window object.
        parent: The parent UI element to attach the section to.
        grat_params (list): List of grating parameter names.

    Returns:
        None
    """
    tab1 = parent
    # === Grating Parameters Section ===
    window.grat_frame = ctk.CTkFrame(tab1)
    window.grat_frame.pack(pady=(10, 20), fill="x", padx=20)
    
    grat_label = ctk.CTkLabel(window.grat_frame, text="Grating Parameters", font=ctk.CTkFont(size=16, weight="bold"))
    grat_label.grid(row=0, column=0, columnspan=6, pady=(0, 10))
    
    window.grat_dropdowns = []
    window.grat_textboxes = []
    
    for i in range(len(grat_params)):
        n_rows = 3
        col = i % window.n_cols
        row = (i // window.n_cols) * n_rows + 1
    
        label = ctk.CTkLabel(window.grat_frame, text=grat_params[i])
        label.grid(row=row, column=col*2, sticky="w", padx=(0, 5))
        
        textbox = ctk.CTkEntry(window.grat_frame)
        textbox.grid(row=row+1, column=col*2, sticky="we", padx=(0, 5), pady=(0, 10))
        window.grat_textboxes.append((grat_params[i], textbox))
        
        help_btn = ctk.CTkButton(window.grat_frame, text="?", width=20, height=20, corner_radius=10,
                                 command=lambda p=grat_params[i]: window.show_help(f"Textbox for {p}"))
        help_btn.grid(row=row+1, column=col*2+1, sticky="w", padx=(0, 10), pady=(0, 10))

def add_sample_section_general(windows, parent, CT=False):
    """
    Adds the sample parameters section to the UI.

    Args:
        windows: The main window object.
        parent: The parent UI element to attach the section to.
        CT (bool, optional): If True, configures for CT simulation.

    Returns:
        None
    """
    tab1 = parent
    # === Grating Parameters Section ===
    windows.samp_frame = ctk.CTkFrame(tab1)
    windows.samp_frame.pack(pady=(10, 20), fill="x", padx=20)
    
    samp_label = ctk.CTkLabel(windows.samp_frame, text="Sample Parameters", font=ctk.CTkFont(size=16, weight="bold"))
    samp_label.grid(row=0, column=0, columnspan=6, pady=(0, 10))
    
    windows.samp_params = ["Geometry", "Whole sample thickness (mm)", "Sample material", "Sample material density (g/cc)", "Background material", "Background material density (g/cc)"]
    if(CT == False):
        samp_options = ["Mammo Phantom", "block", "angio tube", "sphere", "fibre", "wedge"]
    else:
        samp_options = ["Mammo Phantom", "block", "angio tube", "sphere", "fibre"]
    windows.samp_dropdowns = []
    windows.samp_textboxes = []
    windows.len_extra = 0
    
    for i in range(len(windows.samp_params)):
        n_rows = 4

        col = i % windows.n_cols
        row = (i // windows.n_cols) * n_rows + 1
    
        label = ctk.CTkLabel(windows.samp_frame, text=windows.samp_params[i])
        label.grid(row=row, column=col*2, sticky="w", padx=(0, 5))
            
        if(windows.samp_params[i] not in ["Geometry"]):
            textbox = ctk.CTkEntry(windows.samp_frame)
            textbox.grid(row=row+1, column=col*2, sticky="we", padx=(0, 5), pady=(0, 10))
            windows.samp_textboxes.append((windows.samp_params[i], textbox))
        else:
            if(windows.samp_params[i] != "Geometry"):
                dropdown = ctk.CTkComboBox(windows.samp_frame, values=["Element", "Compound"])
            else:
                dropdown = ctk.CTkComboBox(windows.samp_frame, values=samp_options)
                dropdown.set("Mammo Phantom")
                dropdown.configure(command=windows.on_samp_geometry_change)

            dropdown.grid(row=row+1, column=col*2, sticky="we", padx=(0, 5), pady=(0, 10))
            windows.samp_dropdowns.append((windows.samp_params[i], dropdown))
    
        help_btn = ctk.CTkButton(windows.samp_frame, text="?", width=20, height=20, corner_radius=10,
                                 command=lambda p=windows.samp_params[i]: windows.show_help(f"Textbox for {p}"))
        help_btn.grid(row=row+1, column=col*2+1, sticky="w", padx=(0, 10), pady=(0, 10))

    extra_params = ["Sample 2 material", "Sample 2 material density (g/cc)", "Diameter uC (μm)", "Radius pentagon (μm)", "Thickness wax (mm)"]
    windows.extra_sites = len(extra_params)
    start_index = len(windows.samp_textboxes) + len(windows.samp_dropdowns)
    for i, param in enumerate(extra_params):
        col = (start_index + i) % windows.n_cols
        row = ((start_index + i) // windows.n_cols) * n_rows + 1
        label = ctk.CTkLabel(windows.samp_frame, text=param)
        label.grid(row=row, column=col*2, sticky="w", padx=(0, 5))
        textbox = ctk.CTkEntry(windows.samp_frame)
        textbox.grid(row=row+1, column=col*2, sticky="we", padx=(0, 5), pady=(0, 10))
        windows.samp_textboxes.append((label, textbox))

def add_DF_section_general(window, parent):
    """
    Adds the dark-field parameters section to the UI.

    Args:
        window: The main window object.
        parent: The parent UI element to attach the section to.

    Returns:
        None
    """
    tab1 = parent
    # === Detector Parameters Section ===
    window.DF_frame = ctk.CTkFrame(tab1)
    window.DF_frame.pack(pady=(10, 20), fill="x", padx=20)
    
    DF_label = ctk.CTkLabel(window.DF_frame, text="Dark-Field Parameters", font=ctk.CTkFont(size=16, weight="bold"))
    DF_label.grid(row=0, column=0, columnspan=6, pady=(0, 10))
    
    DF_params = ["RMS scattering angle in X (μrad)", "RMS scattering angle in Y (μrad)"]
    window.DF_dropdowns = []
    window.DF_textboxes = []
    
    for i in range(len(DF_params)):
        n_rows = 1

        col = i % window.n_cols
        row = (i // window.n_cols) * n_rows + 1
    
        label = ctk.CTkLabel(window.DF_frame, text=DF_params[i])
        label.grid(row=row, column=col*2, sticky="w", padx=(0, 5))
    
        textbox = ctk.CTkEntry(window.DF_frame)
        textbox.grid(row=row+1, column=col*2, sticky="we", padx=(0, 5), pady=(0, 10))
        window.DF_textboxes.append((DF_params[i], textbox))
    
        help_btn = ctk.CTkButton(window.DF_frame, text="?", width=20, height=20, corner_radius=10,
                                     command=lambda p=DF_params[i]: window.show_help(f"Textbox for {p}"))
        help_btn.grid(row=row+1, column=col*2+1, sticky="w", padx=(0, 10), pady=(0, 10))

def add_geometry_section_general(window, parent, geom_params):
    """
    Adds the setup geometrical parameters section to the UI.

    Args:
        window: The main window object.
        parent: The parent UI element to attach the section to.
        geom_params (list): List of geometry parameter names.

    Returns:
        None
    """
    tab1 = parent
    # === Detector Parameters Section ===
    window.geom_frame = ctk.CTkFrame(tab1)
    window.geom_frame.pack(pady=(10, 20), fill="x", padx=20)
    
    geom_label = ctk.CTkLabel(window.geom_frame, text="Setup Geometrical Parameters", font=ctk.CTkFont(size=16, weight="bold"))
    geom_label.grid(row=0, column=0, columnspan=6, pady=(0, 10))
    
    window.geom_dropdowns = []
    window.geom_textboxes = []
    
    for i in range(len(geom_params)):
        n_rows = 2

        col = i % window.n_cols
        row = (i // window.n_cols) * n_rows + 1
    
        label = ctk.CTkLabel(window.geom_frame, text=geom_params[i])
        label.grid(row=row, column=col*2, sticky="w", padx=(0, 5))
    
        textbox = ctk.CTkEntry(window.geom_frame)
        textbox.grid(row=row+1, column=col*2, sticky="we", padx=(0, 5), pady=(0, 10))
        window.geom_textboxes.append((geom_params[i], textbox))
    
        help_btn = ctk.CTkButton(window.geom_frame, text="?", width=20, height=20, corner_radius=10,
                                     command=lambda p=geom_params[i]: window.show_help(f"Textbox for {p}"))
        help_btn.grid(row=row+1, column=col*2+1, sticky="w", padx=(0, 10), pady=(0, 10))

def add_ct_section_general(window, parent):
    """
    Adds the CT parameters section to the UI.

    Args:
        window: The main window object.
        parent: The parent UI element to attach the section to.

    Returns:
        None
    """
    tab1 = parent
    # === Detector Parameters Section ===
    window.ct_frame = ctk.CTkFrame(tab1)
    window.ct_frame.pack(pady=(10, 20), fill="x", padx=20)
    
    ct_label = ctk.CTkLabel(window.ct_frame, text="Setup CT Parameters", font=ctk.CTkFont(size=16, weight="bold"))
    ct_label.grid(row=0, column=0, columnspan=6, pady=(0, 10))
    
    ct_params = ["Initial angle (deg)", "Final angle (deg)", "Number of projections"]
    window.ct_dropdowns = []
    window.ct_textboxes = []
    
    for i in range(len(ct_params)):
        n_rows = 1

        col = i % window.n_cols
        row = (i // window.n_cols) * n_rows + 1
    
        label = ctk.CTkLabel(window.ct_frame, text=ct_params[i])
        label.grid(row=row, column=col*2, sticky="w", padx=(0, 5))
    
        textbox = ctk.CTkEntry(window.ct_frame)
        textbox.grid(row=row+1, column=col*2, sticky="we", padx=(0, 5), pady=(0, 10))
        window.ct_textboxes.append((ct_params[i], textbox))
    
        help_btn = ctk.CTkButton(window.ct_frame, text="?", width=20, height=20, corner_radius=10,
                                     command=lambda p=ct_params[i]: window.show_help(f"Textbox for {p}"))
        help_btn.grid(row=row+1, column=col*2+1, sticky="w", padx=(0, 10), pady=(0, 10))

def run_ext_sim_general(window):
    """
    Runs an external simulation on a remote server via SSH and SFTP, uploading only changed files and launching the script.

    Args:
        window: The main window object.

    Returns:
        None
    """
    dict_server = np.load("remote_simulations/temp/dict_server.npy", allow_pickle=True).item()

    remote_user = dict_server["remote_user"]
    home_remote = dict_server["home_remote"]
    remote_host = dict_server["remote_host"]
    remote_path = dict_server["remote_path"]
    local_path = dict_server["local_path"]
    entrypoint = dict_server["entrypoint"]
    logfile = dict_server["logfile"]
    password = dict_server["password"] # Optional: avoid by setting up SSH keys

    # Save redirection
    window.original_stdout = sys.stdout
    window.original_stderr = sys.stderr
    sys.stdout = window.stdout_redirector
    sys.stderr = window.stdout_redirector
    
    def task():
        try:
            # === 1. INIT SSH AND SFTP ===
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(hostname=remote_host, username=remote_user, password=password)
            sftp = ssh.open_sftp()
            
            # === 2. SMART UPLOAD DIR FUNCTION (skips unchanged files) ===
            def upload_dir(local_dir, remote_dir):
                try:
                    sftp.stat(remote_dir)
                except FileNotFoundError:
                    sftp.mkdir(remote_dir)
                
                for item in os.listdir(local_dir):
                    local_item = os.path.join(local_dir, item)
                    remote_item = os.path.join(remote_dir, item)
            
                    if os.path.isdir(local_item):
                        upload_dir(local_item, remote_item)
                    else:
                        try:
                            remote_attr = sftp.stat(remote_item)
                            local_mtime = os.path.getmtime(local_item)
                            remote_mtime = remote_attr.st_mtime
                            local_size = os.path.getsize(local_item)
                            remote_size = remote_attr.st_size
            
                            if abs(local_mtime - remote_mtime) < 1 and local_size == remote_size:
                                continue  # skip unchanged
                        except FileNotFoundError:
                            pass  # file doesn't exist remotely
            
                        sftp.put(local_item, remote_item)
                        remote_time = os.path.getmtime(local_item)
                        sftp.utime(remote_item, (remote_time, remote_time))  # match mtime
            
            print("⬆️  Uploading only changed files...")
            upload_dir(local_path, remote_path)
            print("✅ Upload complete.")
            
            
            # === 3. INSTALL DEPENDENCIES SEPARATELY ===
            print("📦 Installing dependencies remotely...")
            install_cmd = f"cd {remote_path} && pip3 install -r requirements.txt"
            stdin, stdout, stderr = ssh.exec_command(install_cmd, get_pty=True)
            stdout.channel.recv_exit_status()
            print(stdout.read().decode())
            print(stderr.read().decode())
            
            # === 4. RUN SCRIPT IN BACKGROUND (CORRECTED) ===
            print("🚀 Launching remote script in background...")
            run_cmd = (
                f"cd {remote_path} && "
                f"nohup python3 {entrypoint} {remote_path} > {logfile} 2>&1 </dev/null &"
            )
            stdin, stdout, stderr = ssh.exec_command(run_cmd)
            time.sleep(10)
            sys.exit(0)
            print(stderr.read().decode())
            print(stdout.read().decode())
                
            print("✅ Remote script started (detached).")
            window.controller.tab1_by_default = True
            window.controller.run_ext_sim = False
            window.controller.ext_script = ""
        finally:
            # Restore output and stop bar safely in main thread
            def finish():
                sys.stdout = window.original_stdout
                sys.stderr = window.original_stderr
            window.after(0, finish)
            
    # Start task in a thread
    threading.Thread(target=task).start()

