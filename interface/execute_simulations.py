from simulation_scripts.poly_sim import Poly_simulationEI
from simulation_scripts.poly_sim import Poly_simulationInline
from simulation_scripts.poly_sim import Poly_simulationSBI
from simulation_scripts.poly_sim import Poly_simulationSGBI
import numpy as np
import matplotlib.pyplot as plt
import threading
import csv
from tkinter import filedialog
import sys

#--------------------- EI Simulations ------------------------------

def show_alignment_EI(window):
    """
    Runs a single-step EI simulation for alignment and displays a histogram and reference image.

    Args:
        window: Main application window containing parameters and plot utilities.

    Returns:
        None
    """
    window.define_param_dict()
    dict_params_new = window.dict_params.copy()
    dict_params_new["Number of steps"] = "1"
    dict_params_new["Type of spectrum"] = "Mono"
    dict_params_new["Energy (keV)"] = "21.0"

    window.progress_bar_hist.start()
    
    def task():
        try:
            poly_sim = Poly_simulationEI(dict_params=dict_params_new, save_path=None)
            I_refs_poly = np.zeros((poly_sim.N, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
            I_samps_poly = np.zeros((poly_sim.N, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
    
            for i, E in enumerate(poly_sim.Es):
                print(f"Obtaining reference and sample images for Energy = {E} keV")
                I_refs, I_samps = poly_sim.single_sim(E=E)
                I_refs_poly += poly_sim.S[i]*I_refs
                I_samps_poly += poly_sim.S[i]*I_samps

            # Now update the plot in the main thread
            def show_histogram():
                image = I_refs_poly[0,:,:]
                odd_columns = image[:, 1::2]
                even_columns = image[:, 0::2]

                n_subplots = (1,2)
                plot_size = (900, 300)
                plots_space = (0.05, 0.05)
                mark_bad = False
                scalebar_color="white"
                scalebar_pad=0.01
                scalebar_alpha= 0.5
                scalebar_fontsize= 10
                cbar_join = False

                fig, axes = plt.subplots(n_subplots[0], n_subplots[1], figsize=(plot_size[0]/72, plot_size[1]/72), gridspec_kw={'wspace': plots_space[0],'hspace': plots_space[1]})  

                if(n_subplots[0]*n_subplots[1] != 1):
                    axes = axes.flatten()
                else:
                    axes = [axes]

                window.plots.show_hist(np.ravel(odd_columns), ax=axes[0], fig=fig, title="Odd and Even Pixels Histogram", label="Odd pixels")
                window.plots.show_hist(np.ravel(even_columns), ax=axes[0], fig=fig, title="Odd and Even Pixels Histogram", label="Even pixels")
                axes[0].set_xlabel("Intensity (a.u.)")

                window.plots.show_image(image, window.dict_params["Sim. pixel (μm)"], mark_bad, axes[1], fig, scalebar_color, scalebar_pad, scalebar_alpha, scalebar_fontsize, "Reference Image", cbar_join)

                fig.canvas.manager.set_window_title("Alignment Histogram")
                plt.show()
    
            window.after(0, show_histogram)
    
        finally:
            def finish():
                window.progress_bar_hist.stop()
            window.after(0, finish)
    
    threading.Thread(target=task).start()

def run_EI_sim(window):
    """
    Runs a full EI simulation and saves reference/sample images and configuration to disk.

    Args:
        window: Main application window containing parameters and plot utilities.

    Returns:
        None
    """
    window.define_param_dict()
    folder_path = filedialog.askdirectory(title="Select Folder")
    if not folder_path:
        return  # If user cancels dialog

    # Save redirection
    window.original_stdout = sys.stdout
    window.original_stderr = sys.stderr
    sys.stdout = window.stdout_redirector
    sys.stderr = window.stdout_redirector

    window.progress_bar.start()

    def task():
        try:
            print("Running simulation...")
    
            poly_sim = Poly_simulationEI(dict_params=window.dict_params, save_path=folder_path)
            I_refs_poly = np.zeros((poly_sim.N, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
            I_samps_poly = np.zeros((poly_sim.N, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
    
            for i, E in enumerate(poly_sim.Es):
                print(f"Obtaining reference and sample images for Energy = {E} keV")
                I_refs, I_samps = poly_sim.single_sim(E=E)
                I_refs_poly += poly_sim.S[i]*I_refs
                I_samps_poly += poly_sim.S[i]*I_samps

            np.save(poly_sim.save_path + "/I_refs_poly.npy", I_refs_poly)
            np.save(poly_sim.save_path + "/I_samps_poly.npy", I_samps_poly)

            print("Simulation finished.")

            sim_param = { 
                            "Energy in keV": window.dict_params["Energy (keV)"], 
                            "Num. events per pixel": window.dict_params["Num. events per pixel"],
                            "Pixel size in m": window.dict_params["Sim. pixel (μm)"], 
                            "Image size in pix": window.dict_params["FOV (pix)"],
                            "Grating period in X in um": window.dict_params["Period (μm)"],
                            "Grating fringe width in um": window.dict_params["Fringe width (μm)"],
                            "Grating thickness in um": window.dict_params["Grating thickness (μm)"],
                            "Grating material": window.dict_params["Material"],                     
                            "Sample geometry": window.dict_params["Geometry"],
                            "Sample material": window.dict_params["Sample material"],
                            "Sample thickness in mm": window.dict_params["Whole sample thickness (mm)"],
                            "Background material": window.dict_params["Background material"],
                            "Binning factor": window.dict_params["Binning factor"],
                            "Source-to-detector distance in m": window.dict_params["Source-Detector distance (m)"],
                            "Grating-to-Sample distance in m": window.dict_params["Grating-Sample distance (m)"],
                            "Shift grating in prop. axis (cm)": window.dict_params["Shift grating in prop. axis (cm)"],
                            "Shift grating lateral axis (μm)": window.dict_params["Shift grating lateral axis (μm)"],
                            "Number of steps": window.dict_params["Number of steps"],
                        }              
            with open(poly_sim.save_path + "/sim_config.csv",'w', 
                      newline="") as file:
                w = csv.writer(file)
                w.writerows(sim_param.items())

            np.save(poly_sim.save_path + "/Param_Card.npy", window.dict_params)
        finally:
            # Restore output and stop bar safely in main thread
            def finish():
                window.progress_bar.stop()
                sys.stdout = window.original_stdout
                sys.stderr = window.original_stderr
            window.after(0, finish)
    
    # Start task in a thread
    threading.Thread(target=task).start()

def run_CT_EI_sim(window):
    """
    Runs a CT EI simulation over a range of angles and saves results to disk.

    Args:
        window: Main application window containing parameters and plot utilities.

    Returns:
        None
    """
    window.define_param_dict()
    initial_angle = float(window.dict_params["Initial angle (deg)"])
    final_angle = float(window.dict_params["Final angle (deg)"])
    N_of_proj = int(float(window.dict_params["Number of projections"]))
    ct_angles = np.linspace(initial_angle, final_angle, N_of_proj, endpoint=False)
    folder_path = filedialog.askdirectory(title="Select Folder")
    if not folder_path:
        return  # If user cancels dialog
    
    # Save redirection
    window.original_stdout = sys.stdout
    window.original_stderr = sys.stderr
    sys.stdout = window.stdout_redirector
    sys.stderr = window.stdout_redirector
    
    window.progress_bar.start()
    
    def task():
        try:
            print("Running simulation...")
    
            poly_sim = Poly_simulationEI(dict_params=window.dict_params, save_path=folder_path)
            I_refs_poly_ct = np.zeros((N_of_proj, poly_sim.N, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
            I_samps_poly_ct = np.zeros((N_of_proj, poly_sim.N, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
    
            for i_ang, theta_y in enumerate(ct_angles):
                print(f"Obtaining reference and sample images for CT angle = {theta_y} deg")
                I_refs_poly = np.zeros((poly_sim.N, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
                I_samps_poly = np.zeros((poly_sim.N, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
                for i, E in enumerate(poly_sim.Es):
                    print(f"Obtaining reference and sample images for Energy = {E} keV")
                    I_refs, I_samps = poly_sim.single_sim(E=E, theta_y=theta_y)
                    I_refs_poly += poly_sim.S[i]*I_refs
                    I_samps_poly += poly_sim.S[i]*I_samps
                I_refs_poly_ct[i_ang,:,:] = I_refs_poly
                I_samps_poly_ct[i_ang,:,:] = I_samps_poly
    
            np.save(poly_sim.save_path + "/I_ref_poly_ct.npy", I_refs_poly_ct)
            np.save(poly_sim.save_path + "/I_samp_poly_ct.npy", I_samps_poly_ct)
            np.save(poly_sim.save_path + "/Param_Card.npy", window.dict_params)
    
            print("Simulation finished.")
        finally:
            # Restore output and stop bar safely in main thread
            def finish():
                window.progress_bar.stop()
                sys.stdout = window.original_stdout
                sys.stderr = window.original_stderr
            window.after(0, finish)
    
    # Start task in a thread
    threading.Thread(target=task).start()

#--------------------- Inline Simulations ------------------------------

def show_reference_Inline(window):
    """
    Runs a single-step Inline simulation and displays reference and sample images.

    Args:
        window: Main application window containing parameters and plot utilities.

    Returns:
        None
    """
    window.define_param_dict()
    dict_params_new = window.dict_params.copy()
    dict_params_new["Type of spectrum"] = "Mono"
    dict_params_new["Energy (keV)"] = "21.0"
    window.progress_bar_hist.start()
    
    def task():
        try:
            poly_sim = Poly_simulationInline(dict_params=dict_params_new, save_path=None)
            I_ref_poly = np.zeros((int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim. binning_factor)))
            I_samp_poly = np.zeros((int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim. binning_factor)))
    
            for i, E in enumerate(poly_sim.Es):
                print(f"Obtaining reference and sample images for Energy = {E} keV")
                I_ref, I_samp = poly_sim.single_sim(E=E)
                I_ref_poly += poly_sim.S[i]*I_ref
                I_samp_poly += poly_sim.S[i]*I_samp
    
            # Now update the plot in the main thread
            def show_image():
    
                n_subplots = (1,2)
                plot_size = (300, 250)
                plots_space = (0.05, 0.05)
                mark_bad = False
                scalebar_color="white"
                scalebar_pad=0.01
                scalebar_alpha= 0.5
                scalebar_fontsize= 10
                cbar_join = False
                fig, axes = plt.subplots(n_subplots[0], n_subplots[1], figsize=(plot_size[0]/72, plot_size[1]/72),  gridspec_kw={'wspace': plots_space[0],'hspace': plots_space[1]})  
                if(n_subplots[0]*n_subplots[1] != 1):
                    axes = axes.flatten()
                else:
                    axes = [axes]
                window.plots.show_image(I_ref_poly, window.dict_params["Sim. pixel (μm)"], mark_bad, axes[0], fig,  scalebar_color, scalebar_pad, scalebar_alpha, scalebar_fontsize, "Reference Image", cbar_join)
                window.plots.show_image(I_samp_poly, window.dict_params["Sim. pixel (μm)"], mark_bad, axes[1], fig,  scalebar_color, scalebar_pad, scalebar_alpha, scalebar_fontsize, "Sample Image", cbar_join)
                fig.canvas.manager.set_window_title("Reference and Sample Images")
                plt.show()
    
            window.after(0, show_image)
    
        finally:
            def finish():
                window.progress_bar_hist.stop()
            window.after(0, finish)
    
    threading.Thread(target=task).start()

def run_Inline_sim(window):
    """
    Runs a full Inline simulation and saves reference/sample images to disk.

    Args:
        window: Main application window containing parameters and plot utilities.

    Returns:
        None
    """
    window.define_param_dict()
    folder_path = filedialog.askdirectory(title="Select Folder")
    if not folder_path:
        return  # If user cancels dialog
    
    # Save redirection
    window.original_stdout = sys.stdout
    window.original_stderr = sys.stderr
    sys.stdout = window.stdout_redirector
    sys.stderr = window.stdout_redirector
    
    window.progress_bar.start()
    
    def task():
        try:
            print("Running simulation...")
    
            poly_sim = Poly_simulationInline(dict_params=window.dict_params, save_path=folder_path)
            I_ref_poly = np.zeros((int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim. binning_factor)))
            I_samp_poly = np.zeros((int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim. binning_factor)))
    
            for i, E in enumerate(poly_sim.Es):
                print(f"Obtaining reference and sample images for Energy = {E} keV")
                I_ref, I_samp = poly_sim.single_sim(E=E)
                I_ref_poly += poly_sim.S[i]*I_ref
                I_samp_poly += poly_sim.S[i]*I_samp
    
            np.save(poly_sim.save_path + "/I_ref_poly.npy", I_ref_poly)
            np.save(poly_sim.save_path + "/I_samp_poly.npy", I_samp_poly)
    
            print("Simulation finished.")
        finally:
            # Restore output and stop bar safely in main thread
            def finish():
                window.progress_bar.stop()
                sys.stdout = window.original_stdout
                sys.stderr = window.original_stderr
            window.after(0, finish)
    
    # Start task in a thread
    threading.Thread(target=task).start()

def run_Inline_sim_ct(window):
    """
    Runs a CT Inline simulation over a range of angles and saves results to disk.

    Args:
        window: Main application window containing parameters and plot utilities.

    Returns:
        None
    """
    window.define_param_dict()
    initial_angle = float(window.dict_params["Initial angle (deg)"])
    final_angle = float(window.dict_params["Final angle (deg)"])
    N_of_proj = int(float(window.dict_params["Number of projections"]))
    ct_angles = np.linspace(initial_angle, final_angle, N_of_proj, endpoint=False)
    folder_path = filedialog.askdirectory(title="Select Folder")
    if not folder_path:
        return  # If user cancels dialog
    
    # Save redirection
    window.original_stdout = sys.stdout
    window.original_stderr = sys.stderr
    sys.stdout = window.stdout_redirector
    sys.stderr = window.stdout_redirector
    
    window.progress_bar.start()
    
    def task():
        try:
            print("Running simulation...")
    
            poly_sim = Poly_simulationInline(dict_params=window.dict_params, save_path=folder_path)
            I_ref_poly_ct = np.zeros((N_of_proj, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
            I_samp_poly_ct = np.zeros((N_of_proj, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
    
            for i_ang, theta_y in enumerate(ct_angles):
                print(f"Obtaining reference and sample images for CT angle = {theta_y} deg")
                I_ref_poly = np.zeros((int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
                I_samp_poly = np.zeros((int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
                for i, E in enumerate(poly_sim.Es):
                    print(f"Obtaining reference and sample images for Energy = {E} keV")
                    I_ref, I_samp = poly_sim.single_sim(E=E, theta_y=theta_y)
                    I_ref_poly += poly_sim.S[i]*I_ref
                    I_samp_poly += poly_sim.S[i]*I_samp
                I_ref_poly_ct[i_ang,:,:] = I_ref_poly
                I_samp_poly_ct[i_ang,:,:] = I_samp_poly
    
            np.save(poly_sim.save_path + "/I_ref_poly_ct.npy", I_ref_poly_ct)
            np.save(poly_sim.save_path + "/I_samp_poly_ct.npy", I_samp_poly_ct)
            np.save(poly_sim.save_path + "/Param_Card.npy", window.dict_params)
    
            print("Simulation finished.")
        finally:
            # Restore output and stop bar safely in main thread
            def finish():
                window.progress_bar.stop()
                sys.stdout = window.original_stdout
                sys.stderr = window.original_stderr
            window.after(0, finish)
    
    # Start task in a thread
    threading.Thread(target=task).start()

#--------------------- SBI Simulations ------------------------------

def show_speckles(window):
    """
    Runs a single-step SBI simulation and displays the reference speckle pattern.

    Args:
        window: Main application window containing parameters and plot utilities.

    Returns:
        None
    """
    window.define_param_dict()
    dict_params_new = window.dict_params.copy()
    dict_params_new["Number of steps"] = "1"
    dict_params_new["Type of spectrum"] = "Mono"
    dict_params_new["Energy (keV)"] = "21.0"
    window.progress_bar_hist.start()
    
    def task():
        try:
            poly_sim = Poly_simulationSBI(dict_params=dict_params_new, save_path=None)
            I_refs_poly = np.zeros((poly_sim.N, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
            I_samps_poly = np.zeros((poly_sim.N, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
    
            for i, E in enumerate(poly_sim.Es):
                print(f"Obtaining reference and sample images for Energy = {E} keV")
                I_refs, I_samps = poly_sim.single_sim(E=E)
                I_refs_poly += poly_sim.S[i]*I_refs
                I_samps_poly += poly_sim.S[i]*I_samps
    
            # Now update the plot in the main thread
            def show_image():
                image = I_refs_poly[0,:,:]
    
                n_subplots = (1,1)
                plot_size = (300, 250)
                plots_space = (0.05, 0.05)
                mark_bad = False
                scalebar_color="white"
                scalebar_pad=0.01
                scalebar_alpha= 0.5
                scalebar_fontsize= 10
                cbar_join = False
                fig, axes = plt.subplots(n_subplots[0], n_subplots[1], figsize=(plot_size[0]/72, plot_size[1]/72), gridspec_kw={'wspace': plots_space[0],'hspace': plots_space[1]})  
                if(n_subplots[0]*n_subplots[1] != 1):
                    axes = axes.flatten()
                else:
                    axes = [axes]
                window.plots.show_image(image, window.dict_params["Sim. pixel (μm)"], mark_bad, axes[0], fig, scalebar_color, scalebar_pad, scalebar_alpha, scalebar_fontsize, "Reference Image", cbar_join)
                fig.canvas.manager.set_window_title("Reference Speckle Pattern")
                plt.show()
    
            window.after(0, show_image)
    
        finally:
            def finish():
                window.progress_bar_hist.stop()
            window.after(0, finish)
    
    threading.Thread(target=task).start()

def run_SBI_sim(window):
    """
    Runs a full SBI simulation and saves reference/sample images to disk.

    Args:
        window: Main application window containing parameters and plot utilities.

    Returns:
        None
    """
    window.define_param_dict()
    folder_path = filedialog.askdirectory(title="Select Folder")
    if not folder_path:
        return  # If user cancels dialog
    
    # Save redirection
    window.original_stdout = sys.stdout
    window.original_stderr = sys.stderr
    sys.stdout = window.stdout_redirector
    sys.stderr = window.stdout_redirector
    
    window.progress_bar.start()
    
    def task():
        try:
            print("Running simulation...")
    
            poly_sim = Poly_simulationSBI(dict_params=window.dict_params, save_path=folder_path)
            I_refs_poly = np.zeros((poly_sim.N, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
            I_samps_poly = np.zeros((poly_sim.N, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
    
            for i, E in enumerate(poly_sim.Es):
                print(f"Obtaining reference and sample images for Energy = {E} keV")
                I_refs, I_samps = poly_sim.single_sim(E=E)
                I_refs_poly += poly_sim.S[i]*I_refs
                I_samps_poly += poly_sim.S[i]*I_samps
    
            np.save(poly_sim.save_path + "/I_refs_poly.npy", I_refs_poly)
            np.save(poly_sim.save_path + "/I_samps_poly.npy", I_samps_poly)
    
            print("Simulation finished.")
        finally:
            # Restore output and stop bar safely in main thread
            def finish():
                window.progress_bar.stop()
                sys.stdout = window.original_stdout
                sys.stderr = window.original_stderr
            window.after(0, finish)
    
    # Start task in a thread
    threading.Thread(target=task).start()

def run_CT_SBI_sim(window):
    """
    Runs a CT SBI simulation over a range of angles and saves results to disk.

    Args:
        window: Main application window containing parameters and plot utilities.

    Returns:
        None
    """
    window.define_param_dict()
    initial_angle = float(window.dict_params["Initial angle (deg)"])
    final_angle = float(window.dict_params["Final angle (deg)"])
    N_of_proj = int(float(window.dict_params["Number of projections"]))
    ct_angles = np.linspace(initial_angle, final_angle, N_of_proj, endpoint=False)
    folder_path = filedialog.askdirectory(title="Select Folder")
    if not folder_path:
        return  # If user cancels dialog
    
    # Save redirection
    window.original_stdout = sys.stdout
    window.original_stderr = sys.stderr
    sys.stdout = window.stdout_redirector
    sys.stderr = window.stdout_redirector
    
    window.progress_bar.start()
    
    def task():
        try:
            print("Running simulation...")
    
            poly_sim = Poly_simulationSBI(dict_params=window.dict_params, save_path=folder_path)
            I_refs_poly_ct = np.zeros((N_of_proj, poly_sim.N, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
            I_samps_poly_ct = np.zeros((N_of_proj, poly_sim.N, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
    
            for i_ang, theta_y in enumerate(ct_angles):
                print(f"Obtaining reference and sample images for CT angle = {theta_y} deg")
                I_refs_poly = np.zeros((poly_sim.N, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
                I_samps_poly = np.zeros((poly_sim.N, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
                for i, E in enumerate(poly_sim.Es):
                    print(f"Obtaining reference and sample images for Energy = {E} keV")
                    I_refs, I_samps = poly_sim.single_sim(E=E, theta_y=theta_y)
                    I_refs_poly += poly_sim.S[i]*I_refs
                    I_samps_poly += poly_sim.S[i]*I_samps
                I_refs_poly_ct[i_ang,:,:] = I_refs_poly
                I_samps_poly_ct[i_ang,:,:] = I_samps_poly
    
            np.save(poly_sim.save_path + "/I_ref_poly_ct.npy", I_refs_poly_ct)
            np.save(poly_sim.save_path + "/I_samp_poly_ct.npy", I_samps_poly_ct)
            np.save(poly_sim.save_path + "/Param_Card.npy", window.dict_params)
    
            print("Simulation finished.")
        finally:
            # Restore output and stop bar safely in main thread
            def finish():
                window.progress_bar.stop()
                sys.stdout = window.original_stdout
                sys.stderr = window.original_stderr
            window.after(0, finish)
    
    # Start task in a thread
    threading.Thread(target=task).start()

#--------------------- SGBI Simulations ------------------------------

def show_reference_sgbi(window):
    """
    Runs a single-step SGBI simulation and displays the reference pattern.

    Args:
        window: Main application window containing parameters and plot utilities.

    Returns:
        None
    """
    window.define_param_dict()
    dict_params_new = window.dict_params.copy()
    dict_params_new["Num. of steps per dir."] = "1"
    dict_params_new["Type of spectrum"] = "Mono"
    dict_params_new["Energy (keV)"] = "21.0"
    window.progress_bar_hist.start()
    
    def task():
        try:
            poly_sim = Poly_simulationSGBI(dict_params=dict_params_new, save_path=None)
            I_refs_poly = np.zeros((poly_sim.N**2, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
            I_samps_poly = np.zeros((poly_sim.N**2, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
    
            for i, E in enumerate(poly_sim.Es):
                print(f"Obtaining reference and sample images for Energy = {E} keV")
                I_refs, I_samps = poly_sim.single_sim(E=E)
                I_refs_poly += poly_sim.S[i]*I_refs
                I_samps_poly += poly_sim.S[i]*I_samps
    
            # Now update the plot in the main thread
            def show_image():
                image = I_refs_poly[0,:,:]
    
                n_subplots = (1,1)
                plot_size = (300, 250)
                plots_space = (0.05, 0.05)
                mark_bad = False
                scalebar_color="white"
                scalebar_pad=0.01
                scalebar_alpha= 0.5
                scalebar_fontsize= 10
                cbar_join = False
                fig, axes = plt.subplots(n_subplots[0], n_subplots[1], figsize=(plot_size[0]/72, plot_size[1]/72), gridspec_kw={'wspace': plots_space[0],'hspace': plots_space[1]})  
                if(n_subplots[0]*n_subplots[1] != 1):
                    axes = axes.flatten()
                else:
                    axes = [axes]
                window.plots.show_image(image, window.dict_params["Sim. pixel (μm)"], mark_bad, axes[0], fig, scalebar_color, scalebar_pad, scalebar_alpha, scalebar_fontsize, "Reference Image", cbar_join)
                fig.canvas.manager.set_window_title("Reference Pattern")
                plt.show()
    
            window.after(0, show_image)
    
        finally:
            def finish():
                window.progress_bar_hist.stop()
            window.after(0, finish)
    
    threading.Thread(target=task).start()

def run_SGBI_sim(window):
    """
    Runs a full SGBI simulation and saves reference/sample images to disk.

    Args:
        window: Main application window containing parameters and plot utilities.

    Returns:
        None
    """
    window.define_param_dict()
    folder_path = filedialog.askdirectory(title="Select Folder")
    if not folder_path:
        return  # If user cancels dialog
    
    # Save redirection
    window.original_stdout = sys.stdout
    window.original_stderr = sys.stderr
    sys.stdout = window.stdout_redirector
    sys.stderr = window.stdout_redirector
    
    window.progress_bar.start()
    
    def task():
        try:
            print("Running simulation...")
    
            poly_sim = Poly_simulationSGBI(dict_params=window.dict_params, save_path=folder_path)
            I_refs_poly = np.zeros((poly_sim.N**2, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
            I_samps_poly = np.zeros((poly_sim.N**2, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
    
            for i, E in enumerate(poly_sim.Es):
                print(f"Obtaining reference and sample images for Energy = {E} keV")
                I_refs, I_samps = poly_sim.single_sim(E=E)
                I_refs_poly += poly_sim.S[i]*I_refs
                I_samps_poly += poly_sim.S[i]*I_samps
    
            np.save(poly_sim.save_path + "/I_refs_poly.npy", I_refs_poly)
            np.save(poly_sim.save_path + "/I_samps_poly.npy", I_samps_poly)
    
            print("Simulation finished.")
        finally:
            # Restore output and stop bar safely in main thread
            def finish():
                window.progress_bar.stop()
                sys.stdout = window.original_stdout
                sys.stderr = window.original_stderr
            window.after(0, finish)
    
    # Start task in a thread
    threading.Thread(target=task).start()

def run_CT_SGBI_sim(window):
    """
    Runs a CT SGBI simulation over a range of angles and saves results to disk.

    Args:
        window: Main application window containing parameters and plot utilities.

    Returns:
        None
    """
    window.define_param_dict()
    initial_angle = float(window.dict_params["Initial angle (deg)"])
    final_angle = float(window.dict_params["Final angle (deg)"])
    N_of_proj = int(float(window.dict_params["Number of projections"]))
    ct_angles = np.linspace(initial_angle, final_angle, N_of_proj, endpoint=False)
    folder_path = filedialog.askdirectory(title="Select Folder")
    if not folder_path:
        return  # If user cancels dialog
    
    # Save redirection
    window.original_stdout = sys.stdout
    window.original_stderr = sys.stderr
    sys.stdout = window.stdout_redirector
    sys.stderr = window.stdout_redirector
    
    window.progress_bar.start()
    
    def task():
        try:
            print("Running simulation...")
    
            poly_sim = Poly_simulationSGBI(dict_params=window.dict_params, save_path=folder_path)
            I_refs_poly_ct = np.zeros((N_of_proj, poly_sim.N**2, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
            I_samps_poly_ct = np.zeros((N_of_proj, poly_sim.N**2, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
    
            for i_ang, theta_y in enumerate(ct_angles):
                print(f"Obtaining reference and sample images for CT angle = {theta_y} deg")
                I_refs_poly = np.zeros((poly_sim.N**2, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
                I_samps_poly = np.zeros((poly_sim.N**2, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
                for i, E in enumerate(poly_sim.Es):
                    print(f"Obtaining reference and sample images for Energy = {E} keV")
                    I_refs, I_samps = poly_sim.single_sim(E=E, theta_y=theta_y)
                    I_refs_poly += poly_sim.S[i]*I_refs
                    I_samps_poly += poly_sim.S[i]*I_samps
                I_refs_poly_ct[i_ang,:,:] = I_refs_poly
                I_samps_poly_ct[i_ang,:,:] = I_samps_poly
    
            np.save(poly_sim.save_path + "/I_ref_poly_ct.npy", I_refs_poly_ct)
            np.save(poly_sim.save_path + "/I_samp_poly_ct.npy", I_samps_poly_ct)
            np.save(poly_sim.save_path + "/Param_Card.npy", window.dict_params)
    
            print("Simulation finished.")
        finally:
            # Restore output and stop bar safely in main thread
            def finish():
                window.progress_bar.stop()
                sys.stdout = window.original_stdout
                sys.stderr = window.original_stderr
            window.after(0, finish)
    
    # Start task in a thread
    threading.Thread(target=task).start()

