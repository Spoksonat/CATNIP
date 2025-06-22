import customtkinter as ctk
from tkinter import Toplevel, Label
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
import sys
from interface.help_window import HelpWindow
from retrieval_methods.Class_UMPA import UMPA
import threading
from utils.plots import Plots

class UMPARetrievalWindow(ctk.CTk):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.title("UMPA Parameters")
        self.geometry("650x420")  # Increased height to accommodate everything

        self.plots = Plots()

        ctk.CTkButton(self, text="← Back", width=70, command=self.go_back).place(x=10, y=10)

        # --- Section: Retrieval Parameters ---
        ctk.CTkLabel(self, text="Specify retrieval parameters", font=("Segoe UI", 14)).pack(pady=(50, 10))

        param_frame = ctk.CTkFrame(self, fg_color="transparent")
        param_frame.pack(pady=(0, 20))

        # Analysis Window Index (Nw)
        nw_frame = ctk.CTkFrame(param_frame, fg_color="transparent")
        nw_frame.grid(row=0, column=0, padx=10)
        ctk.CTkLabel(nw_frame, text="Analysis Window Index").grid(row=0, column=0, sticky="w")
        self.nw_entry = ctk.CTkEntry(nw_frame, width=100)
        self.nw_entry.grid(row=1, column=0)
        ctk.CTkButton(nw_frame, text="?", width=30, height=30, corner_radius=15,
                      command=lambda: self.show_help(f"Analysis Window Index (Nw)")).grid(row=1, column=1, padx=(0, 5))
        
        # Max Shift
        ms_frame = ctk.CTkFrame(param_frame, fg_color="transparent")
        ms_frame.grid(row=0, column=1, padx=10)
        ctk.CTkLabel(ms_frame, text="Max. Speckle Shift (pix)").grid(row=0, column=0, sticky="w")
        self.ms_entry = ctk.CTkEntry(ms_frame, width=100)
        self.ms_entry.grid(row=1, column=0)
        ctk.CTkButton(ms_frame, text="?", width=30, height=30, corner_radius=15,
                      command=lambda: self.show_help(f"Max. Speckle Shift (pix)")).grid(row=1, column=1, padx=(0, 5))
        
        # Dark Field Bool
        df_frame = ctk.CTkFrame(param_frame, fg_color="transparent")
        df_frame.grid(row=0, column=2, padx=10)
        ctk.CTkLabel(df_frame, text="Find Dark Field image?").grid(row=0, column=0, sticky="w")
        self.df_dropdown = ctk.CTkOptionMenu(df_frame, values=["Yes", "No"])
        self.df_dropdown.set("No")
        self.df_dropdown.grid(row=1, column=0)
        ctk.CTkButton(df_frame, text="?", width=30, height=30, corner_radius=15,
                      command=lambda: self.show_help(f"Find Dark Field image?")).grid(row=1, column=1, padx=(0, 5))

        # Correct Damaged Pixels
        correct_frame = ctk.CTkFrame(param_frame, fg_color="transparent")
        correct_frame.grid(row=1, column=0, padx=10)
        ctk.CTkLabel(correct_frame, text="Correct Damaged Pixels").grid(row=0, column=0, sticky="w")
        self.correct_dropdown = ctk.CTkOptionMenu(correct_frame, values=["Yes", "No"])
        self.correct_dropdown.set("No")
        self.correct_dropdown.grid(row=1, column=0)
        ctk.CTkButton(correct_frame, text="?", width=30, height=30, corner_radius=15,
                      command=lambda: self.show_help(f"Correct")).grid(row=1, column=1, padx=(0, 5))

        # Save as NPY
        save_frame = ctk.CTkFrame(param_frame, fg_color="transparent")
        save_frame.grid(row=1, column=1, padx=10)
        ctk.CTkLabel(save_frame, text="Save results as NPY").grid(row=0, column=0, sticky="w")
        self.save_dropdown = ctk.CTkOptionMenu(save_frame, values=["Yes", "No"])
        self.save_dropdown.set("Yes")
        self.save_dropdown.grid(row=1, column=0)
        ctk.CTkButton(save_frame, text="?", width=30, height=30, corner_radius=15,
                      command=lambda: self.show_help(f"Save")).grid(row=1, column=1, padx=(0, 5))

        # --- Section: Load Options ---
        ctk.CTkLabel(self, text="Select data loading method", font=("Segoe UI", 14)).pack(pady=(10, 10))

        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.pack()
        ctk.CTkButton(button_frame, text="Load from text files", width=150, command=self.load_from_txt).pack(side="left", padx=20)
        ctk.CTkButton(button_frame, text="Load from npy files", width=150, command=self.load_from_npy).pack(side="left", padx=20)

        # === Progress Bar Frame ===
        pb_frame = ctk.CTkFrame(self)
        pb_frame.pack(pady=(10, 20))

        self.progress_bar = ctk.CTkProgressBar(pb_frame, mode="indeterminate")
        self.progress_bar.pack(pady=10)
        self.progress_bar.stop()

        # General help button (bottom left)
        ctk.CTkButton(self, text="?", width=30, height=30, corner_radius=15,
                      command=lambda: HelpWindow(self, "retrieval")).place(x=10, y=380)

        self.mainloop()

    def go_back(self):
        self.destroy()
        self.controller.show_main_window()

    def load_from_txt(self):
        pass

    def load_from_npy(self):

        Nw = self.nw_entry.get()
        max_shift = self.ms_entry.get()
        if(self.df_dropdown.get() == "Yes"):
            df_bool = True
        else:
            df_bool = False
        correct = self.correct_dropdown.get()
        save = self.save_dropdown.get()

        dict_params = {"Nw": int(float(Nw)), "max_shift": int(float(max_shift)), "df_bool": df_bool ,"correct damaged": correct}

        file_path_ref = filedialog.askopenfilename(title="Select Reference Stack", filetypes=[("npy files", "*.npy"), ("All files", "*.*")])
        file_path_samp = filedialog.askopenfilename(title="Select Sample Stack", filetypes=[("npy files", "*.npy"), ("All files", "*.*")])

        Iref = np.load(file_path_ref)
        Isamp = np.load(file_path_samp)

        self.progress_bar.start()
    
        def task():
            try:
                retrieved = UMPA(I_refs=Iref, I_samps=Isamp, dict_params=dict_params)
                # Now update the plot in the main thread
                def show_image():
                    n_subplots = (2,2)
                    plot_size = (800, 600)
                    plots_space = (0.05, 0.2)
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

                    self.plots.show_image(retrieved.T, None, mark_bad, axes[0], fig, scalebar_color, scalebar_pad, scalebar_alpha, scalebar_fontsize, "Transmission", cbar_join)

                    self.plots.show_image(retrieved.D, None, mark_bad, axes[1], fig, scalebar_color, scalebar_pad, scalebar_alpha, scalebar_fontsize, "Dark Field", cbar_join, colormap=plt.cm.gist_yarg.copy())

                    self.plots.show_image(retrieved.dx, None, mark_bad, axes[2], fig, scalebar_color, scalebar_pad, scalebar_alpha, scalebar_fontsize, "Differential Phase in X", cbar_join)

                    self.plots.show_image(retrieved.dy, None, mark_bad, axes[3], fig, scalebar_color, scalebar_pad, scalebar_alpha, scalebar_fontsize, "Differential Phase in Y", cbar_join)

                    fig.canvas.manager.set_window_title("Retrieved Images")
                    plt.show()
                    """
                    fig = plt.figure(figsize=(8,8))
                    ax = fig.add_subplot(221)
                    plt.imshow(retrieved.T, cmap="gist_gray")
                    plt.colorbar()
                    plt.title("´Transmission")
             
                    ax = fig.add_subplot(222)
                    plt.imshow(retrieved.D, cmap="gist_gray")
                    plt.colorbar()
                    plt.title("Dark Field")
            
                    ax = fig.add_subplot(223)
                    plt.imshow(retrieved.dx, cmap="gist_gray")
                    plt.colorbar()
                    plt.title("Differential Phase in X")
            
                    ax = fig.add_subplot(224)
                    plt.imshow(retrieved.dy, cmap="gist_gray")
                    plt.colorbar()
                    plt.title("Differential Phase in Y")
            
                    plt.show()
                    """
    
                self.after(0, show_image)
                if(save == "Yes"):
                    folder_path_save = filedialog.askdirectory(title="Select Folder to Save")
                    dict_retrieval = {"T": retrieved.T, "dx": retrieved.dx, "dy": retrieved.dy, "D": retrieved.D}
                    np.save(folder_path_save + "/Multimodal_UMPA_images.npy", dict_retrieval)
            finally:
                def finish():
                    self.progress_bar.stop()
                self.after(0, finish)
    
        threading.Thread(target=task).start()

    def on_close(self):
        if(self.controller.flag_sim_to_ret == False):
            self.destroy()
            sys.exit(0)
        else:
            self.destroy()
            self.controller.flag_sim_to_ret = False

    def show_help(self, message):
        help_win = Toplevel(self)
        help_win.title("Help")
        label = Label(help_win, text=message, justify="left", padx=10, pady=10)
        label.pack()