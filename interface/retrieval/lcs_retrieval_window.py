import customtkinter as ctk
from tkinter import Toplevel, Label
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
import sys
from interface.help_window import HelpWindow
from retrieval_methods.Class_LCS import LCS
from utils.plots import Plots

class LCSRetrievalWindow(ctk.CTk):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.title("EI Retrieval Parameters")
        self.geometry("650x420")  # Increased height to accommodate everything

        self.plots = Plots()

        ctk.CTkButton(self, text="← Back", width=70, command=self.go_back).place(x=10, y=10)

        # --- Section: Retrieval Parameters ---
        ctk.CTkLabel(self, text="Specify retrieval parameters", font=("Segoe UI", 14)).pack(pady=(50, 10))

        param_frame = ctk.CTkFrame(self, fg_color="transparent")
        param_frame.pack(pady=(0, 20))

        # Column 2: Correct Damaged Pixels
        correct_frame = ctk.CTkFrame(param_frame, fg_color="transparent")
        correct_frame.grid(row=0, column=0, padx=10)
        ctk.CTkLabel(correct_frame, text="Correct Damaged Pixels").grid(row=0, column=0, sticky="w")
        self.correct_dropdown = ctk.CTkOptionMenu(correct_frame, values=["Yes", "No"])
        self.correct_dropdown.set("No")
        self.correct_dropdown.grid(row=1, column=0)
        ctk.CTkButton(correct_frame, text="?", width=30, height=30, corner_radius=15,
                      command=lambda: self.show_help(f"Correct")).grid(row=1, column=1, padx=(0, 5))

        # Column 3: Save as NPY
        save_frame = ctk.CTkFrame(param_frame, fg_color="transparent")
        save_frame.grid(row=0, column=1, padx=10)
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

        correct = self.correct_dropdown.get()
        save = self.save_dropdown.get()

        dict_params = {"correct damaged": correct}

        file_path_ref = filedialog.askopenfilename(title="Select Reference Stack", filetypes=[("npy files", "*.npy"), ("All files", "*.*")])
        file_path_samp = filedialog.askopenfilename(title="Select Sample Stack", filetypes=[("npy files", "*.npy"), ("All files", "*.*")])

        Iref = np.load(file_path_ref)
        Isamp = np.load(file_path_samp)

        retrieved = LCS(I_refs=Iref, I_samps=Isamp, dict_params=dict_params)

        n_subplots = (1,3)
        plot_size = (900, 200)
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

        self.plots.show_image(retrieved.T, None, mark_bad, axes[0], fig, scalebar_color, scalebar_pad, scalebar_alpha, scalebar_fontsize, "Transmission", cbar_join)

        self.plots.show_image(retrieved.Dphi_x, None, mark_bad, axes[1], fig, scalebar_color, scalebar_pad, scalebar_alpha, scalebar_fontsize, "Differential phase in X", cbar_join)

        self.plots.show_image(retrieved.Dphi_y, None, mark_bad, axes[2], fig, scalebar_color, scalebar_pad, scalebar_alpha, scalebar_fontsize, "Differential phase in Y", cbar_join)

        fig.canvas.manager.set_window_title("Retrieved Images")
        plt.show()

        """
        fig = plt.figure(figsize=(18,4))
        ax = fig.add_subplot(131)
        plt.imshow(retrieved.T, cmap="gist_gray")
        plt.colorbar()
        plt.title("Retrieved Transmission")
 
        ax = fig.add_subplot(132)
        plt.imshow(retrieved.Dphi_x, cmap="gist_gray")
        plt.colorbar()
        plt.title("Differential phase in X")

        ax = fig.add_subplot(133)
        plt.imshow(retrieved.Dphi_y, cmap="gist_gray")
        plt.colorbar()
        plt.title("Differential phase in Y")

        plt.show()
        """

        if(save == "Yes"):
            folder_path_save = filedialog.askdirectory(title="Select Folder to Save")
            dict_retrieval = {"T": retrieved.T, "dx": retrieved.Dphi_x, "dy": retrieved.Dphi_y}
            np.save(folder_path_save + "/Multimodal_SBI_images.npy", dict_retrieval)

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