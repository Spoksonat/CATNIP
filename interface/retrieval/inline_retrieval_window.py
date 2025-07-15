import customtkinter as ctk
from tkinter import Toplevel, Label
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
import sys
from interface.help_window import HelpWindow
from retrieval_methods.Class_Inline_Retrieval import Inline
import threading
from utils.plots import Plots

class InlineRetrievalWindow(ctk.CTk):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.title("Inline Parameters")
        self.geometry("650x420")  # Increased height to accommodate everything

        self.plots = Plots()

        ctk.CTkButton(self, text="← Back", width=70, command=self.go_back).place(x=10, y=10)

        # --- Section: Retrieval Parameters ---
        ctk.CTkLabel(self, text="Specify retrieval parameters", font=("Segoe UI", 14)).pack(pady=(50, 10))

        param_frame = ctk.CTkFrame(self, fg_color="transparent")
        param_frame.pack(pady=(0, 20))

        # Correct Damaged Pixels
        correct_frame = ctk.CTkFrame(param_frame, fg_color="transparent")
        correct_frame.grid(row=0, column=0, padx=10)
        ctk.CTkLabel(correct_frame, text="Correct Damaged Pixels").grid(row=0, column=0, sticky="w")
        self.correct_dropdown = ctk.CTkOptionMenu(correct_frame, values=["Yes", "No"])
        self.correct_dropdown.set("No")
        self.correct_dropdown.grid(row=1, column=0)
        ctk.CTkButton(correct_frame, text="?", width=30, height=30, corner_radius=15,
                      command=lambda: self.show_help(f"Correct")).grid(row=1, column=1, padx=(0, 5))

        # Save as NPY
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

        correct = self.correct_dropdown.get()
        save = self.save_dropdown.get()

        dict_params = {"Param": 0}

        file_path_ref = filedialog.askopenfilename(title="Select Reference Image", filetypes=[("npy files", "*.npy"), ("All files", "*.*")])
        file_path_samp = filedialog.askopenfilename(title="Select Sample Image", filetypes=[("npy files", "*.npy"), ("All files", "*.*")])

        Iref = np.load(file_path_ref)
        Isamp = np.load(file_path_samp)

        self.progress_bar.start()
    
        def task():
            try:
                retrieved = Inline(I_ref=Iref, I_samp=Isamp, dict_params=dict_params)
                # Now update the plot in the main thread
                def show_image():
                    n_subplots = (1,1)
                    plot_size = (600, 600)
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

                    fig.canvas.manager.set_window_title("Retrieved Images")
                    plt.show()
    
                self.after(0, show_image)
                if(save == "Yes"):
                    folder_path_save = filedialog.askdirectory(title="Select Folder to Save")
                    dict_retrieval = {"T": retrieved.T}
                    np.save(folder_path_save + "/Inline_images.npy", dict_retrieval)
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