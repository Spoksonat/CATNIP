import customtkinter as ctk
from tkinter import Toplevel, Label
import numpy as np
from tkinter import filedialog
import sys
from interface.help_window import HelpWindow

class MainWindow(ctk.CTk):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.title("CATNIP")
        self.geometry("800x250")

        ctk.CTkLabel(self, text="Welcome to CATNIP", font=("Segoe UI", 20, "bold")).pack(pady=30)

        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.pack(pady=20)

        ctk.CTkButton(button_frame, text="2D Simulations", command=self.open_simulations, width=150).pack(side="left", padx=20)
        ctk.CTkButton(button_frame, text="CT Simulations", command=self.open_ct_simulations, width=150).pack(side="left", padx=20)
        ctk.CTkButton(button_frame, text="Multimodal retrieval", command=self.open_retrieval, width=150).pack(side="left", padx=20)

        ctk.CTkButton(self, text="?", width=30, height=30, corner_radius=15, command=lambda: HelpWindow(self, "main")).place(x=10, y=210)

    def open_simulations(self):
        self.destroy()
        self.controller.show_simulation_window()

    def open_ct_simulations(self):
        self.destroy()
        self.controller.show_ct_simulation_window()

    def open_retrieval(self):
        self.destroy()
        self.controller.show_retrieval_window()

    def on_close(self):
        self.destroy()     # destroys the window
        sys.exit(0)        # exits the whole program