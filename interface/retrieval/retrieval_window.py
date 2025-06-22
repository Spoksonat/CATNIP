import customtkinter as ctk
from tkinter import Toplevel, Label
import numpy as np
from tkinter import filedialog
import sys
from interface.help_window import HelpWindow

class RetrievalWindow(ctk.CTk):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.title("Select Retrieval Algorithm")
        self.geometry("650x300")

        ctk.CTkButton(self, text="← Back", width=70, command=self.go_back).place(x=10, y=10)
        ctk.CTkLabel(self, text="Please select your retrieval algorithm", font=("Segoe UI", 16)).pack(pady=(50, 20))

        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.pack()

        ctk.CTkButton(button_frame, text="UMPA", width=100, command=self.open_umpa_window).pack(side="left", padx=15)
        #for text in ["MIST"]:
        #    ctk.CTkButton(button_frame, text=text, width=100).pack(side="left", padx=15)

        ctk.CTkButton(button_frame, text="LCS", width=100, command=self.open_lcs_window).pack(side="left", padx=15)
        ctk.CTkButton(button_frame, text="EI", width=100, command=self.open_ei_window).pack(side="left", padx=15)
        

        ctk.CTkButton(self, text="?", width=30, height=30, corner_radius=15,
                      command=lambda: HelpWindow(self, "retrieval")).place(x=10, y=260)

        self.mainloop()

    def go_back(self):
        self.destroy()
        self.controller.show_main_window()

    def open_ei_window(self):
        self.destroy()
        self.controller.show_ei_retrieval_window()

    def open_lcs_window(self):
        self.destroy()
        self.controller.show_lcs_retrieval_window()

    def open_umpa_window(self):
        self.destroy()
        self.controller.show_umpa_retrieval_window()

    def on_close(self):
        self.destroy()     # destroys the window
        sys.exit(0)        # exits the whole program