import customtkinter as ctk
from tkinter import Toplevel, Label
import numpy as np
from tkinter import filedialog
import sys
from interface.help_window import HelpWindow

class SimulationsWindow(ctk.CTk):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.title("Select Technique")
        self.geometry("600x250")

        ctk.CTkButton(self, text="← Back", width=70, command=self.go_back).place(x=10, y=10)
        ctk.CTkLabel(self, text="Please select your technique", font=("Segoe UI", 16)).pack(pady=(50, 20))

        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.pack()

        ctk.CTkButton(button_frame, text="Inline", width=100, command=self.open_inline_window).pack(side="left", padx=20)
        ctk.CTkButton(button_frame, text="SBI", width=100, command=self.open_sbi_window).pack(side="left", padx=20)
        ctk.CTkButton(button_frame, text="SGBI", width=100, command=self.open_sgbi_window).pack(side="left", padx=20)
        ctk.CTkButton(button_frame, text="EI", width=100, command=self.open_ei_window).pack(side="left", padx=20)
        

        ctk.CTkButton(self, text="?", width=30, height=30, corner_radius=15,
                      command=lambda: HelpWindow(self, "simulations")).place(x=10, y=210)

        self.mainloop()

    def go_back(self):
        self.destroy()
        self.controller.show_main_window()

    def open_ei_window(self):
        self.destroy()
        self.controller.show_ei_window()

    def open_sbi_window(self):
        self.destroy()
        self.controller.show_sbi_window()

    def open_sgbi_window(self):
        self.destroy()
        self.controller.show_sgbi_window()

    def open_inline_window(self):
        self.destroy()
        self.controller.show_inline_window()

    def on_close(self):
        self.destroy()     # destroys the window
        sys.exit(0)        # exits the whole program
