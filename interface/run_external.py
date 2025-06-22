import customtkinter as ctk
from tkinter import Toplevel, Label
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
import sys
from interface.help_window import HelpWindow
import threading
from utils.plots import Plots
import paramiko
import os
import sys
import time

class RunExternalWindow(ctk.CTk):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.title("Run in Server")
        self.geometry("650x420")  # Increased height to accommodate everything

        self.plots = Plots()

        ctk.CTkButton(self, text="← Back", width=70, command=self.go_back).place(x=10, y=10)

        # --- Section: Retrieval Parameters ---
        ctk.CTkLabel(self, text="Specify connection parameters", font=("Segoe UI", 14)).pack(pady=(50, 10))

        param_frame = ctk.CTkFrame(self, fg_color="transparent")
        param_frame.pack(pady=(0, 20))

        # User
        user_frame = ctk.CTkFrame(param_frame, fg_color="transparent")
        user_frame.grid(row=0, column=0, padx=10)
        ctk.CTkLabel(user_frame, text="User").grid(row=0, column=0, sticky="w")
        self.user_entry = ctk.CTkEntry(user_frame, width=200)
        self.user_entry.grid(row=1, column=0)
        ctk.CTkButton(user_frame, text="?", width=30, height=30, corner_radius=15,
                      command=lambda: self.show_help(f"User")).grid(row=1, column=1, padx=(0, 5))
        
        # Server
        server_frame = ctk.CTkFrame(param_frame, fg_color="transparent")
        server_frame.grid(row=0, column=1, padx=10)
        ctk.CTkLabel(server_frame, text="Server").grid(row=0, column=0, sticky="w")
        self.server_entry = ctk.CTkEntry(server_frame, width=300)
        self.server_entry.grid(row=1, column=0)
        ctk.CTkButton(server_frame, text="?", width=30, height=30, corner_radius=15,
                      command=lambda: self.show_help(f"Server")).grid(row=1, column=1, padx=(0, 5))
        
        param2_frame = ctk.CTkFrame(self, fg_color="transparent")
        param2_frame.pack(pady=(0, 20))

        # Password
        password_frame = ctk.CTkFrame(param2_frame, fg_color="transparent")
        password_frame.grid(row=0, column=0, padx=10)
        ctk.CTkLabel(password_frame, text="Password").grid(row=0, column=0, sticky="w")
        self.password_entry = ctk.CTkEntry(password_frame, width=200, show="*")
        self.password_entry.grid(row=1, column=0)
        ctk.CTkButton(password_frame, text="?", width=30, height=30, corner_radius=15,
                      command=lambda: self.show_help(f"Password")).grid(row=1, column=1, padx=(0, 5))

        # --- Section: Load Options ---

        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.pack()
        ctk.CTkButton(button_frame, text="Run simulation in server", width=150, command=self.run_in_server).pack(side="left", padx=20)

        # General help button (bottom left)
        ctk.CTkButton(self, text="?", width=30, height=30, corner_radius=15,
                      command=lambda: HelpWindow(self, "retrieval")).place(x=10, y=380)

        self.mainloop()

    def go_back(self):
        self.destroy()
        self.controller.show_main_window()

    def run_in_server(self):
        remote_user = self.user_entry.get()
        home_remote = f"/home/{remote_user}"
        remote_host = self.server_entry.get()
        remote_path = home_remote + "/remote_project"
        local_path = "remote_simulations"
        entrypoint = self.controller.ext_script #"ct_inline.py"
        logfile = "log.txt"
        password = self.password_entry.get()  # Optional: avoid by setting up SSH keys

        dict_server = {"remote_user": remote_user,
                       "home_remote": home_remote,
                       "remote_host": remote_host,
                       "remote_path": remote_path,
                       "local_path": local_path,
                       "entrypoint": entrypoint,
                       "logfile": logfile,
                       "password": password}
        
        np.save("remote_simulations/temp/dict_server.npy", dict_server)
        self.controller.tab1_by_default = False
        self.controller.run_ext_sim = True
        self.destroy()
        if(self.controller.ext_script == "ei.py"):
            self.controller.show_ei_window()
        elif(self.controller.ext_script == "inline.py"):
            self.controller.show_inline_window()
        elif(self.controller.ext_script == "sbi.py"):
            self.controller.show_sbi_window()
        elif(self.controller.ext_script == "sgbi.py"):
            self.controller.show_sgbi_window()
        elif(self.controller.ext_script == "ct_inline.py"):
            self.controller.show_ct_inline_window()
        elif(self.controller.ext_script == "ct_sbi.py"):
            self.controller.show_ct_sbi_window()
        elif(self.controller.ext_script == "ct_sgbi.py"):
            self.controller.show_ct_sgbi_window()
        elif(self.controller.ext_script == "ct_ei.py"):
            self.controller.show_ct_ei_window()
        else:
            raise ValueError("Invalid option")

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