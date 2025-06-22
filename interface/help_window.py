import customtkinter as ctk
from tkinter import Toplevel, Label
import numpy as np
from tkinter import filedialog

class HelpWindow(ctk.CTkToplevel):
    def __init__(self, parent, context):
        super().__init__(parent)
        self.title("Help")
        self.geometry("400x300")
        self.resizable(False, False)

        help_texts = {
            "main": "Welcome to XRsim!\n\n"
                    "• Simulations: Choose a simulation technique like SBI, SGBI, or EI.\n"
                    "• Multimodal retrieval: Select an algorithm to extract contrast information.",

            "simulations": "Simulation Techniques:\n\n"
                           "• SBI: Single Grating Beam Interferometry.\n"
                           "• SGBI: Scanning Grating Beam Interferometry.\n"
                           "• EI: Edge Illumination technique.",

            "retrieval": "Retrieval Algorithms:\n\n"
                         "• UMPA: Unified Modulated Pattern Analysis.\n"
                         "• MIST: Multi-Image Spline Technique.\n"
                         "• LCS: Least-squares method for contrast separation.\n"
                         "• EI: Algorithm used for EI data.",

            "ei": "EI Tab Information:\n\n"
                  "Tab 1:\n• Select a parameter from the dropdown.\n• Enter related input in the text box.\n\n"
                  "Other tabs can be extended for configuration, visualization, or results."
        }

        explanation = help_texts.get(context, "No help available for this section.")
        label = ctk.CTkLabel(self, text=explanation, justify="left", wraplength=380)
        label.pack(padx=20, pady=20)