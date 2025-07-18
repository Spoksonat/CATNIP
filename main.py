import customtkinter as ctk
from tkinter import Toplevel, Label
import numpy as np
from tkinter import filedialog
from interface.base_window import MainWindow
from interface.simulations.simulation_window import SimulationsWindow
from interface.CT_simulations.ct_simulation_window import CTSimulationsWindow
from interface.retrieval.retrieval_window import RetrievalWindow
from interface.retrieval.ei_retrieval_window import EIRetrievalWindow
from interface.retrieval.lcs_retrieval_window import LCSRetrievalWindow
from interface.retrieval.umpa_retrieval_window import UMPARetrievalWindow
from interface.retrieval.inline_retrieval_window import InlineRetrievalWindow
from interface.simulations.ei_window import EIWindow
from interface.simulations.sbi_window import SBIWindow
from interface.simulations.gbi_window import GBIWindow
from interface.simulations.inline_window import InlineWindow
from interface.CT_simulations.ct_inline_window import CTInlineWindow
from interface.CT_simulations.ct_sbi_window import CTSBIWindow
from interface.CT_simulations.ct_gbi_window import CTGBIWindow
from interface.CT_simulations.ct_ei_window import CTEIWindow
from interface.run_external import RunExternalWindow

class AppController:
    def __init__(self):
        self.flag_sim_to_ret = False
        self.tab1_by_default = True
        self.run_ext_sim = False
        self.ext_script = ""

        self.current_window = MainWindow(self)
        self.current_window.mainloop()

    def show_main_window(self):
        #self._close_current()
        self.current_window = MainWindow(self)
        self.current_window.mainloop()   

    def show_simulation_window(self):
        #self._close_current()
        self.current_window = SimulationsWindow(self)
        self.current_window.mainloop()

    def show_ct_simulation_window(self):
        #self._close_current()
        self.current_window = CTSimulationsWindow(self)
        self.current_window.mainloop()

    def show_retrieval_window(self):
        #self._close_current()
        self.current_window = RetrievalWindow(self)
        self.current_window.mainloop()

    def show_ei_retrieval_window(self):
        #self._close_current()
        self.current_window = EIRetrievalWindow(self)
        self.current_window.mainloop()

    def show_lcs_retrieval_window(self):
        #self._close_current()
        self.current_window = LCSRetrievalWindow(self)
        self.current_window.mainloop()

    def show_umpa_retrieval_window(self):
        #self._close_current()
        self.current_window = UMPARetrievalWindow(self)
        self.current_window.mainloop()

    def show_inline_retrieval_window(self):
        #self._close_current()
        self.current_window = InlineRetrievalWindow(self)
        self.current_window.mainloop()

    def show_ei_window(self):
        #self._close_current()
        self.current_window = EIWindow(self)
        self.current_window.mainloop()

    def show_sbi_window(self):
        #self._close_current()
        self.current_window = SBIWindow(self)
        self.current_window.mainloop()

    def show_GBI_window(self):
        #self._close_current()
        self.current_window = GBIWindow(self)
        self.current_window.mainloop()

    def show_inline_window(self):
        #self._close_current()
        self.current_window = InlineWindow(self)
        self.current_window.mainloop()

    def show_ct_inline_window(self):
        #self._close_current()
        self.current_window = CTInlineWindow(self)
        self.current_window.mainloop()

    def show_ct_sbi_window(self):
        #self._close_current()
        self.current_window = CTSBIWindow(self)
        self.current_window.mainloop()

    def show_ct_GBI_window(self):
        #self._close_current()
        self.current_window = CTGBIWindow(self)
        self.current_window.mainloop()

    def show_ct_ei_window(self):
        #self._close_current()
        self.current_window = CTEIWindow(self)
        self.current_window.mainloop()

    def show_ct_sbi_window_server(self):
        #self._close_current()
        self.current_window = RunExternalWindow(self)
        self.current_window.mainloop()

    def show_ct_inline_window_server(self):
        #self._close_current()
        self.current_window = RunExternalWindow(self)
        self.current_window.mainloop()

    def show_ct_GBI_window_server(self):
        #self._close_current()
        self.current_window = RunExternalWindow(self)
        self.current_window.mainloop()

    def show_ct_ei_window_server(self):
        #self._close_current()
        self.current_window = RunExternalWindow(self)
        self.current_window.mainloop()

    def show_ei_window_server(self):
        #self._close_current()
        self.current_window = RunExternalWindow(self)
        self.current_window.mainloop()

    def show_inline_window_server(self):
        #self._close_current()
        self.current_window = RunExternalWindow(self)
        self.current_window.mainloop()

    def show_sbi_window_server(self):
        #self._close_current()
        self.current_window = RunExternalWindow(self)
        self.current_window.mainloop()

    def show_GBI_window_server(self):
        #self._close_current()
        self.current_window = RunExternalWindow(self)
        self.current_window.mainloop()

    def _close_current(self):
        if self.current_window:
            self.current_window.destroy()

if __name__ == "__main__":
    app = AppController()
    app.show_simulation_window()