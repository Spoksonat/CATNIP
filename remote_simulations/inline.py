import numpy as np
import matplotlib.pyplot as plt
from simulation_scripts.poly_sim import Poly_simulationInline
import json
import sys


def run_Inline_sim(dict_params, path):

    folder_path = "temp"
    if not folder_path:
        return  # If user cancels dialog

    print("Running simulation...")
    
    poly_sim = Poly_simulationInline(dict_params=dict_params, save_path=folder_path)
    I_ref_poly = np.zeros((int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
    I_samp_poly = np.zeros((int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
    
    for i, E in enumerate(poly_sim.Es):
        print(f"Obtaining reference and sample images for Energy = {E} keV")
        I_ref, I_samp = poly_sim.single_sim(E=E, theta_y=1e-10)
        I_ref_poly += poly_sim.S[i]*I_ref
        I_samp_poly += poly_sim.S[i]*I_samp
    
    np.save(path + "/I_ref_poly.npy", I_ref_poly)
    np.save(path + "/I_samp_poly.npy", I_samp_poly)
    
    print("Simulation finished.")

if len(sys.argv) > 1:
    path = sys.argv[1]

with open(path + "/temp/Param_Card.json") as f:
    dict_params = json.load(f)

run_Inline_sim(dict_params=dict_params, path=path)