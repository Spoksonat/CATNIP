import numpy as np
import matplotlib.pyplot as plt
from simulation_scripts.poly_sim import Poly_simulationInline
import json
import sys


def run_Inline_sim(dict_params, path):

    initial_angle = float(dict_params["Initial angle (deg)"])
    if(initial_angle == 0):
        initial_angle += 1e-10
    final_angle = float(dict_params["Final angle (deg)"])
    N_of_proj = int(float(dict_params["Number of projections"]))
    ct_angles = np.linspace(initial_angle, final_angle, N_of_proj, endpoint=False)
    folder_path = "temp"
    if not folder_path:
        return  # If user cancels dialog
    

    print("Running simulation...")
                
    poly_sim = Poly_simulationInline(dict_params=dict_params, save_path=folder_path)
    I_ref_poly_ct = np.zeros((N_of_proj, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
    I_samp_poly_ct = np.zeros((N_of_proj, int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
                
    for i_ang, theta_y in enumerate(ct_angles):
        print(f"Obtaining reference and sample images for CT angle = {theta_y} deg")
        I_ref_poly = np.zeros((int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
        I_samp_poly = np.zeros((int(poly_sim.img_size[0]/poly_sim.binning_factor), int(poly_sim.img_size[1]/poly_sim.binning_factor)))
        for i, E in enumerate(poly_sim.Es):
            print(f"Obtaining reference and sample images for Energy = {E} keV")
            I_ref, I_samp = poly_sim.single_sim(E=E, theta_y=theta_y)
            I_ref_poly += poly_sim.S[i]*I_ref
            I_samp_poly += poly_sim.S[i]*I_samp
        I_ref_poly_ct[i_ang,:,:] = I_ref_poly
        I_samp_poly_ct[i_ang,:,:] = I_samp_poly
                
    np.save(path + "/temp/I_ref_poly_ct.npy", I_ref_poly_ct)
    np.save(path + "/temp/I_samp_poly_ct.npy", I_samp_poly_ct)
                
    print("Simulation finished.")

if len(sys.argv) > 1:
    path = sys.argv[1]

with open(path + "/temp/Param_Card.json") as f:
    dict_params = json.load(f)

run_Inline_sim(dict_params=dict_params, path=path)
