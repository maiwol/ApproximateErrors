import sys
import functions as fun
import Simulation_classes_cleaning_notOO as sim
import Script2 as scr
import FT_quantum_circuits.steane as st
import FT_quantum_circuits.correction as cor
from visualizer import browser_vis as brow


n_data = 7
n_ancilla = 7
n_qubits = 14

circ, bs = cor.Steane_Correct.FT_encoded_zero_Steane()

##pre_outcomes = [['0','1'] for i in range(7)]
##outcomes = [pre_outcomes]

circ_sim = sim.State_and_Operations(circ, n_qubits)
brow.from_circuit(circ, True)
