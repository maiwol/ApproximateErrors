import sys
import functions as fun
import Script2 as scr
import faultTolerant.steane as st
import scriptsIARPA.correction as cor
from visualizer import browser_vis as brow


# input parameters
n_data = 7      # number data qubits
n_ancilla = 4   # number of ancilla qubits
stabilizers = st.Code.stabilizer   # stabilizers
X_stabs = stabilizers[:3]
Z_stabs = stabilizers[3:]
redundancy = 3    # number of repetitions
alternating = True    # not important    
with_initial_I = True    # whether or not we want Identity
                         # gates at the beginning 
error_gate = 'AD 0.1'    # what error we want to add
faulty_gates = ['I','H','CX','CZ']   # what gates we'll add
                                     # the error to
error_kind = 3   # kind of error: play with this (1, 2, or 3)




### Circuit 1: X stabilizer of the Steane code repeated 3 times.

circ1 = scr.create_FT_EC_no_time(n_data, n_ancilla, X_stabs, redundancy,
                                alternating, with_initial_I)
#fun.insert_errors(circ1, error_gate, faulty_gates, error_kind)
#brow.from_circuit(circ1, True)


### Circuit 2: FT prep of a logical |0>.

circ2, meas_gates =  cor.Steane_Correct.FT_encoded_zero_Steane()
fun.insert_errors(circ2, error_gate, faulty_gates, error_kind)
brow.from_circuit(circ2, True)
