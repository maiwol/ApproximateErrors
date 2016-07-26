import numpy as np
import circuit as c
import X_Errors_2 as X
import Hadamard_Gates as H
import Z_Errors as Z
import CNOT_Circuit_Final as CNOT
import CZ_Circuit as CZ
import functions as fun
import Simulation_real_circuits as sim


class Efficient_State_and_Operations(object):
	## 1. Classify Gates
	## 2. Assign # Qubits
	## 3. Prep initial_state
	
	def __init__(self, circuit, number_qubits,
	             initial_state=None, desired_outcomes = [None]):
		print number_qubits
		classified_gates = self.classify_gates(circuit.gates)
		self.prep_gates, self.oper_gates, self.meas_gates = classified_gates 
		print 'Im done with classification'
		self.number_qubits = number_qubits
		self.number_ancilla_qubits = len(circuit.ancilla_qubits())
		self.number_data_qubits = self.number_qubits - self.number_ancilla_qubits

		# This next "if" allows us to postpone the specification of 
		# an initial state until later on.  This is used to reduce 
		# the overhead associated with initializing a State_and_Operations
		# object.  This overhead comes from translating every gate.
		if initial_state != None or len(self.prep_gates) > 0:
			print 'Im preparing states'
			self.current_state = sim.State(self.prep_gates, initial_state,
						       self.number_qubits,
						       self.number_data_qubits)
		else:
			self.current_state = initial_state



	def classify_gates(self, gates):
		"""
		"""
		preparation_gates, operation_gates, meas_gates = [], [], []
		for gate in gates:
			if gate.gate_name[:7] == 'Prepare':
				preparation_gates += [gate]
			elif gate.gate_name[:7] == 'Measure':
				meas_gates += [gate]
			else:
				operation_gates += [gate]

		return preparation_gates, operation_gates, meas_gates



	def translate_gate(self, gate):
		"""
		"""
		
		if gate.qubits[0].qubit_type == 'ancilla':
			Qubit_ID = 1 + number_data_qubits + gate.qubits[0].qubit_id
		else:
			Qubit_ID = 1 + gate.qubits[0].qubit_id
		
		#if gate.gate_name in ['X','Z','H','CX','CZ']:
		if False:
			if gate.gate_name == 'X':
				current_gate = np.matrix(X.X_Circuit(number_qubits,
									Qubit_ID))
			elif gate.gate_name == 'H':
				current_gate = np.matrix(H.Hada_Circuit(number_qubits,
									Qubit_ID))
			elif gate.gate_name == 'Z':
				current_gate = np.matrix(Z.Z_Circuit(number_qubits, Qubit_ID))
			elif gate.gate_name == 'CX':
				if gate.qubits[0].qubit_type == 'ancilla':
					control = 1 + self.number_data_qubits + gate.qubits[0].qubit_id
					target = 1 + gate.qubits[1].qubit_id
				else:
					control = 1 + gate.qubits[0].qubit_id
					target = 1 + self.number_data_qubits + gate.qubits[1].qubit_id
				
				current_gate = np.matrix(CNOT.CNOT_Circuit(self.number_qubits,
									control, target))
			elif gate.gate_name == 'CZ':
				if gates.qubits[0].qubit_type == 'ancilla':
					control = 1 + number_data_qubits + gate.qubits[0].qubit_id
					target = 1 + gate.qubits[1].qubit_id
				else:
					control = 1 + gate.qubits[0].qubit_id
					target = 1 + number_data_qubits + gate.qubits[1].qubit_id
				
				current_gate = np.matrix(CZ.CZ_Circuit(number_qubits,
									control, target))			
		else:
			if len(gate.qubits) == 1:
				current_gate = sim.State_and_Operations.tensor_product_one_qubit_gate(
									gate)
			elif len(gate.qubits) == 2:
				current_gate = sim.State_and_Operations.tensor_product_two_qubit_gate(
									gate)
			else:
				raise NameError('Only 1- and 2-qubit gates '\
						'currently implemented')
		
		return current_gate


	################################################################################
	### Methods copied exactly from Simulatio_real_circuits.State_and_Operations ###

	



	################################################################################



	def apply_single_operation(self, current_state, operation):

		trans_list = [(oper*self.current_state) \
			     *(oper.H) for oper in operation]
		current_state = sum(trans_list)


	def apply_all_operations(self):
		for gate in self.oper_gates:
			print 'Translating gate %s ...' %gate.gate_name
			current_oper = self.translate_gate(gate)
			self.current_state = self.apply_single_operation(self.current_state, current_oper)
			

		## 1) Translate into matrix
		## 2) apply_single_operation
		## 3) erase operation
#gates = [c.Gate(gate_name = 'CX', qubits = [c.Qubit(qubit_id=0, qubit_type = 'data', level=0), \
#	c.Qubit(qubit_id=1, qubit_type = 'data', level = 0)])]
#circ1 = c.Circuit(gates, 'None', 'None')
#zero_zero = np.matrix([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]]) 
#print zero_zero
#CX_Circ = Efficient_State_and_Operations(circ1, 2)

