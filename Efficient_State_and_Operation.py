import numpy as np
import circuit as c
import X_Errors_2 as X
import Hadamard_Gates as H
import Z_Errors as Z
import CNOT_Circuit_Final as CNOT
import CZ_Circuit as CZ
import functions as fun
import Simulation_real_circuits as sim
import sympy as sp


class Efficient_State_and_Operations(object):
	## 1. Classify Gates
	## 2. Assign # Qubits
	## 3. Prep initial_state
	
	def __init__(self, circuit, number_qubits,
	             initial_state=None, desired_outcomes = [None],
		         precompiled_gates=False, precompiled_errors=False,
		         output_folder='.'):
		print number_qubits
		classified_gates = self.classify_gates(circuit.gates)
		self.prep_gates, self.oper_gates, self.meas_gates = classified_gates 
		print 'Im done with classification'
		self.number_qubits = number_qubits
		self.number_ancilla_qubits = len(circuit.ancilla_qubits())
		self.number_data_qubits = self.number_qubits - self.number_ancilla_qubits
		self.rot_errors = None
		self.sym = False
		self.output_folder = output_folder

		if precompiled_gates == False:
			for gate in self.oper_gates:
				print 'Precompiling gate %s' %gate.gate_name
				if len(gate.qubits) > 1:
					if gate.qubits[0].qubit_type == 'ancilla':
						Control_ID = 1  + self.number_data_qubits + gate.qubits[0].qubit_id
					else:
						Control_ID = 1 + gate.qubits[0].qubit_id
					if gate.qubits[1].qubit_type == 'ancilla':
						Target_ID = 1  + self.number_data_qubits + gate.qubits[1].qubit_id
					else:
						Target_ID = 1 + gate.qubits[1].qubit_id
		            
               
					output_filename = '_'.join([gate.gate_name, str(Control_ID-1), str(Target_ID-1)]) + '.npy'
                  
                    
					if gate.gate_name == 'CX':
						trans_gate = np.matrix(CNOT.CNOT_Circuit(self.number_qubits, Control_ID, Target_ID))
					elif gate.gate_name == 'CZ':
						trans_gate = np.matrix(CZ.CZ_Circuit(number_qubits,Control_ID, Target_ID))
					else:
						raise NameError('We havent implemented that gate yet.')    
                    
                    
				else:
					if gate.qubits[0].data_type == 'ancilla':
						Qubit_ID = 1  + self.number_data_qubits + gate.qubits[0].qubit_id
					else:
						Qubit_ID = 1 + gate.qubits[0].qubit_id
		                
					output_filename = '_'.join([gate.gate_name, str(Qubit_ID-1)]) + '.npy'
		            
		                
					if gate.gate_name == 'X':
						trans_gate = np.matrix(X.X_Circuit(self.number_qubits, Qubit_ID))
					elif gate.gate_name == 'H':	
						trans_gate = np.matrix(H.Hada_Circuit(self.number_qubits, Qubit_ID))
					elif gate.gate_name == 'Z':
						trans_gate = np.matrix(Z.Z_Circuit(self.number_qubits, Qubit_ID))
					else:
						pass	
				
				output_filename = output_folder + output_filename
				np.save(output_filename, trans_gate)
                    

		# compile every error and save it on hard drive
		if precompiled_errors == False:
			pass
		# compile every error and save it on hard drive

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
				current_gate = self.tensor_product_one_qubit_gate(
									gate)
			elif len(gate.qubits) == 2:
				current_gate = self.tensor_product_two_qubit_gate(
									gate)
			else:
				raise NameError('Only 1- and 2-qubit gates '\
						'currently implemented')
		
		return current_gate


	################################################################################
	### Methods copied exactly from Simulatio_real_circuits.State_and_Operations ###
	def translate_one_qubit_gate(self, gate_name):
		"""
		input: gate
		output: list of Kraus operators
		"""
		split_gate = gate_name.split(' ')
		if len(split_gate) == 1:
			if self.sym:
				return [fun.gate_matrix_dic_sym[split_gate[0]]]
			else:
				return [fun.gate_matrix_dic[split_gate[0]]]	

		elif len(split_gate) > 1:
			if split_gate[0] == 'Pauli':
				if self.sym:
					pass
				else:
					px = float(split_gate[1])
					py = float(split_gate[2])
					pz = float(split_gate[3])
				return fun.Pauli(px,py,pz, self.sym)
    
            		elif split_gate[0] == 'DC_1q':
                		if self.sym:
                    			pass
                		else:
                    			p = float(split_gate[1])
                		return fun.DC_1q(p, self.sym)
            
			elif split_gate[0] == 'DC_2q':
                		''' 
                		Not really a 1-qubit gate, 
                		but that's fine for now.
                		'''
                		if self.sym:
                    			pass
                		else:
                    			p = float(split_gate[1])
                		return fun.DC_2q(p, self.sym)

			elif split_gate[0] == 'CX_ion_trap':
				'''
				error after CX in ion trap
				'''
				if self.sym:
					pass
				else:
					p = float(split_gate[1])
				return fun.CX_ion_trap(p, self.sym)				
 
            		elif split_gate[0] == 'bit_flip':
                		if self.sym:
                    			pass
                		else:
                    			p = float(split_gate[1])
                		return fun.bit_flip(p, self.sym)

			elif split_gate[0] == 'AD':
				if self.sym:
					gamma = sp.Symbol(split_gate[1], 
							  positive=True)
				else:
					gamma = float(split_gate[1])
				return fun.AD(gamma, self.sym)		

			elif split_gate[0] == 'CMCapproxAD_cons':
				if self.sym:
					pm = sp.Symbol(split_gate[1], 
						       positive=True)
				else:
					pm = float(split_gate[1])
				return fun.CMCapproxAD_cons(pm, self.sym)

			elif split_gate[0] == 'CMCapproxAD_uncons':
				if self.sym:
					gamma = sp.Symbol(split_gate[1], 
							  positive=True)
				else:
					gamma = float(split_gate[1])
				return fun.CMCapproxAD_uncons(gamma, self.sym)
			
			elif split_gate[0] == 'PCapproxAD_cons':
				if self.sym:
					#px = sp.Symbol(split_gate[1], 
					#		positive=True)
					#py = sp.Symbol(split_gate[2], 
					#		positive=True)
					#pz = sp.Symbol(split_gate[3], 
					#		positive=True)
					gamma = sp.Symbol(split_gate[1], 
							  positive=True)
				else:	
					#px = float(split_gate[1])
					#py = float(split_gate[2])
					#pz = float(split_gate[3])
					gamma = float(split_gate[1])
				#return fun.PCapproxAD_cons(px,py,pz, self.sym)
				return fun.PCapproxAD_cons(gamma, self.sym)

			elif split_gate[0] == 'PCapproxAD_uncons':
				if self.sym:
					gamma = sp.Symbol(split_gate[1], 
							  positive=True)
				else:
					gamma = float(split_gate[1])
				return fun.PCapproxAD_uncons(gamma, self.sym)

			elif split_gate[0] == 'PauliapproxAD_uncons':
				if self.sym:
					pass
				else:
					gamma = float(split_gate[1])
				return fun.PauliapproxAD_uncons(gamma, False)


			elif split_gate[0] == 'PolXY':
				if self.sym:
					pass
				else:
					p = float(split_gate[1])
					phi = float(split_gate[2])
				return fun.PolXY(p, phi, self.sym)

			elif split_gate[0] == 'CMCapproxPolXY_cons':
				if self.sym:
					pass
				else:
					p = float(split_gate[1])
					phi = float(split_gate[2])
				return fun.CMCapproxPolXY_cons(p, phi, 
								self.sym)

			elif split_gate[0] == 'CMCapproxPolXY_uncons':
				if self.sym:
					pass
				else:
					p = float(split_gate[1])
					phi = float(split_gate[2])
				return fun.CMCapproxPolXY_uncons(p, phi, 
								self.sym)

			elif split_gate[0] == 'PCapproxPolXY_cons':
				if self.sym:
					pass
				else:
					p = float(split_gate[1])
					phi = float(split_gate[2])
				return fun.PCapproxPolXY_cons(p, phi, 
								self.sym)

			elif split_gate[0] == 'PCapproxPolXY_uncons':
				if self.sym:
					pass
				else:
					p = float(split_gate[1])
					phi = float(split_gate[2])
				return fun.PCapproxPolXY_uncons(p, phi, 
								self.sym)
			
			elif split_gate[0] == 'PauliapproxPolXY_uncons':
				if self.sym:
					pass
				else:
					gamma = float(split_gate[1])
			
				return fun.PauliapproxPolXY_uncons(gamma, False)

			elif split_gate[0] == 'RZC':
				if self.sym:
					pass
				else:
					theta = float(split_gate[1])
				return fun.RZC(theta, self.sym)

			elif split_gate[0] == 'PCapproxRZ_uncons':
				if self.sym:
					pass
				else:
					theta = float(split_gate[1])
				return fun.PCapproxRZ_uncons(theta, self.sym)
		
			elif split_gate[0] == 'PCapproxRZ_cons':
				if self.sym:
					pass
				else:
					theta = float(split_gate[1])
				return fun.PCapproxRZ_cons(theta, self.sym)
			
			elif split_gate[0] == 'PCapproxRZ_geom_mean':
				if self.sym:
					pass
				else:
					theta = float(split_gate[1])
				return fun.PCapproxRZ_geom_mean(theta, self.sym)
			
			elif split_gate[0] == 'CMCapproxRZ_uncons':
				if self.sym:
					pass
				else:
					theta = float(split_gate[1])
				return fun.CMCapproxRZ_uncons(theta, self.sym)

			elif split_gate[0] == 'CMCapproxRZ_cons':
				if self.sym:
					pass
				else:
					theta = float(split_gate[1])
				return fun.CMCapproxRZ_cons(theta, self.sym)

			elif split_gate[0] == 'PauliapproxRZ_uncons':
				if self.sym:
					pass
				else:
					theta = float(split_gate[1])
				return fun.PauliapproxRZ_uncons(theta, self.sym)	

			elif split_gate[0] == 'RHC':
				if self.sym:
					pass
				else:
					theta = float(split_gate[1])
				return fun.RHC(theta, self.sym)
				
			elif split_gate[0] == 'PCapproxRH_uncons':
			    if self.sym:
			        pass
			    else:
			        theta = float(split_gate[1])
			    return fun.PCapproxRH_uncons(theta, self.sym)
			    
			elif split_gate[0] == 'PCapproxRH_cons':
			    if self.sym:
			        pass
			    else:
			        theta = float(split_gate[1])
			    return fun.PCapproxRH_cons(theta, self.sym)
			
			elif split_gate[0] == 'PCapproxRH_geom_mean':
				if self.sym:
					pass
				else:
					theta = float(split_gate[1])
				return fun.PCapproxRH_geom_mean(theta, self.sym)
			    
			elif split_gate[0] == 'CMCapproxRH_uncons':
			    if self.sym:
			        pass
			    else:
			        theta = float(split_gate[1])
			    return fun.CMCapproxRH_uncons(theta, self.sym)
			    
			elif split_gate[0] == 'CMCapproxRH_cons':
			    if self.sym:
			        pass
			    else:
			        theta = float(split_gate[1])
			    return fun.CMCapproxRH_cons(theta, self.sym)
			    
			elif split_gate[0] == 'PauliapproxRH_uncons':
			    if self.sym:
			        pass
			    else:
			        theta = float(split_gate[1])
			    return fun.PauliapproxRH_uncons(theta, self.sym)
			
                        elif split_gate[0] == 'RTC':
			    if self.sym:
			        pass
			    else:
			        theta = float(split_gate[1])
			    return fun.RTC(theta, self.sym)

            		#else:
			#	if self.sym:
			#		pass
			#	else:
			#		channel = split_gate[0]
			#		theta = float(split_gate[1])
			#		cutoff = 1.e-8
			#	
			#	return fun.read_operation_from_json(channel, theta, 
			#					    self.sym, cutoff)
	
			#else:
			#	raise NameError('This gate is not currently implemented.')




	def tensor_product_one_qubit_gate(self, gate):
		"""
		"""
		gate_name = gate.gate_name
		if gate.qubits[0].qubit_type == 'ancilla':
			qubit = gate.qubits[0].qubit_id + self.number_data_qubits
		else:
			qubit = gate.qubits[0].qubit_id

		if gate_name == 'dens':
			gate_operators = [gate.dens]
		else:
			gate_operators = self.translate_one_qubit_gate(gate_name)
	
		if self.rot_errors != None:
			gate_operators=[(self.rot_errors*oper)*(self.rot_errors.H)
					   for oper in gate_operators]
	
		if self.number_qubits == 1:
			return gate_operators 
		else:
			if self.sym:
				identity_list = [fun.gate_matrix_dic_sym['I'] 
					for i in range(self.number_qubits)]
			else:
				identity_list = [fun.gate_matrix_dic['I'] 
					for i in range(self.number_qubits)]
			tensored_gate_list = []
			for operator in gate_operators:
				gate_list = identity_list[:]
				gate_list[qubit] = operator
				tensored_gate_list += [fun.tensor_product(
							gate_list, self.sym)]
			return tensored_gate_list
	


	def tensor_product_two_qubit_gate(self, gate):
		"""
		"""
		gate_name = gate.gate_name
		qubits = [0, 0]
		for i in range(len(gate.qubits)):
			if gate.qubits[i].qubit_type == 'ancilla':
				qubits[i] = gate.qubits[i].qubit_id + self.number_data_qubits
			else:
				qubits[i] = gate.qubits[i].qubit_id


		if gate_name[0] == 'C' and gate_name[:11] != 'CX_ion_trap':
			if len(gate_name) == 1:
				target_gate = gate.dens
			else:
				target_gate = gate_name[1:]
			if target_gate == 'NOT':
				target_gate = 'X'
			c = qubits[0]
			t = qubits[1]

			if self.sym:
				list_I = [fun.gate_matrix_dic_sym['I'] 
					  for i in range(self.number_qubits)]
				list_T = [fun.gate_matrix_dic_sym['I'] 
					  for i in range(self.number_qubits)]
				list_I[c] = mat.Matrix([[1,0],[0,0]])
				list_T[c] = mat.Matrix([[0,0],[0,1]])
				if type(target_gate) == str:
					list_T[t] = gate_matrix_dic_sym[
								target_gate]
				else:
					list_T[t] = target_gate
			
			else:
				list_I = [fun.gate_matrix_dic['I'] 
					  for i in range(self.number_qubits)]
				list_T = [fun.gate_matrix_dic['I'] 
					  for i in range(self.number_qubits)]
				list_I[c] = np.matrix([[1.,0.],[0.,0.]])
				list_T[c] = np.matrix([[0.,0.],[0.,1.]])
				if type(target_gate) == str:
					list_T[t] = fun.gate_matrix_dic[target_gate]
				else:
					list_T[t] = target_gate
			print 'About to start tensoring ...'
			tensored_gate = fun.tensor_product(list_I, self.sym) + fun.tensor_product(list_T, self.sym)
			return [tensored_gate] 

		else:
			'''for now only the 2-qubit DC and the CX_ion_trap'''
			identity_list = [fun.gate_matrix_dic['I'] 
			          	  for i in range(self.number_qubits)]
			tensored_gate_list = []
			gate_operators = self.translate_one_qubit_gate(gate_name) 
			for operator in gate_operators:
				gate_list = identity_list[:]
				gate_list[qubits[0]] = operator[0]
                		gate_list[qubits[1]] = operator[1]
				tensored_gate_list += [fun.tensor_product(
							gate_list, self.sym)]
			return tensored_gate_list

	



	################################################################################



	def apply_single_operation(self, dens_matrix, operation):
		trans_list = [(oper*dens_matrix) \
			     *(oper.H) for oper in operation]
		
        	return sum(trans_list)


	def apply_all_operations(self):
		for gate in self.oper_gates:
			print 'Translating gate %s ...' %gate.gate_name
##			current_oper = self.translate_gate(gate)
			if gate.gate_name in ['CX', 'CZ']:
				current_oper = np.load(self.output_folder + str(gate.gate_name) + '_' + str(gate.qubits[0].qubit_id) + '_' + str(gate.qubits[1].qubit_id))
			elif gate.gate_name in ['H','X','Z']:
				current_oper = np.load(self.output_folder + str(gate.gate_name) + '_' + str(gate.qubits[0].qubit_id))
			else:
				current_oper = self.translate_gate(gate)
			print 'Done translating'
			print 'About to start applying operation ...'
			self.current_state.density_matrix = self.apply_single_operation(
                                                self.current_state.density_matrix,
                                                current_oper)
			

		## 1) Translate into matrix
		## 2) apply_single_operation
		## 3) erase operation
#gates = [c.Gate(gate_name = 'CX', qubits = [c.Qubit(qubit_id=0, qubit_type = 'data', level=0), \
#	c.Qubit(qubit_id=1, qubit_type = 'data', level = 0)])]
#circ1 = c.Circuit(gates, 'None', 'None')
#zero_zero = np.matrix([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]]) 
#print zero_zero
#CX_Circ = Efficient_State_and_Operations(circ1, 2)

