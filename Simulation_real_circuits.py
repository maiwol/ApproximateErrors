"""
Mauricio Gutierrez Arguedas
September 2012

with important contributions by
Alonzo Hernandez and :(
regarding the parallelization of 
the subcircuit running using pp.
July 2014.
"""

import circuit as c
import functions as fun
from math import sqrt, sin, cos, log
import numpy as np
import sympy as sp
import sympy.matrices as mat
import sympy.physics.quantum as quant
import random as rd
import collections as col
import Approx_Errors as ap
#import visualizer.visualizer as vis
import time as t
#import pp


# The next two functions are still not finished and
# are not being currently used.


#def extract_parallelizable_interval(interval):
#	if len(interval) == 1:
#		return interval, []
#	i = 0
#	partial_list = [interval[i]]
#	merged_list = interval[i] + interval[i+1]
#	reduced_list = list(col.Counter(merged_list))
#	while (len(merged_list) == len(reduced_list)):
#		i += 1
#		partial_list += [interval[i]]
#		merged_list += interval[i+1]
#		reduced_list = list(col.Counter(merged_list))
	
#	for item in partial_list:
#		interval.remove(item)
	
#	return partial_list, interval



#def parallelize_interval(interval):
#	"""
#	This function is used to find which gates can be performed in parallel and
#	reduce the number of matrix multiplications.
#	"""
#	parallelized_interval = []
#	while (interval != []):
#		partial_list, interval = extract_parallelizable_interval(interval)
#		parallelized_interval += [partial_list]
#	return parallelized_interval





class State(object):

	def __init__(self, gates, init_state, n_qs, n_data_qs, sym=False):
		"""
		"""
		if len(gates) == 0:
			self.density_matrix = init_state
			self.qubits = range(n_qs)
		else:
			self.density_matrix, self.qubits = self.prepare_state(
							      gates, init_state, 
                                                              n_qs, n_data_qs, sym)
		#self.is_pure = self.determine_if_pure(sym) 


	def determine_if_pure(self):
		"""
		"""
		if sym:
			x = ((self.density_matrix)**2).trace()
		else:	
			x = np.trace((self.density_matrix)**2)
		if x == 1.:
			return True
		else:
			return False

	def prepare_state(self, gates, init_state, n_qs, n_data_qs, sym):
		"""
		"""
		qubits = range(n_qs)

		if init_state == None:
			initial_states = [0 for i in range(len(gates))]
		
			for gate in gates:
				if gate.gate_name[:7] != 'Prepare':
					raise NameError('Only preparation gates!!')
				if gate.qubits[0].qubit_type == 'ancilla':
					qubit = gate.qubits[0].qubit_id + n_data_qs
				else:
					qubit = gate.qubits[0].qubit_id
	
				if sym:
					initial_states[qubit] = fun.state_vector_dic_sym[gate.gate_name]
				else:
					initial_states[qubit] = fun.state_vector_dic[gate.gate_name]
		
			if len(initial_states) == 1:
				initial_state = initial_states[0]				
			else:
				initial_state = fun.tensor_product(initial_states, sym)
		
			if sym:
				density_matrix = initial_state*quant.Dagger(initial_state)
			else:	
				density_matrix = initial_state*(initial_state.H)
		

		else:
			if len(gates) == 0:
				density_matrix = init_state

			else:	# Right now, we're assuming that only ancilla qubits have Prep gates.
				initial_states = [0 for i in range(len(gates))]
				
				for gate in gates:
					if gate.gate_name[:7] != 'Prepare':
						raise NameError('Only preparation gates!!')
					qubit = gate.qubits[0].qubit_id
				
					if sym:
						initial_states[qubit] = fun.state_vector_dic_sym[gate.gate_name]
					else:
						initial_states[qubit] = fun.state_vector_dic[gate.gate_name]

				if len(initial_states) == 1:
					initial_state = initial_states[0]
				else:
					initial_state = fun.tensor_product(initial_states, sym)
				if sym:
					density_matrix = initial_state*quant.Dagger(initial_state)
					density_matrix = quant.TensorProduct(init_state, density_matrix)
				else:  
					density_matrix = initial_state*(initial_state.H)	
					density_matrix = np.kron(init_state, density_matrix)		

		
		return density_matrix, qubits




class State_and_Operations(object):

	def __init__(self, circuit, number_qubits, initial_state=None, 
		     desired_outcomes=[None], sym=False, rot_errors=None):
		"""
		"""
		prep_gates, oper_gates, meas_gates = self.classify_gates(
								circuit.gates)
		#for gate in oper_gates:
		#	print gate.gate_name
		self.number_qubits = number_qubits
		self.number_ancilla_qubits = len(circuit.ancilla_qubits())
		dif = self.number_qubits - self.number_ancilla_qubits
		self.number_data_qubits = dif 
		self.desired_outcomes = desired_outcomes
		self.sym = sym
		self.rot_errors = rot_errors
		self.stage = 0

		# This next "if" allows us to postpone the specification of 
		# an initial state until later on.  This is used to reduce 
		# the overhead associated with initializing a State_and_Operations
		# object.  This overhead comes from translating every gate.
		if initial_state != None or len(prep_gates) > 0:
			self.initial_state = State(prep_gates, initial_state, 
						   self.number_qubits,
						   self.number_data_qubits, sym)
			self.current_state = State(prep_gates, initial_state, 
						   self.number_qubits, 
					           self.number_data_qubits, sym)
		self.operations = self.prep_operations(oper_gates)
		self.measurements = self.prep_measurements(meas_gates)



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



	def translate_gate(self, gate):
		"""
		"""
		if len(gate.qubits) == 1:
			return self.tensor_product_one_qubit_gate(gate)
		elif len(gate.qubits) == 2:
			return self.tensor_product_two_qubit_gate(gate)
		else:
			raise NameError('Only 1- and 2-qubit gates currently implemented')



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



	def prep_operations(self, operation_gates):
		"""
		"""
		operations = [self.translate_gate(gate) 
			      for gate in operation_gates]
		return operations



	def prep_measurements(self, meas_gates):
		"""
		"""
		measurements = []
		for gate in meas_gates:
			if gate.qubits[0].qubit_type == 'ancilla':
				qubit = gate.qubits[0].qubit_id + self.number_data_qubits
			else:
				qubit = gate.qubits[0].qubit_id
			if self.sym:
				measurements += [[fun.gate_matrix_dic_sym[gate.gate_name], qubit]]
			else:
				measurements += [[fun.gate_matrix_dic[gate.gate_name], qubit]]
		
		return measurements




	def apply_single_operation(self, operation):
		if self.sym:
			final_dens_matr = mat.zeros(2**self.number_qubits)
			trans_list = [(oper*self.current_state.density_matrix)*(quant.Dagger(oper)) for oper in operation]
			for trans_matrix in trans_list:
				final_dens_matr += trans_matrix
			self.current_state.density_matrix = final_dens_matr
		
		else:
			trans_list = [(oper*self.current_state.density_matrix)*(oper.H) for oper in operation]
			self.current_state.density_matrix = sum(trans_list)




	def apply_all_operations(self):
		total = len(self.operations)
		for i in range(total):
			#print "Applying operation %i of %i" %(i+1, total)
			self.apply_single_operation(self.operations[i])
		self.stage =+ 1



	
	def apply_projection(self, meas_operators, qubit, n_qubits, dens, 
			     desired_outcome=['0','1']):
		"""
		Notice that the output is a dictionary whose keys are the desired outcome
		and whose values are the UNNORMALIZED resulting density matrix. The 
		probability of having obtained that measurement is just given by the trace
		of such density matrix.
		"""
		if n_qubits == 1:
			M0 = meas_operators[0]
			M1 = meas_operators[1]
		else:
			if self.sym:
				meas_list0 = [fun.gate_matrix_dic_sym['I'] for i in range(n_qubits)]
				meas_list1 = [fun.gate_matrix_dic_sym['I'] for i in range(n_qubits)]	
			else:
				meas_list0 = [fun.gate_matrix_dic['I'] for i in range(n_qubits)]
				meas_list1 = [fun.gate_matrix_dic['I'] for i in range(n_qubits)]	

			meas_list0[qubit] = meas_operators[0]
			meas_list1[qubit] = meas_operators[1]

			M0 = fun.tensor_product(meas_list0, self.sym)
			M1 = fun.tensor_product(meas_list1, self.sym)

		if self.sym:
			x = sp.simplify(((quant.Dagger(M0))*(M0*dens)).trace())
		else:
			x = round(abs(np.trace((M0.H)*(M0*dens))), 20)
		
		###################################################################### 
		#  The None option was just copied from the do_single_measurement    #
		#  Need to check it later on to make sure it makes sense.  However,  #
		#  we won't use it for the FT EC, so it's OK for now.  Mau 11/14/13  #
		######################################################################

		if desired_outcome == None:
			if x == 0.:
				dens = (M1)*(dens*M1.H)
				meas_outcome = 1
				prob = 1.
				#random = False
			elif x == 1.:
				dens = (M0)*(dens*M0.H)
				meas_outcome = 0
				prob = 1.
				#random = False
			else:
				r = rd.random()
				if r < x:
					#dens = (1./x)*(M0)*(dens*M0.H)
					dens = (M0*dens)*(M0.H)
					meas_outcome = 0
					prob = x	
				else:
					#dens = (1./(1-x))*(M1)*(dens*M1.H)		
					dens = (M1*dens)*(M1.H)
					meas_outcome = 1
					prob = 1-x
				#random = True

		else:

			if len(desired_outcome) == 2:  # ['0', '1']
				if self.sym:
					dens0 = (M0*dens)*(quant.Dagger(M0))
					dens1 = (M1*dens)*(quant.Dagger(M1))
				else:
					dens0 = (M0*dens)*(M0.H)
					dens1 = (M1*dens)*(M1.H)

				return {'0' : dens0, '1' : dens1}


			elif desired_outcome == ['0']:
				if self.sym:
					if x == 0 or x == 0.:
						#raise ValueError("This outcome is impossible")
						#print "This outcome is impossible. This is not an error; \
						#       just be aware that the outcome you desired will \
						#       never happen."
						dens = 0.*dens
					else:
						#dens = (1./x)*(M0*dens)*(quant.Dagger(M0))
						dens = (M0*dens)*(quant.Dagger(M0))
						
				else:
					if x == 0.:
						#raise ValueError("This outcome is impossible.")
						#print "This outcome is impossible. This is not an error; \
						#       just be aware that the outcome you desired will \
						#       never happen."
						dens = 0.*dens
					else:
						#dens = (1./x)*(M0)*(dens*M0.H)
						dens = (M0*dens)*(M0.H)
				
				return {'0' : dens} 
			
		
			elif desired_outcome == ['1']:
				if self.sym:
					if x == 1 or x == 1.:
						#raise ValueError("This outcome is impossible")
						#print "This outcome is impossible. This is not an error; \
						#       just be aware that the outcome you desired will \
						#       never happen."
						dens = 0.*dens
					else:
						#dens = (1./(1.-x))*(M1*dens)*(quant.Dagger(M1))
						dens = (M1*dens)*(quant.Dagger(M1))

				else:
					if x == 1.:
						#raise ValueError("This outcome is impossible.")
						#print "This outcome is impossible. This is not an error; \
						#       just be aware that the outcome you desired will \
                                                #       never happen."
						dens = 0.*dens
					else:
						#dens = (1./(1-x))*(M1)*(dens*M1.H)
						dens = (M1*dens)*(M1.H)
			
				return {'1' : dens}
		


		
########################################################################################
#										       #	
#	Still need to implement the symbolic part when desired_outcomes is None        #
#										       #	
########################################################################################





	def do_single_measurement(self, meas_operators, qubit, desired_outcome=['0','1']):
		"""
		"""
		qubits = self.number_qubits
		new_dic = {}
		for entry in self.current_state.density_matrix:
			outcomes = self.apply_projection(meas_operators, qubit, qubits,
					               self.current_state.density_matrix[entry],
					               desired_outcome)
			for key in outcomes:	
				new_key = entry + key
				new_dic[new_key] = outcomes[key]

		self.current_state.density_matrix = new_dic
		self.current_state.qubits.remove(qubit)
	
		return None	



	def do_all_measurements(self):
		"""
		"""
		#qubit_outcome_dic = {}
		if len(self.desired_outcomes) != len(self.measurements):
			raise ValueError("The number of desired outcomes is NOT " \
					 "the same as the number of measurements " \
					  "in this circuit.")

		# At this point self.current_state.density_matrix is still in fact
		# a density matrix.  The first we do is turn it into a dictionary
		# of the form {'' : density matrix}

		self.current_state.density_matrix = {'' : self.current_state.density_matrix}

		for i in range(len(self.measurements)):
			self.do_single_measurement(self.measurements[i][0], 
					           self.measurements[i][1], 
						   self.desired_outcomes[i])

		self.stage += 1
		
		return None
		#return qubit_outcome_dic



	def trace_out_ancilla_qubits(self):
		"""
		After the measurements on the ancilla qubits have been done,
		we want to trace out the ancillary subsystem and stay only 
		with the data qubits.		
		"""
		num_data = self.number_data_qubits
		num_ancilla = self.number_ancilla_qubits
		for entry in self.current_state.density_matrix:
			dens = self.current_state.density_matrix[entry]
			self.current_state.density_matrix[entry] = fun.trace_out_ancillary_subsystem(
									dens, num_data, num_ancilla, 
									self.sym)
		return None
			

				
	def run_everything(self, initial_state=None):
		if initial_state != None:
			self.initial_state = State([], initial_state, 
						   self.number_qubits,
						   self.number_data_qubits, 
						   self.sym)
			self.current_state = State([], initial_state, 
						    self.number_qubits, 
					        self.number_data_qubits, 
						    self.sym)
		#print '\n\n(1) Very start:\n'
		#print 'type:', type(self.current_state.density_matrix)
		#print 'length:', len(self.current_state.density_matrix)	
		
		#print 'Applying the operations ...'	
		self.apply_all_operations()
		#print '\n\n(2) After applying all operations:\n'
		#print 'type:', type(self.current_state.density_matrix)
		#print 'length:',  len(self.current_state.density_matrix)
		
		#print 'Doing the measurements ...'
		self.do_all_measurements()
		#print '\n\n(3) After doing all measurements:\n'
		#print 'type:', type(self.current_state.density_matrix)
		#print 'keys:', self.current_state.density_matrix.keys()
		
		#print 'Tracing out the ancillae ...'
		self.trace_out_ancilla_qubits()
		#print '\n\n(4) After tracing out the ancillae:\n'
		#print 'type:', type(self.current_state.density_matrix)
		#print 'keys:', self.current_state.density_matrix.keys()
				
		return None
	



class Whole_Circuit(object):
	'''
	circuit can be a proper quantum circuit object (circuit.Circuit()) or a 
	list of quantum circuit objects (the list of subcircuits)
	'''

	def __init__(self, circuit, desired_outcomes=[[None]], 
		     initial_state_data=None, initial_state_ancilla=None, 
		     n_qs=None, n_anc_qs=None, sym=False, rot_errors=None):
		self.sym = sym
		if desired_outcomes[0] == 'already_defined':
			if type(n_qs) != type(1): 
				raise Exception('Need value for n_qs')
			self.n_qs = n_qs
			if type(n_anc_qs) != type(1): 
				raise Exception('Need value for n_anc_qs')
			self.n_ancilla_qs = n_anc_qs
			self.sub_circuits = circuit
		else:
			if type(circuit) == type([]):
				if n_qs != None:  
					self.n_qs = n_qs
				else:             
					self.n_qs = len(circuit[0].qubits())
				self.n_ancilla_qs = len(circuit[0].ancilla_qubits())
				self.sub_circuits = circuit
			else:
				if n_qs != None:  
					self.n_qs = n_qs
				else:             
					self.n_qs = len(circuit.qubits())
				self.n_ancilla_qs = len(circuit.ancilla_qubits())
				self.sub_circuits = fun.split_circuit(circuit, 
								self.n_ancilla_qs)
		
		self.initial_state_data = initial_state_data
		self.initial_state_ancilla = initial_state_ancilla
		self.initial_state = np.kron(initial_state_data, 
					     initial_state_ancilla)
		self.desired_outcomes = desired_outcomes
		if len(self.desired_outcomes) != len(self.sub_circuits):
			raise ValueError("The number of desired outcomes does \
 					  not match the number of subcircuits \
					  in the circuit.")
		self.rot_errors = rot_errors


	def convert_dic_keys(self, states):
        	'''
        	states needs to be a dictionary
        	converts from say '0000 0001' to '01'
        	'''
        	new_states = {}
        	for key in states:
                	key_list = key.split()
                	len_list = len(key_list)
                	# new_key converts from say '0000 0001' to '01'
                	new_key = ''.join([str(key_list[i].count('0')%2) for i in range(len_list)])
                	new_states[new_key] = states[key]

        	return new_states

	

	def get_new_dic_maj_vote(self, outcomes=[0,0,None], dic={},
				 take_final_outcome=False):
		'''
		'''
		new_dic = {}

		if take_final_outcome:
			new_dic = dic

		else:
			for key in dic:
				new_key = ''
				i = 0
				for outcome in outcomes:
					if outcome == None:
						new_key += key[i]
						i += 1
					else:
						new_key += str(outcome)

				new_dic[new_key] = dic[key]
		
		return new_dic 
	

	
	def run_n_subcircs(self, n=6, outcomes=[[['0'],['0','1']] for i in range(4)], 
			   initial_state_data=None, circuit_list=None, n_qs=None,
			   sym=False, initial_state_ancilla=None,
		           sequential=True):
        	'''
        	Specific for subspace stabilizer codes 
		(don't know if it will work for BS)
        	initial_state refers only to the data qubits
		circuit_list can be a list of Circuit objects or a list of
		State_and_Operations objects.  In the latter case, outcomes 
		needs to be 'already_defined'.
        	'''
		if initial_state_data == None:
			initial_state_data = self.initial_state_data
		if circuit_list == None:  
			circuit_list = self.sub_circuits
		if n_qs == None:
			n_qs = self.n_qs
        	next_states = {'': np.kron(initial_state_data, 
					self.initial_state_ancilla)}
        	tensor_ancilla = True
        	for i in range(n):

			#print 'Running circuit %i of %i' %(i+1, n)

			#t1 = t.clock()

        	        if i == n-1:  tensor_ancilla = False
        	        
			if sequential:
				next_states = self.run_one_subcircuit(
							      circuit_list[i], 
                	    				      next_states,
                        	                              outcomes[i],
                                	                      tensor_ancilla,
							      n_qs)
			else:
				next_states = self.parallel_run_one_subcircuit(
							      circuit_list[i], 
                	    				      next_states,
                        	                              outcomes[i],
                                	                      tensor_ancilla,
							      n_qs)
				        	


			#t2 = t.clock()

			#print 'It took me %f seconds.' %(t2-t1)


		return next_states




	#def run_maj_final(self, outcomes, data_dens, ancilla_dens, next_circs):
        #	'''
        #	'''
        #	n_stabs = len(outcomes)/2
        #	out_list1, out_list2 = fun.translate_maj_vote_key(outcomes, n_stabs)

        #	if len(out_list2) == 0:
        #	        out_s = ''.join(out_list1)
        #	        return {out_s:data_dens}

        #	else:
        #	       	circs_to_apply = [next_circs[i] for i in out_list2]
        #        	outcomes_temp = [[['0'],['0','1']] for i in range(len(out_list2))]
        #        	n_data_qs = int(m.log(len(states[key]), 2))
        #        	n_anc_qs = int(m.log(len(ancilla_dens), 2))
        #        	n_qs = n_data_qs + n_anc_qs
        #        	next_states = circ_sim.run_n_subcircs(len(out_list2), outcomes_temp,
        #                                              	      data_dens, circs_to_apply,
        #                                              	      n_qs, False, ancilla_dens)
        #        	next_states = circ_sim.convert_dic_keys(next_states)
        #        	n_ancilla = circ_sim.n_ancilla_qs
        #        	norm_factor = (2**(n_ancilla-1))**(len(out_list2))
        #        	next_states = fun.normalize_states(next_states, norm_factor)
        #        	next_states = circ_sim.get_new_dic(out_list1, next_states)
        #        	return next_states


	def run_one_subcircuit(self, circ, initial_states, desired_outcomes, 
			       tensor_anc=True, n_qs=None, sym=None,
			       init_state_anc=None):
		"""
		initial_states should be a dictionary
		"""
		output = {}
		if n_qs == None:  n_qs = self.n_qs		
		if sym == None:   sym = self.sym
		if init_state_anc == None:  
			init_state_anc = self.initial_state_ancilla

		#print 'Running one subcircuit'
		#print 'keys:', initial_states.keys()

		if desired_outcomes != 'already_defined':
			circ = State_and_Operations(circ, n_qs, None,
						    desired_outcomes,
					            sym, self.rot_errors)
			
		for key in initial_states:
			circ_sim = circ
			circ_sim.run_everything(initial_states[key])
			
			for partial_key in circ_sim.current_state.density_matrix:
				dens = circ_sim.current_state.density_matrix[partial_key]
				if tensor_anc:
					dens = np.kron(dens, init_state_anc)
				if key == '':  new_key = partial_key
				else:  new_key = key + ' ' + partial_key
				output[new_key] = dens

		return output

		#final_state, prob = [], []
		#for output in st_o.current_state.density_matrix:
		#	final_state += [output[0]]
		#prob += [output[0]]
		# We're assuming the non-branching case.	
	
		#dic_parity = 0
		#probabilities = []
		#for value in dic.values():
	 	#	dic_parity += value[0]
		#	probabilities += [value[1]] 
		#dic_parity = dic_parity%2
		#dic_parity = dic.values()[0][0]
		#probabilities = [prob]		
		#return final_state, dic_parity, probabilities
		
		#return final_state


	def circ_alonzo(self, circ, dens_matr, key, tensor_anc, init_state_anc):
        	circ_sim = circ
    		circ_sim.run_everything(dens_matr)

      		array = []
        	for partial_key in circ_sim.current_state.density_matrix:
              		dens = circ_sim.current_state.density_matrix[partial_key]
                   	if tensor_anc:
                  	        dens = numpy.kron(dens, init_state_anc)
                   	if key == '':  new_key = partial_key
                   	else:   new_key = key + ' ' + partial_key
                    	array.append([new_key, dens])
          
		return array


   	def parallel_run_one_subcircuit(self, circ, initial_states, 
					desired_outcomes, tensor_anc=True ,
					n_qs=None, sym=None, 
				        init_state_anc=None):

		print 'Running parallel one subcircuit...'


            	outputs_dict = {}
             	if n_qs == None:  n_qs = self.n_qs
             	if sym == None:   sym = self.sym
             	if init_state_anc == None:  
			init_state_anc = self.initial_state_ancilla

                #loops here
                #Create new name for circ
         	if not desired_outcomes == 'already_defined':
                 	circ = State_and_Operations(circ, n_qs, None,
                                                    desired_outcomes,
                                                    sym)

		#########Parfor###########
      		#Shortcomings:
      		#Only one parameter, so you have to put all the parameters in a list of lists.
        	#Seems to be incredibly slow? Possibly because swap
        	#Testable by adding more memoru
        	#Requires called function to be top-level (in this case must be in a different file)
        	#Not a huge problem
        	#Key Concern: Applicability
        	#Asway Concern: More memory speeds things up
        	#
		#Questions:
		#Is the output correct?
		#Why is not recieving the parameters?
		##########################
		#from qutip import parfor
		#import sub_circuit
		#print [key for key in initial_states]
		#output = parfor(sub_circuit.circ_alonzo, [[circ, initial_states, key, tensor_anc, init_state_anc, output] for key in initial_states])

		#####Parallel Python######
		#Shortcomings:
		#Running out of memory
			#Fixable (?) by adding more memory
		#Key Concern: Practicality
		#Asway Concern: More memory stops it from crashing
		#
		#Questions:
		#Is the output correct?
		#Is it telling me that the number of parameters is incorrect because the function requires self? (possibly outdated question)
		#Are we not getting multi-core usage because it's not working or because it's running out of memory?
		#
		#Tips:
		#Don't import functions using 'as' because parallel python doesn't recognize it.
		##########################

		job_server = pp.Server()

		print 'Initial dens matrs = %i' %len(initial_states)

		print 'Submitting the jobs...'

		jobs = [job_server.submit(self.circ_alonzo,
			(circ, initial_states[key], key, tensor_anc, init_state_anc), 
			(), ("numpy",)) for key in initial_states]

	
		print 'Running jobs...' 
		
		outputs = [job() for job in jobs]

		#job_server.print_stats()

		print 'Done. Now creating dict...'

		for out in outputs:
			for sub in out:
				outputs_dict[sub[0]] = sub[1]

		return outputs_dict



	def run_1_stab_majority_vote(self, circuit_list, initial_state):
		'''
		Specific for the Steane code.
		initial_state refers only to the data qubits
		'''

		#print 'Running run 1 stab...'

		next_states = {'': np.kron(initial_state, self.initial_state_ancilla)}
		tensor_ancilla = True
		for i in range(2):
			next_states = self.run_one_subcircuit(circuit_list[i], 
							      next_states, 
						              [['0'],['0'],['0'],['0','1']],
							      tensor_ancilla)
			tensor_ancilla = False

		#print next_states.keys()

		dens0 = (8**2)*next_states.pop('0000 0000')
		dens1 = (8**2)*next_states.pop('0001 0001')
		for key in next_states:
			next_states[key] = np.kron(next_states[key], self.initial_state_ancilla)		

		next_states = self.run_one_subcircuit(circuit_list[2], next_states,
						      [['0'],['0'],['0'],['0','1']],
						      False)

		#print next_states.keys()

		dens0 += (8**3)*(next_states['0000 0001 0000'] + next_states['0001 0000 0000'])
		dens1 += (8**3)*(next_states['0000 0001 0001'] + next_states['0001 0000 0001'])

		return {'0': dens0, '1': dens1}
		 
		

	def run_3_stab_majority_vote(self):
		'''
		Specific for the Steane code
		'''
		circuit_list = self.sub_circuits
		initial_state = self.initial_state_data
		
		print 'Running first stab...'
		
		first_stab = self.run_1_stab_majority_vote(circuit_list[:3], initial_state)
		
		print 'First stab keys:', first_stab.keys()
	
		second_stab = {}
		third_stab = {}

		print 'Running second stab...'

		for key in first_stab:
			sec_dic = self.run_1_stab_majority_vote(circuit_list[3:6], first_stab[key])
			for sec_key in sec_dic:
				new_key = key + ' ' + sec_key
				second_stab[new_key] = sec_dic[sec_key]

		print 'Second stab keys:', second_stab.keys()

		print 'Running third stab...'

		for key in second_stab:
			third_dic = self.run_1_stab_majority_vote(circuit_list[6:], second_stab[key])
			for third_key in third_dic:
				new_key = key + ' ' + third_key
				third_stab[new_key] = third_dic[third_key]

		print 'Third stab keys:', third_stab.keys()

		return third_stab


	def run_all_in_tree(self, tensor_ancilla_first=True):
		"""
		Run all possible combinations of measurement outcomes.
		"""
		#print '\nSubcircuit 1\n'
		#t0 = t.clock()
			
		circ_sim = State_and_Operations(self.sub_circuits[0], 
						self.n_qs, self.initial_state, 
						self.desired_outcomes[0], 
						self.sym, self.rot_errors)
		circ_sim.run_everything()
		next_states = {}
		for key in circ_sim.current_state.density_matrix:
			state_data = circ_sim.current_state.density_matrix[key]
			if tensor_ancilla_first:
				next_states[key] = np.kron(state_data, 
							self.initial_state_ancilla)
			else:
				next_states[key] = state_data

		#t1 = t.clock()
		#print 'Time: %f s' %(t1-t0)
		tensor_ancilla = True
		for i in range(1,len(self.desired_outcomes)):
			#print '\nSubcircuit %i\n' %(i+1)
			#print 'type:', type(next_states)
			#print 'keys:', next_states.keys()
			#t1 = t.clock()
			print 'Hola'

			if i == (len(self.desired_outcomes)-1):  
				tensor_ancilla = False
			next_states = self.run_one_subcircuit(
						self.sub_circuits[i],
						next_states, 
						self.desired_outcomes[i],
						tensor_ancilla)
			#t2 = t.clock()
			#print "Time: %f s" %(t2-t1)
		

		#t2 = t.clock()
		#print '\nSubcircuit %i\n' %(len(self.desired_outcomes))
		#print 'type:', type(next_states)
		#print 'keys:', next_states.keys()
		#final_states = {}
		#for key in next_states:
		#	circ_sim = State_and_Operations(self.sub_circuits[-1], self.n_qs, 
		#					next_states[key],
		#					self.desired_outcomes[-1], self.sym)
		#	circ_sim.run_everything()
		#	
		#	for partial_key in circ_sim.current_state.density_matrix:
		#		dens = circ_sim.current_state.density_matrix[partial_key]
		#		new_key = key + ' ' + partial_key
		#		final_states[new_key] = dens
		
		#t3 = t.clock()
		#print 'Time: %f s' %(t3-t2)
		#print 'Total time: %f s' %(t3-t0)
		
		return next_states
	

	#def run_all_subcircuits(self):
	#	"""
	#	Temporarily not working.
	#	"""
	#	initial_state = self.initial_state
	#	syndromes = []
	#	probabilities = []
	#	for i in range(len(self.sub_circuits)):
	#		state, dic_parity, probability = self.run_one_subcircuit(
	#							self.sub_circuits[i], 
	#							initial_state, 
	#							self.desired_outcomes[i])
	#		syndromes += [dic_parity]
	#		probabilities += probability
	#		initial_state = state

	#	return [state, syndromes, probabilities]








#circ = Circuit()
#circ.add_gate_at([0], 'PrepareXPlus')
#circ.add_gate_at([0], 'AD g')
#circ.add_gate_at([0], 'MeasureZ')
#circ.add_gate_at([1], 'PrepareZPlus')
#circ.add_gate_at([0,1], 'CX')
#circ.add_gate_at([2], 'PrepareZMinus')
#circ.add_gate_at([3], 'PrepareZPlus')
#circ.add_gate_at([0], 'AD g')
#circ.add_gate_at([1], 'AD g')
#circ.add_gate_at([2], 'AD g')
#circ.add_gate_at([0], 'MeasureZ')
#circ.add_gate_at([1], 'MeasureZ')
#circ.to_ancilla([1])

#x = State_and_Operations(circ, 1, None, [0], True)
#dens1 = x.initial_state.density_matrix
#print dens1
#print '\n'
#x.apply_all_operations()
#dens2 = x.current_state.density_matrix
#print dens2
#print '\n'
#dic = x.do_all_measurements()
#print dic
#print x.current_state.density_matrix
#print '\n'
#x.trace_out_ancilla_qubits()
#print x.current_state.density_matrix
#circ1 = Circuit()
#circ1.add_gate_at([0], 'PrepareZPlus')
#circ1.add_gate_at([1], 'PrepareZPlus')
#circ1.add_gate_at([2], 'PrepareZPlus')
#circ1.add_gate_at([0], 'H')
#circ1.add_gate_at([0], 'ADapprox 0.1')
#circ1.add_gate_at([0,1], 'CX')
#circ1.add_gate_at([1,2], 'CX')
#circ1.add_gate_at([0], 'ADapprox 0.1')
#circ1.add_gate_at([1], 'ADapprox 0.1')
#circ1.add_gate_at([0], 'MeasureZ')
#circ1.add_gate_at([1], 'MeasureZ')
#circ1.add_gate_at([2], 'MeasureZ')
#circ1.to_ancilla([1])


#st1 = State_and_Operations(circ1)

#print st1.measurements

#st1.apply_all_operations()
#dic = st1.do_all_measurements()
#print dic
#print st1.current_state.density_matrix
#st1.trace_out_ancilla_qubits()
#print st1.current_state.density_matrix

#circ2 = Circuit()
#circ2.add_gate_at([0], 'PrepareZPlus')
#circ2.add_gate_at([1], 'PrepareZPlus')
#circ2.add_gate_at([0], 'H')
#circ2.add_gate_at([0], 'AD 0.1')
#circ2.add_gate_at([0,1], 'CX')
#circ2.add_gate_at([0], 'AD 0.1')
#circ2.add_gate_at([1], 'AD 0.1')
#circ2.add_gate_at([0], 'MeasureZ')
#circ2.add_gate_at([1], 'MeasureZ')

#st2 = State_and_Operations(None, circ2.gates)
#print st1.initial_state.density_matrix
#print st2.initial_state.density_matrix
#print st1.stage
#print st.number_qubits
#print st1.operations
#print st.measurements
#st1.apply_all_operations()
#st2.apply_all_operations()
#dens1 = st1.current_state.density_matrix
#dens2 = st2.current_state.density_matrix
#print dens1
#print dens2
#print trace_distance(dens1, dens2)
#print st1.stage
#print st1.do_all_measurements()
#print st1.current_state.density_matrix
#print st1.stage
#print st.current_state.density_matrix, st.current_state.qubits
#print st.do_single_measurement(st.measurements[0][0], st.measurements[0][1])
#print st.do_single_measurement(st.measurements[1][0], st.measurements[1][1])
#print st.run_everything()
#print st.current_state.density_matrix, st.current_state.qubits
#print st.stage
#print st.measurements
