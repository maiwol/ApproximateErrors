"""
Mauricio Gutierrez Arguedas
September 2012

"""
import sys
import os
import circuit as c
import functions as fun
from math import sqrt, sin, cos, log
import numpy as np
import sympy as sp
import sympy.matrices as mat
import sympy.physics.quantum as quant
import random as rd
import collections as col
import faultTolerant.steane as st
import faultTolerant.correction as cor
import faultTolerant.decoder as dec
#import Approx_Errors as ap
#import visualizer.visualizer as vis
#from visualizer import browser_vis as brow
import time as t
#import pp
import multiprocessing as mp


# ---------------- add on to deal with inability to pickle bound methods --------
#import copy_reg
#import types

#def _pickle_method(method):
#	func_name = method.im_func.__name__
#	obj = method.im_self
#	cls = method.im_class
#	return _unpickle_method, (func_name, obj, cls)

#def _unpickle_method(func_name, obj, cls):
#	for cls in cls.mro():
#		try:
#			func = cls.__dict__[func_name]
#		except KeyError:
#			pass
#		else:
#			break
#	return func.__get__(obj, cls)

#copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)

# by Alex Martelli

# ---------------------------------------------------------------------------------


# --------------------- second add on ------------------------------
#def _pickle_method(m):
#	if m.im_self is None:
#		return getattr, (m.im_class, m.im_func.func_name)
#	else:
#		return getattr, (m.im_self, m.im_func.func_name)

#copy_reg.pickle(types.MethodType, _pickle_method)
# ------------------------------------------------------------------



# ----------------- decoding functions for the color code --------------------




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
		print 'init state =', init_state
		print 'n qs =', n_qs
		print 'n data qubits =', n_data_qs

		if init_state == 'None':
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
					print qubit
					print gate.gate_name
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

	def __init__(self, circuit, number_qubits, initial_state='None', 
		     desired_outcomes=[None], sym=False, rot_errors=None):
		"""
		"""
		prep_gates, oper_gates, meas_gates = self.classify_gates(
									circuit.gates)
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

			if split_gate[0] == 'AD':
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
			
			elif split_gate[0] == 'PCapproxRZ_geom_mean':
				if self.sym:
					pass
				else:
					theta = float(split_gate[1])
				return fun.PCapproxRZ_geom_mean(theta, self.sym)
		
			elif split_gate[0] == 'PCapproxRZ_cons':
				if self.sym:
					pass
				else:
					theta = float(split_gate[1])
				return fun.PCapproxRZ_cons(theta, self.sym)
			
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
	
			else:
				raise NameError('This gate is not currently implemented.')

	
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
				gate_list = identity_list
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

		if gate_name[0] == 'C':
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
					list_T[t] = fun.gate_matrix_dic[
								target_gate]
				else:
					list_T[t] = target_gate
			tensored_gate = fun.tensor_product(list_I, self.sym) + fun.tensor_product(list_T, self.sym)
			return [tensored_gate] 

		else:
			raise NameError('Only control something currently implemented')



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
			

				
	def run_everything(self, initial_state):
		'''
		Right now, we have only 3 options:
		(1) initial_state == 'None', and there are prep gates on data and ancilla.
		(2) initial_state != 'None', and there are no prep gates.
		(3) initial_state != 'None', but it's only for the data. Ancilla has prep gates
		'''
		#self.initial_state = State(self.prep_gates, initial_state, 
		#			   self.number_qubits,
		#			   self.number_data_qubits, 
		#			   self.sym)
		#self.current_state = State(self.prep_gates, initial_state, 
		#			   self.number_qubits, 
		#		           self.number_data_qubits, 
		#			   self.sym)

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
		     initial_state_data='None', initial_state_ancilla='None', 
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
		
		self.n_data_qs = self.n_qs - self.n_ancilla_qs
		self.initial_state_data = initial_state_data
		self.initial_state_ancilla = initial_state_ancilla
		if initial_state_data != 'None':
			if initial_state_ancilla != 'None':
				self.initial_state = np.kron(initial_state_data, 
					     		     initial_state_ancilla)
			else:
				self.initial_state = initial_state_data
		else:
			self.initial_state = 'None'


		self.desired_outcomes = desired_outcomes
		if len(self.desired_outcomes) != len(self.sub_circuits):
			raise ValueError("The number of desired outcomes does \
 					  not match the number of subcircuits \
					  in the circuit.")
		self.rot_errors = rot_errors



	def optional_last_part(self, init_string, init_state_data, init_state_anc='None',
			       circuit_list=None, n_qs=None, prob_limit=1.e-10, sym=None,
			       corr='color', last_subcirc_num='final'):
		'''
		new function (MGA 6/9/15)
		To run optional last part of color code circs.
		Right now only for either X's or Z's
		'''
		comp_string = init_string.replace(' ','')
		out_list1, out_list2 = fun.translate_maj_vote_key(comp_string, 3)

		#print comp_string
		#print out_list1, out_list2
        	#sys.exit(0)

		if len(out_list2) == 0:
			out_s = ' '.join(out_list1)
			return {out_s : init_state_data}
		
		else:
			if circuit_list == None:
				circuit_list = self.sub_circuits[-3:]
			if corr == 'color': outc = [['0','1']]
			elif corr == 'Shor':  outc = [['0'], ['0'], ['0'], ['0','1']]
			return self.run_initial_subcircs_tree(outc,
					10000, init_state_data, init_state_anc,
					circuit_list, n_qs, prob_limit, sym,
					corr, last_subcirc_num)	


	def run_initial_subcircs_tree(self, outcomes, n_proc=2, init_state_data='None', 
					  init_state_anc='None', circuit_list=None, 
					  n_qs=None, prob_limit=1.e-10, sym=None, 
					  corr='Shor', last_subcirc_num='final'):
		'''
		new function (MGA 4/7/2014)
		The idea is to serially (non-parallelly) run the first subcircuits of our 
		circuit of interest until we have reached enough branches to send at least
		one to each processor.  This will be obtained when n_branches >= n_proc.
		
		Notice that, in general, the number of branches will not necessarily be a
		power of 2, because if the probability of obtaining a given outcome is 0,
		then that branch is eliminated. 
		
		Notice that we are assumming that the initial ancillary density matrix is 
		the same for each subcircuit.

		For Shor EC,    outcomes = [['0'], ['0'], ['0'], ['0','1']]
		For color code, outcomes = [['0','1']]
		We're assuming the same outcomes for EVERY stabilizer measurement.  If
		we want different outcomes for some stabilizer measurements, we will have
		to change this.

		prob_limit refers to the tolerance of our branching elimination.  More
		specifically, if the ratio of the probability of obtaining a particular 
		outcome over the highest probability is less than 'prob_limit' that whole 
		branch of the tree is eliminated.
		
		last_subcirc_num:  the index of the last subcircuit to be run.
				   By default, we run it until the last subcircuit.
		'''
		
		if init_state_data == 'None':
			init_state_data = self.initial_state_data
		if init_state_anc == 'None':
			init_state_anc = self.initial_state_ancilla
		if circuit_list == None:  
			circuit_list = self.sub_circuits[:]
		if n_qs == None:
			n_qs = self.n_qs
        

		# the dictionary of states keeps track of the branching
		states_dict = {'': init_state_data}
        	
		# whether or not we want to automatically tensor the
		# ancillary density matrix at the end of each run. 
		tensor_ancilla = False
		i = 0

		if last_subcirc_num == 'final': 
			last_subcirc_num = len(circuit_list)


		while len(states_dict.keys()) < n_proc:

			print 'Doing subcirc %i' %i

			# first we tensor the ancillary density matrix
			if init_state_anc != 'None':
				for key in states_dict:
					states_dict[key] = np.kron(states_dict[key],
								   init_state_anc)
			

			# then we run the next subcircuit
			states_dict = self.run_one_subcircuit(circuit_list[i],
							      states_dict,
							      outcomes,
							      tensor_ancilla,
							      n_qs, sym,
							      'None', corr,
						              prob_limit)
			
			i += 1	
			if i >= last_subcirc_num:  break

		return states_dict	



	def run_one_subcircuit(self, circ, initial_states, desired_outcomes, 
			       tensor_anc=True, n_qs=None, sym=None,
			       init_state_anc='None', corr='Shor', prob_limit=1.e-10):
		"""
		initial_states should be a dictionary

		Additions MGA 4/7/15.  With these new additions, run_one_subcircuit will
		not give the right result when called by the old methods.  Therefore,
		when testing the old methods, we should import Script_real_circuits.py, not 
		Simulation_classes_cleaning.py.

		"""
		output = {}
		if n_qs == None:  n_qs = self.n_qs		
		if sym == None:   sym = self.sym
		if init_state_anc == 'None':  
			init_state_anc = self.initial_state_ancilla
			

		#print 'Running one subcircuit'
		#print 'keys:', initial_states.keys()

		if desired_outcomes != 'already_defined':
			circ = State_and_Operations(circ, n_qs, 'None',
						    desired_outcomes,
					            sym, self.rot_errors)
		
		
		#print 'initial total trace=', sum(mat.trace() for mat in initial_states.values())


		for key in initial_states:
			circ_sim = circ

			circ_sim.run_everything(initial_states[key])

			#print 'Ran circuit for key %s' %key
			

			# We want to eliminate outcomes with very small probabilities,
			# to reduce the branching.  But we only do this if we are 
			# collapsing to all outcomes.

			if len(circ_sim.current_state.density_matrix) > 1:
				# first we calculate the probability associated with each
				# outcome.  It's just the trace of each final matrix.
				prob_dict = {}
				
				#print 'probabilities:'
				for partial_key in circ_sim.current_state.density_matrix:
					dens = circ_sim.current_state.density_matrix[partial_key]
					prob_dict[partial_key] = dens.trace()
			
					#print 'partial key =', partial_key
					#print 'prob =', prob_dict[partial_key]
	
				# then we calculate the highest probability
				max_prob = max(prob_dict.values())
				#print 'max prob =', max_prob				


				# finally for each outcome, if the ration between that outcome's
				# probability and the highest probability is less than the limit
				# set by the user, we eliminate that outcome.
				for partial_key in prob_dict:
					if prob_dict[partial_key]/max_prob < prob_limit:
						circ_sim.current_state.density_matrix.pop(partial_key) 
			
			#print 'After eliminating branches...'
			#print 'number of keys =', len(circ_sim.current_state.density_matrix.keys())
			#print '---------------------------------'				
			#t.sleep(10)


			for partial_key in circ_sim.current_state.density_matrix:
				dens = circ_sim.current_state.density_matrix[partial_key]
				if tensor_anc:
					dens = np.kron(dens, init_state_anc)
				# This if is specific to the Steane code with Shor ancilla
				if corr == 'Shor':  new_partial_key = str(partial_key.count('1'))
				else:  new_partial_key = partial_key
				if key == '':  new_key = new_partial_key
				else:  new_key = key + ' ' + new_partial_key
				output[new_key] = dens

		#print '======================================='

		# this next step is done for the new implementation of Shor EC.
		# In this case, we partially normalize the resulting density matrices
		# by a factor of 8, because we have 16 outcomes in total, but all fit into
		# two groups of 8 matrices each: the odd ones and the even ones.
		# Should we save every outcome and then add all the odd ones and all the
		# even ones?  Too slow.  Just check that after this step the total
		# normalization is 1. 
		# Once again, specific to Steane code with Shor ancilla
		if corr == 'Shor':
			norm_factor = 8
			for key in output:
				output[key] *= norm_factor
		
		return output


	
	##############  Methods for branching  ##################

	def _run_intermediate_subcirc(self, subcirc, total_initial_branch_id, local_branch_id,
				      outcomes, initial_dens_mat='None', dens_mat_folder='./',
				      corr='Shor', prob_limit=1.e-10):
		'''
		total_initial_branch_id:  the initial branch
		'''
	
		#print 'running intermediate subcirc'

		if initial_dens_mat == 'None':
			# if the initial density matrix is None, we get it
			# from the npy file.  We then erase the npy file to save
			# disk space.  We might want to change this in the future
			# and erase the npy file later in the simulation.
			npy_filename = ''.join([dens_mat_folder, total_initial_branch_id,
						local_branch_id, '.npy'])
			initial_dens_mat = np.load(npy_filename)
			os.remove(npy_filename)	

		# run the sub-circuit
		init_dict = {'': np.kron(initial_dens_mat, self.initial_state_ancilla)}
		results = self.run_one_subcircuit(subcirc, init_dict, outcomes,
					          False, None, None, None, corr,
					          prob_limit)
		
		if '0' not in results.keys():
			# if the outcome 0 is not present, then we continue
			# along the branch 1.
			next_local_branch_id = local_branch_id + '1'
			dens_mat = results['1']
			elim_branch = None
		else:
			# else, we continue along branch 0.
			next_local_branch_id = local_branch_id + '0'
			dens_mat = results['0']
			if '1' not in results.keys():
				# if the outcome 1 is not present,
				# we add this branch to the list
				# of branches to eliminate, so that
				# we don't have to traverse it.
				elim_branch = local_branch_id + '1'
			else:
				# otherwise, we save this density matrix
				out_npy_filename = ''.join([dens_mat_folder, 
							    total_initial_branch_id,
							    local_branch_id, '1.npy'])
				np.save(out_npy_filename, results['1'])
				elim_branch = None


		return (next_local_branch_id, dens_mat, elim_branch)



	def _run_last_subcirc(self, subcirc, total_initial_branch_id, local_branch_id,
			      outcomes, syndrome_function, initial_dens_mat='None', 
			      dens_mat_folder='./', corr='Shor', prob_limit=1.e-10):
		'''
		'''
		
		#print 'running last subcirc'

		if initial_dens_mat == 'None':
			# if the initial density matrix is None, we get it
			# from the npy file.  We then erase the npy file to save
			# disk space.  We might want to change this in the future
			# and erase the npy file later in the simulation.
			npy_filename = ''.join([dens_mat_folder, total_initial_branch_id,
						local_branch_id, '.npy'])
			initial_dens_mat = np.load(npy_filename)
			os.remove(npy_filename)	

		# run the sub-circuit
		init_dict = {'': np.kron(initial_dens_mat, self.initial_state_ancilla)}
		results = self.run_one_subcircuit(subcirc, init_dict, outcomes,
					          False, None, None, None, corr,
					          prob_limit)

		
		#print 'keys =', results.keys()
		#print 'len matrix =', len(results.values()[0])

		# syndrome function is the function that returns the operation
		# to apply to correct the state.
		corr_results = []
		for key in results:
			total_key = total_initial_branch_id + local_branch_id + key
			#print 'total key =', total_key
			correction = syndrome_function(total_key)
			#print 'correction =', len(correction)
			corr_results += [fun.apply_operation(results[key], correction)]
			#print 'corrected results =', corr_results
		
		return sum(corr_results)



	def run_branching_circ(self, total_initial_branch_id, syndrome_function, 
			       initial_dens_mat='None', circ_list=None, dens_mat_folder='./', 
			       corr='Shor', prob_limit=1.e-10):
		'''
		we don't include outcomes in the inputs, because we are assuming that
		they will always be 0 and 1.
		The initial density matrix CANNOT be specified in a npy file.  It has
		to be specified directly as a numpy matrix.  If it is == None, then
		it will take the initial data density matrix from self.
		'''
		if circ_list == None:
			circ_list = self.sub_circuits
		if initial_dens_mat == 'None':
			initial_dens_mat = self.initial_state_data	

		if corr == 'pseudo' or corr == 'color':
			outcomes = [['0', '1']]
		elif corr == 'Shor':
			outcomes = [['0'], ['0'], ['0'], ['0','1']]
		

		n = len(circ_list)
		local_branch_id = ''
		next_dens_mat = initial_dens_mat
		final_dens_mat = np.matrix([[complex(0.,0.) for i in range(2**self.n_data_qs)]
					                    for j in range(2**self.n_data_qs)])
		
		elim_local_branches = []
		while len(local_branch_id) < n:

			#print 'local branch =', local_branch_id			

			k = len(local_branch_id)

			#print 'running subcirc %i' %k

			if k < n-1:
				partial_results = self._run_intermediate_subcirc(
								circ_list[k], 
								total_initial_branch_id,
								local_branch_id,
								outcomes,
								next_dens_mat,
								dens_mat_folder,
								corr,
								prob_limit)						
				local_branch_id, next_dens_mat, elim_branch = partial_results
				
				#print 'elim branch =', elim_branch
				
				if elim_branch != None:
					elim_local_branches += [elim_branch]

			else:
				final_dens_mat += self._run_last_subcirc(
								circ_list[k],
								total_initial_branch_id,
								local_branch_id,
								outcomes,
								syndrome_function,
								next_dens_mat,
								dens_mat_folder,
								corr,
								prob_limit)
 
				next_dens_mat = None
				#print 'elim branches =', elim_local_branches
				local_branch_id = fun.compute_next_branch(local_branch_id,
									  elim_local_branches,
									  n)
				#print 'outside function'
				#print local_branch_id			
	
		return final_dens_mat		
	
		

	def run_several_branching_circs(self, initial_dict, syndrome_function,
					circ_list, dens_mat_folder='./',
					corr='color', prob_limit=1.e-10):
		'''
		This function is the one the multiprocessing is going to call.
		Most likely, each processor will start only with one branch, but
		this will be able to handle each processor starting with more
		than one branch.
		initial dict is the dictionary of initial branches and initial
		density matrices.
			'''
		final_dens_matrs = []
		for key in initial_dict:
			final_dens_matrs += [self.run_branching_circ(
							key, 
							syndrome_function,
							initial_dict[key],
							circ_list,
							dens_mat_folder,
							corr,
							prob_limit)
					    ]
			
		return sum(final_dens_matrs)


	
	def run_whole_circ_tree_parallel(self, outcomes, syndrome_function, n_proc=4, 
					 init_state_data='None', init_state_anc='None', 
					 circuit_list=None, n_qs=None, dens_mat_folder='./', 
					 prob_limit=1.e-10, sym=None, corr='color', 
					 save_dens=True, output_filename='output.npy'):
		'''
		Originally to be applied to the 7-qubit color code, but easily
		generalizable to any code whose branches don't stop at a middle point,
		like for example the Steane code with Shor ancilla, where the branches
		stop if the first two rounds of stabs gave the same results.
		'''
	
		# first we use a single processor to run the circuit until we get
		# enough branches to send at least one to every processor.
		if circuit_list == None:
			circuit_list = self.sub_circuits
		inter_states_dict = self.run_initial_subcircs_tree(outcomes, n_proc,
								   init_state_data,
								   init_state_anc,
								   circuit_list,
								   n_qs, prob_limit,
								   sym, corr)

		# the next circuit list starts wherever the tree stopped.
		# We can get that from the length of one of the keys.
		len_key = len(inter_states_dict.keys()[0])
		n_spaces = inter_states_dict.keys()[0].count(' ')
		initial_k = len_key - n_spaces

		print 'initial_k =', initial_k
		print 'keys =', inter_states_dict.keys()
		
		# if we already reached the end of the circuit
		if initial_k == len(circuit_list):
			corr_results = []
			for key in inter_states_dict:
				print 'key =', key
				correction = syndrome_function(key)
				corr_results += [fun.apply_operation(
						 inter_states_dict[key], correction)]
			
			return sum(corr_results)

		#sys.exit(0)
	
		
		branch_circuit_list = circuit_list[initial_k:]		

		n_per_proc = len(inter_states_dict) // n_proc
		lists_keys = [inter_states_dict.keys()[i*n_per_proc:(i+1)*n_per_proc]
 						      for i in range(n_proc-1)]
		lists_keys += [inter_states_dict.keys()[(n_proc-1)*n_per_proc:]]
		list_dicts = [dict((k, inter_states_dict[k]) for k in list_key)
			      for list_key in lists_keys]
		

		pool = mp.Pool(n_proc)
		results = [pool.apply_async(self.run_several_branching_circs,
					     (list_dicts[i],
					      syndrome_function,
					      branch_circuit_list,
					      dens_mat_folder,
					      corr,
					      prob_limit))
				for i in range(n_proc)]	
		
		results_list = [r.get() for r in results]

		final_dens_matrix = sum(results_list)
		if save_dens:
			np.save(dens_mat_folder + output_filename, final_dens_matrix)

		return sum(results_list)			

	

def run_one_subcircuit(whole_circ_obj, circ, initial_states, desired_outcomes, 
		       tensor_anc=True, n_qs=None, sym=None,
		       init_state_anc='None', corr='Shor', prob_limit=1.e-10):
	"""
	initial_states should be a dictionary

	Additions MGA 4/7/15.  With these new additions, run_one_subcircuit will
	not give the right result when called by the old methods.  Therefore,
	when testing the old methods, we should import Script_real_circuits.py, not 
	Simulation_classes_cleaning.py.
	"""
	output = {}
	if n_qs == None:  n_qs = whole_circ_obj.n_qs		
	if sym == None:   sym = whole_circ_obj.sym
	if init_state_anc == 'None':  
		init_state_anc = whole_circ_obj.initial_state_ancilla
			

	#print 'Running one subcircuit'
	#print 'keys:', initial_states.keys()

	if desired_outcomes != 'already_defined':
		circ = State_and_Operations(circ, n_qs, None,
					    desired_outcomes,
				            sym, whole_circ_obj.rot_errors)
		
		
	#print 'initial total trace=', sum(mat.trace() for mat in initial_states.values())

	
	for key in initial_states:
		circ_sim = circ
		circ_sim.run_everything(initial_states[key])

		#print 'Ran circuit for key %s' %key
			

		# We want to eliminate outcomes with very small probabilities,
		# to reduce the branching.  But we only do this if we are 
		# collapsing to all outcomes.

		if len(circ_sim.current_state.density_matrix) > 1:
			# first we calculate the probability associated with each
			# outcome.  It's just the trace of each final matrix.
			prob_dict = {}
				
			#print 'probabilities:'
			for partial_key in circ_sim.current_state.density_matrix:
				dens = circ_sim.current_state.density_matrix[partial_key]
				prob_dict[partial_key] = dens.trace()
			
				#print 'partial key =', partial_key
				#print 'prob =', prob_dict[partial_key]
	
			# then we calculate the highest probability
			max_prob = max(prob_dict.values())
			#print 'max prob =', max_prob				


			# finally for each outcome, if the ration between that outcome's
			# probability and the highest probability is less than the limit
			# set by the user, we eliminate that outcome.
			for partial_key in prob_dict:
				if prob_dict[partial_key]/max_prob < prob_limit:
					circ_sim.current_state.density_matrix.pop(partial_key) 
			
		#print 'After eliminating branches...'
		#print 'number of keys =', len(circ_sim.current_state.density_matrix.keys())
		#print '---------------------------------'				
		#t.sleep(10)


		for partial_key in circ_sim.current_state.density_matrix:
			dens = circ_sim.current_state.density_matrix[partial_key]
			if tensor_anc:
				dens = np.kron(dens, init_state_anc)
			# This if is specific to the Steane code with Shor ancilla
			if corr == 'Shor':  new_partial_key = str(partial_key.count('1'))
			else:  new_partial_key = partial_key
			if key == '':  new_key = new_partial_key
			else:  new_key = key + ' ' + new_partial_key
			output[new_key] = dens

	#print '======================================='

	# this next step is done for the new implementation of Shor EC.
	# In this case, we partially normalize the resulting density matrices
	# by a factor of 8, because we have 16 outcomes in total, but all fit into
	# two groups of 8 matrices each: the odd ones and the even ones.
	# Should we save every outcome and then add all the odd ones and all the
	# even ones?  Too slow.  Just check that after this step the total
	# normalization is 1. 
	# Once again, specific to Steane code with Shor ancilla
	if corr == 'Shor':
		norm_factor = 8
		for key in output:
			output[key] *= norm_factor
		
	return output
	
	

##############  Methods for branching  ##################

def run_intermediate_subcirc(circ_sim_obj, subcirc, total_initial_branch_id, local_branch_id,
			      outcomes, initial_dens_mat='None', dens_mat_folder='./',
			      corr='Shor', prob_limit=1.e-10):
	'''
	total_initial_branch_id:  the initial branch
	'''
	
	#print 'running intermediate subcirc'

	if initial_dens_mat == 'None':
		# if the initial density matrix is None, we get it
		# from the npy file.  We then erase the npy file to save
		# disk space.  We might want to change this in the future
		# and erase the npy file later in the simulation.
		npy_filename = ''.join([dens_mat_folder, total_initial_branch_id,
					local_branch_id, '.npy'])
		initial_dens_mat = np.load(npy_filename)
		os.remove(npy_filename)	

	# run the sub-circuit
	if circ_sim_obj.initial_state_ancilla != 'None':
		initial_dens_mat = np.kron(initial_dens_mat, circ_sim_obj.initial_state_ancilla)
	init_dict = {'': initial_dens_mat}
	results = circ_sim_obj.run_one_subcircuit(subcirc, init_dict, outcomes,
				          False, None, None, None, corr,
				          prob_limit)
		
	if '0' not in results.keys():
		# if the outcome 0 is not present, then we continue
		# along the branch 1.
		next_local_branch_id = local_branch_id + '1'
		dens_mat = results['1']
		elim_branch = None
	else:
		# else, we continue along branch 0.
		next_local_branch_id = local_branch_id + '0'
		dens_mat = results['0']
		if '1' not in results.keys():
			# if the outcome 1 is not present,
			# we add this branch to the list
			# of branches to eliminate, so that
			# we don't have to traverse it.
			elim_branch = local_branch_id + '1'
		else:
			# otherwise, we save this density matrix
			out_npy_filename = ''.join([dens_mat_folder, 
						    total_initial_branch_id,
						    local_branch_id, '1.npy'])
			np.save(out_npy_filename, results['1'])
			elim_branch = None


	return (next_local_branch_id, dens_mat, elim_branch)



def run_last_subcirc(circ_sim_obj, subcirc, total_initial_branch_id, local_branch_id,
		      outcomes, syndrome_function, initial_dens_mat='None', 
		      dens_mat_folder='./', corr='Shor', prob_limit=1.e-10,
		      maj_vote=False, num_stabs=3, X_or_Z_error='X'):
	'''
	'''
		
	#print 'running last subcirc'

	#print total_initial_branch_id
	#print local_branch_id

	if initial_dens_mat == 'None':
		# if the initial density matrix is None, we get it
		# from the npy file.  We then erase the npy file to save
		# disk space.  We might want to change this in the future
		# and erase the npy file later in the simulation.
		npy_filename = ''.join([dens_mat_folder, total_initial_branch_id,
					local_branch_id, '.npy'])
		initial_dens_mat = np.load(npy_filename)
		os.remove(npy_filename)	

	# run the sub-circuit
	if circ_sim_obj.initial_state_ancilla != 'None':
		initial_dens_mat = np.kron(initial_dens_mat, circ_sim_obj.initial_state_ancilla)
	init_dict = {'': initial_dens_mat}
	results = circ_sim_obj.run_one_subcircuit(subcirc, init_dict, outcomes,
				          False, None, None, None, corr,
				          prob_limit)

		
	#print 'keys =', results.keys()
	#print type(results.values()[1])
	#print 'len matrix =', len(results.values()[0])

	# syndrome function is the function that returns the operation
	# to apply to correct the state.
	corr_results = []
	for key in results:
		total_key = total_initial_branch_id.replace(' ','') + local_branch_id + key
		if maj_vote:  
			#print total_key
			total_key = total_key[-num_stabs:]
			correction = syndrome_function(total_key, X_or_Z_error)
		else:
			correction = syndrome_function(total_key)
		corr_results += [fun.apply_operation(results[key], correction)]
		#print 'corrected results =', corr_results
	
	return sum(corr_results)



def run_branching_circ(circ_sim_obj, total_initial_branch_id, syndrome_function, 
		       initial_dens_mat='None', circ_list=None, dens_mat_folder='./', 
		       corr='Shor', prob_limit=1.e-10, maj_vote=False,
		       num_stabs=3, X_or_Z_error='X'):
	'''
	we don't include outcomes in the inputs, because we are assuming that
	they will always be 0 and 1.
	The initial density matrix CANNOT be specified in a npy file.  It has
	to be specified directly as a numpy matrix.  If it is == None, then
	it will take the initial data density matrix from self.

	maj_vote, num_stabs, and X_or_Z_errors are only important for the 
	Brown-type decoding, not the naive majority vote.
	'''
	if circ_list == None:
		circ_list = circ_sim_object.sub_circuits
	if initial_dens_mat == 'None':
		initial_dens_mat = circ_sim_object.initial_state_data	

	if corr == 'pseudo' or corr == 'color':
		outcomes = [['0', '1']]
	elif corr == 'Shor':
		outcomes = [['0'], ['0'], ['0'], ['0','1']]
		

	n = len(circ_list)
	local_branch_id = ''
	next_dens_mat = initial_dens_mat
	final_dens_mat = np.matrix([[complex(0.,0.) for i in range(2**circ_sim_obj.n_data_qs)]
				                    for j in range(2**circ_sim_obj.n_data_qs)])
		
	#print 'n =', n
	#print 'initial branch id =', total_initial_branch_id	
	#print maj_vote
	
	total_initial_branch_id = total_initial_branch_id.replace(' ','')
	elim_local_branches = []
	while len(local_branch_id) < n:

		#print 'local branch =', local_branch_id			

		k = len(local_branch_id)

		#print 'running subcirc %i' %k

		total_branch_id = total_initial_branch_id + local_branch_id
		t_k = len(total_branch_id)		

		# We check if we're doing majority voting and if the partial branch
		# has two identical syndromes. If this is true, we don't keep on
		# with this branch.
		stop_partial_branch = False
		if maj_vote and (t_k == 2*num_stabs):
			if (total_branch_id[:num_stabs] == total_branch_id[num_stabs:]):
				stop_partial_branch = True		



		if stop_partial_branch:
			
			#print 'total_branch_id =', total_branch_id
			#print 'next dens mat =', next_dens_mat
			#print type(next_dens_mat)
			if next_dens_mat == 'None':
				# if the initial density matrix is None, we get it
				# from the npy file.  We then erase the npy file to save
				# disk space.  We might want to change this in the future
				# and erase the npy file later in the simulation.
				npy_filename = ''.join([dens_mat_folder, total_initial_branch_id,
				  		        local_branch_id, '.npy'])
				next_dens_mat = np.load(npy_filename)
				os.remove(npy_filename)	

			#print 'total_branch_id', total_branch_id
			#print 'total_branch_id[:num_stabs] =', total_branch_id[:num_stabs]
			correction = syndrome_function(total_branch_id[:num_stabs], 
						       X_or_Z_error)
			#print type(next_dens_mat)
			final_dens_mat += fun.apply_operation(next_dens_mat, correction)
			
			# if we hit a case where the two first syndromes coincide and
			# it's unncessary to measure the stabs a third time, then
			# we assume that we did go into that branch, add 1's at the end,
			# and calculate the next local branch.
			next_dens_mat = 'None'
			local_branch_id += '1'*num_stabs
			local_branch_id = fun.compute_next_branch(local_branch_id,
								  elim_local_branches,
								  n)
 

		else:

			if k < n-1:
				partial_results = run_intermediate_subcirc(
								circ_sim_obj,
								circ_list[k], 
								total_initial_branch_id,
								local_branch_id,
								outcomes,
								next_dens_mat,
								dens_mat_folder,
								corr,
								prob_limit)						
				local_branch_id, next_dens_mat, elim_branch = partial_results
				
				#print 'elim branch =', elim_branch
				
				if elim_branch != None:
					elim_local_branches += [elim_branch]

			else:

				#print 'last branch circ'
				#print total_branch_id

				final_dens_mat += run_last_subcirc(
								circ_sim_obj,
								circ_list[k],
								total_initial_branch_id,
								local_branch_id,
								outcomes,
								syndrome_function,
								next_dens_mat,
								dens_mat_folder,
								corr,
								prob_limit,
								maj_vote,
								num_stabs,
								X_or_Z_error)
 
				next_dens_mat = 'None'
				#print 'elim branches =', elim_local_branches
				local_branch_id = fun.compute_next_branch(local_branch_id,
									  elim_local_branches,
									  n)
				#print 'outside function'
				#print local_branch_id			

	#print 'made it here'
	
	return final_dens_mat		

	

def run_several_branching_circs(circ_sim_obj, initial_dict, syndrome_function,
				circ_list, dens_mat_folder='./',
				corr='color', prob_limit=1.e-10, maj_vote=False,
			        num_stabs=3, X_or_Z_error='X'):
	'''
	This function is the one the multiprocessing is going to call.
	Most likely, each processor will start only with one branch, but
	this will be able to handle each processor starting with more
	than one branch.
	initial dict is the dictionary of initial branches and initial
	density matrices.
	'''
	final_dens_matrs = []

	#print 'keys =', initial_dict.keys()

	for key in initial_dict:
		final_dens_matrs += [run_branching_circ(
					circ_sim_obj,
					key, 
					syndrome_function,
					initial_dict[key],
					circ_list,
					dens_mat_folder,
					corr,
					prob_limit,
					maj_vote,
					num_stabs,
					X_or_Z_error)
				    ]
			
	return sum(final_dens_matrs)



def run_whole_circ_tree_parallel(outcomes_real, outcomes_for_init, syndrome_function, n_proc=4, 
				 init_state_data='None', init_state_anc='None', 
				 circ_or_circ_list=None, n_qs=None, final_dens_mat_folder='./',
				 temp_dens_mat_folder='./', prob_limit=1.e-10, sym=None, 
				 corr='color', save_dens=True, output_filename='output.npy',
				 last_subcirc_num='final'):
	'''
	Originally to be applied to the 7-qubit color code, but easily
	generalizable to any code whose branches don't stop at a middle point,
	like for example the Steane code with Shor ancilla, where the branches
	stop if the first two rounds of stabs gave the same results.
	'''
	
	if not os.path.exists(temp_dens_mat_folder):
		os.makedirs(temp_dens_mat_folder)
	# first we use a single processor to run the circuit until we get
	# enough branches to send at least one to every processor.

	circ_sim = Whole_Circuit(circ_or_circ_list, outcomes_for_init, init_state_data,
				 init_state_anc)

	if last_subcirc_num == 'final':
		last_subcirc_num = len(circ_sim.sub_circuits)

	inter_states_dict = circ_sim.run_initial_subcircs_tree(outcomes_real, n_proc,
								   init_state_data,
								   init_state_anc,
								   None,
								   n_qs, prob_limit,
								   sym, corr,
								   last_subcirc_num)

	# the next circuit list starts wherever the tree stopped.
	# We can get that from the length of one of the keys.
	len_key = len(inter_states_dict.keys()[0])
	n_spaces = inter_states_dict.keys()[0].count(' ')
	initial_k = len_key - n_spaces

	#print 'initial_k =', initial_k
	#print 'keys =', inter_states_dict.keys()
		
	# if we already reached the end of the circuit
	if initial_k == last_subcirc_num:
		corr_results = []
		for key in inter_states_dict:
			#print 'key =', key
			correction = syndrome_function(key)
			corr_results += [fun.apply_operation(
					 inter_states_dict[key], correction)]
		
		final_dens_matrix = sum(corr_results)				

		if save_dens:
			if not os.path.exists(final_dens_mat_folder):
				os.makedirs(final_dens_mat_folder)
		
			np.save(final_dens_mat_folder + output_filename, final_dens_matrix)
		
		return final_dens_matrix

	
	branch_circuit_list = circ_sim.sub_circuits[initial_k:]		

	n_per_proc = len(inter_states_dict) // n_proc
	lists_keys = [inter_states_dict.keys()[i*n_per_proc:(i+1)*n_per_proc]
 					      for i in range(n_proc-1)]
	lists_keys += [inter_states_dict.keys()[(n_proc-1)*n_per_proc:]]
	list_dicts = [dict((k, inter_states_dict[k]) for k in list_key)
		      for list_key in lists_keys]

	
	#sys.exit(0)
	
	#print 'list dicts =', list_dicts[0].keys()
	#print len(branch_circuit_list)
	#print n_proc
	#sys.exit(0)

	
	# serial part
	#results_list = []
	#for i in range(n_proc):
	#	results_list += [run_several_branching_circs(circ_sim,
	#				     list_dicts[i],
	#				     syndrome_function,
	#				     branch_circuit_list,
	#				     temp_dens_mat_folder,
	#				     corr,
	#				     prob_limit)]

	# parallel part
	pool = mp.Pool(n_proc)
	results = [pool.apply_async(run_several_branching_circs,
				     (circ_sim,
				      list_dicts[i],
				      syndrome_function,
				      branch_circuit_list,
				      temp_dens_mat_folder,
				      corr,
				      prob_limit))
			for i in range(n_proc)]	
	

	# new section to make sure we close the pool
	# Otherwise, the memory usage keeps increasing
	# and the whole thing crashes after ~17 min.
	pool.close()
	pool.join()
	# end new section

	
	results_list = [r.get() for r in results]

	final_dens_matrix = sum(results_list)
	if save_dens:
		if not os.path.exists(final_dens_mat_folder):
			os.makedirs(final_dens_mat_folder)
		
		np.save(final_dens_mat_folder + output_filename, final_dens_matrix)

	return sum(results_list)			

	

def run_whole_circ_tree_parallel_dec3(outcomes_real, outcomes_for_init, syndrome_function,
				      n_proc=4, init_state_data='None', init_state_anc='None',
				      circ_or_circ_list=None, n_qs=None, final_dens_mat_folder='./',
			   	      temp_dens_mat_folder='./', prob_limit=1.e-10, sym=None,
				      corr='color', save_dens=True, output_filename='output.npy',
				      last_subcirc_num='final', X_or_Z_error='X'):

	
	maj_vote = True
	num_stabs = 3

	if not os.path.exists(temp_dens_mat_folder):
		os.makedirs(temp_dens_mat_folder)
	# first we use a single processor to run the circuit until we get
	# enough branches to send at least one to every processor.

	#print 'Im here'
	#print outcomes_for_init	

	circ_sim = Whole_Circuit(circ_or_circ_list, outcomes_for_init, init_state_data,
				 init_state_anc)
	#print 'Im here'

	if last_subcirc_num == 'final':
		last_subcirc_num = len(circ_sim.sub_circuits)

	inter_states_dict = circ_sim.run_initial_subcircs_tree(outcomes_real, n_proc,
								   init_state_data,
								   init_state_anc,
								   None,
								   n_qs, prob_limit,
								   sym, corr,
								   last_subcirc_num)
	
	# the next circuit list starts wherever the tree stopped.
	# We can get that from the length of one of the keys.
	len_key = len(inter_states_dict.keys()[0])
	n_spaces = inter_states_dict.keys()[0].count(' ')
	initial_k = len_key - n_spaces

	#print initial_k
	#print inter_states_dict.keys()
	#probs = [value.trace() for value in inter_states_dict.values()]	
	#print probs
	#print sum(probs)	
	#sys.exit(0)

	# if we already reached the end of the circuit
	if initial_k == last_subcirc_num:
		corr_results = []
	
		for key in inter_states_dict:
			one_branch_last_dict = circ_sim.optional_last_part(key, 
							inter_states_dict[key],
							'None', None, None,
							prob_limit, sym, corr, 
							'final')
			
			print 'key =', key
			print one_branch_last_dict.keys()
			#sys.exit(0)

			for last_key in one_branch_last_dict:
				correction = syndrome_function(last_key, X_or_Z_error)
				corr_results += [fun.apply_operation(
					 one_branch_last_dict[last_key], correction)]

		
		final_dens_matrix = sum(corr_results)				

		if save_dens:
			if not os.path.exists(final_dens_mat_folder):
				os.makedirs(final_dens_mat_folder)
		
			np.save(final_dens_mat_folder + output_filename, final_dens_matrix)
		
		return final_dens_matrix



	
	branch_circuit_list = circ_sim.sub_circuits[initial_k:]		

	n_per_proc = len(inter_states_dict) // n_proc
	lists_keys = [inter_states_dict.keys()[i*n_per_proc:(i+1)*n_per_proc]
 					      for i in range(n_proc-1)]
	lists_keys += [inter_states_dict.keys()[(n_proc-1)*n_per_proc:]]
	list_dicts = [dict((k, inter_states_dict[k]) for k in list_key)
		      for list_key in lists_keys]

	#print 'n_per_proc =', n_per_proc
	#print 'list keys', lists_keys
	#print 'keys of dics ='
	#for dic in list_dicts:
	#	print dic.keys()


	# This part is only for debugging purposes.
	# It turns out that it is very hard to debug functions
	# when they are being called in parallel by the
	# multiprocessing module.  So, to debug them, we call
	# them serially. 
	#result_list = []
	#for dic in list_dicts:	
	#	result_list += [run_several_branching_circs(
	#			      circ_sim,
	#			      dic,
	#			      syndrome_function,
	#			      branch_circuit_list,
	#			      temp_dens_mat_folder,
	#			      corr,
	#			      prob_limit,
	#			      maj_vote,
	#			      num_stabs,
	#			      X_or_Z_error)]

	#return sum(result_list)


	# parallel part
	pool = mp.Pool(n_proc)
	results = [pool.apply_async(run_several_branching_circs,
				     (circ_sim,
				      list_dicts[i],
				      syndrome_function,
				      branch_circuit_list,
				      temp_dens_mat_folder,
				      corr,
				      prob_limit,
				      maj_vote,
				      num_stabs,
				      X_or_Z_error))
			for i in range(n_proc)]	
	

	# new section to make sure we close the pool
	# Otherwise, the memory usage keeps increasing
	# and the whole thing crashes after ~17 min.
	pool.close()
	pool.join()
	# end new section

	
	results_list = [r.get() for r in results]

	final_dens_matrix = sum(results_list)
	if save_dens:
		if not os.path.exists(final_dens_mat_folder):
			os.makedirs(final_dens_mat_folder)
		
		np.save(final_dens_mat_folder + output_filename, final_dens_matrix)

	return sum(results_list)		



def run_Steane_code_rep3(error_channel, approx, error_strength, compiling,
                    error_type, theta, phi, prob_limit, n_proc,
                    start_from_half=False):
    '''
    start_from_half:  True if we already have the density matrix for the first
                      round of stabilizers (X stabilizers) and we will read it
                      from the file.
    '''
   
    # define circuits for X and Z stabilizers 
    n_data, n_ancilla = 7, 4
    stabilizers = st.Code.stabilizer
    redundancy = 3
    alternating = True 
    with_initial_I = True
    Steane_code_circ_X = fun.create_FT_EC_no_time(n_data, n_ancilla, stabilizers[:3],
                                redundancy, alternating, with_initial_I)
    Steane_code_circ_Z = fun.create_FT_EC_no_time(n_data, n_ancilla, stabilizers[3:],
                                redundancy, alternating, with_initial_I)
    
    # insert errors
    error_gate = approx + ' ' + str(error_strength)
    gates_after = ['I', 'H', 'CX', 'CZ', 'CY', 'PrepareZ', 'PrepareX']
    #gate0 = Steane_code_circ_X.gates[12]
    #gate1 = Steane_code_circ_X.gates[14]
    #faulty_gate = Steane_code_circ_X.gates[50]
    #fault = 'Z'
    #fault = 'PCapproxRZ_uncons 0.1' 
    #fault = error_gate
    #Steane_code_circ_X.insert_gate(gate0, [gate0.qubits[1]], '', fault, False)
    #Steane_code_circ_X.insert_gate(gate1, [gate1.qubits[1]], '', fault, False)
    fun.insert_errors(Steane_code_circ_X, error_gate, gates_after, 3)
    fun.insert_errors(Steane_code_circ_Z, error_gate, gates_after, 3)
    #brow.from_circuit(Steane_code_circ_X, True)
    #sys.exit(0)    

    # create initial state
    n_qs = 11
    code = 'Steane'
    theta_vector, phi_vector = 0., 0.
    dens_matrs = fun.initial_state_general_different_basis(
				theta_vector, phi_vector,
				theta, phi, code)
    phys_dens, log_dens = dens_matrs

    # create ancilla (4-qubit cat state)
    ancilla_dens = fun.cat_state_dens_matr(n_ancilla)
    #sys.exit(0)

    # outcomes for the branching part
    pre_outcomes = [['0'] for i in range(n_ancilla-1)] + [['0','1']]
    outcomes = [pre_outcomes for i in range(3*3)]   # 3 rounds of 3 stabilizers each
	
    # other parameters
    syndrome_function = fun.get_syndrome_Steane
    last_subcirc_num = 6
    sym = None
    corr = 'Shor' 
    decoding = 'Steane'
    save_dens = True
    state_folder = 'theta' + str(theta) + '_phi' + str(phi)
    temp_dens_mat_folder = '/'.join([os.getcwd(), 'temp', ''])
    final_dens_mat_folder = '/'.join([os.getcwd(), error_channel, approx,
					  compiling, decoding, str(error_type),
					  state_folder, ''])

    dens_filename_partial = str(error_strength) + 'partial.npy'
    dens_filename_final = str(error_strength) + '.npy'
	

    if start_from_half:
        dens_mat_path = final_dens_mat_folder + dens_filename_partial
        state_after_X_stabs = np.load(dens_mat_path)
        print 'Started from half'

    else:
        state_after_X_stabs = run_whole_circ_tree_parallel_dec3(pre_outcomes[:],
		    						outcomes[:],
			    					syndrome_function,
				    				n_proc, log_dens,
					    			ancilla_dens,
						    		Steane_code_circ_X,
							    	n_qs, final_dens_mat_folder,
								    temp_dens_mat_folder,
								    prob_limit, sym, corr,
								    save_dens, dens_filename_partial,
								    last_subcirc_num,
								    'Z')

    #dist = fun.trace_distance(log_dens, state_after_X_stabs)
    #print 'Distance =', dist

    final_state = run_whole_circ_tree_parallel_dec3(pre_outcomes[:],
							outcomes[:],
							syndrome_function,
							n_proc, state_after_X_stabs,
							ancilla_dens,
							Steane_code_circ_Z,
							n_qs, final_dens_mat_folder,
							temp_dens_mat_folder,
							prob_limit, sym, corr,
							save_dens, dens_filename_final,
							last_subcirc_num,
							'X')

    return final_state



def run_color_code_Brown_intermediate(error_channel, approx, error_strength, compiling,
				      error_type, theta, phi, prob_limit, n_proc):
	'''
	run 1 EC step of the 7-qubit color code with Brown-type decoding after the X and
	the Z stabilizer measurements.
	'''	
	decoding = 'Brown_correct_intermediate'
	stabs = st.Code.stabilizer_color_code
	add_I_gates = error_type
	gates_after = ['I']
	error_gate = approx + ' ' + str(error_strength)
	
	syndrome_function = dec.corr_color_code_1_type_error
	color_code_circ_X = cor.color_code.correct_end(stabs[:3], compiling, add_I_gates)

	fun.insert_errors(color_code_circ_X, error_gate, gates_after, 1)
	brow.from_circuit(color_code_circ_X, True)
	sys.exit(0)

	color_code_circ_Z = cor.color_code.correct_end(stabs[3:], compiling, add_I_gates)
	fun.insert_errors(color_code_circ_Z, error_gate, gates_after, 1)

	n_qs = 8
	code = 'color'
	last_subcirc_num = 6
	theta_vector, phi_vector = 0., 0.

	dens_matrs = fun.initial_state_general_different_basis(
				theta_vector, phi_vector,
				theta, phi, code)
	phys_dens, log_dens = dens_matrs
	ancilla_dens = 'None'
	outcomes_for_init = [[['0','1']] for i in range(3*3)] 
	outcomes_real = [['0','1']]
	sym = None
	corr = 'color'
	save_dens = True
	state_folder = 'theta' + str(theta) + '_phi' + str(phi)
	temp_dens_mat_folder = '/'.join([os.getcwd(), 'temp', ''])
	final_dens_mat_folder = '/'.join([os.getcwd(), error_channel, approx,
					  compiling, decoding, str(error_type),
					  state_folder, ''])

	dens_filename_partial = str(error_strength) + '_partial.npy'
	dens_filename_final = str(error_strength) + '.npy'

	state_after_X_stabs = run_whole_circ_tree_parallel_dec3(outcomes_real[:],
								outcomes_for_init[:],
								syndrome_function,
								n_proc, log_dens,
								ancilla_dens,
								color_code_circ_X,
								n_qs, final_dens_mat_folder,
								temp_dens_mat_folder,
								prob_limit, sym, corr,
								save_dens, dens_filename_partial,
								last_subcirc_num,
								'Z')


	final_state = run_whole_circ_tree_parallel_dec3(outcomes_real[:],
							outcomes_for_init[:],
							syndrome_function,
							n_proc, state_after_X_stabs,
							ancilla_dens,
							color_code_circ_Z,
							n_qs, final_dens_mat_folder,
							temp_dens_mat_folder,
							prob_limit, sym, corr,
							save_dens, dens_filename_final,
							last_subcirc_num,
							'X')

	return final_state

