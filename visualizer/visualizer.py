from circuit import Container_Gate


def translate_chp_to_qasm(chp_filename, qasm_filename, n_data_qubits):
	''' 
	This function translates CHP files into flat-QASM files.
	CHP and flat-QASM are very similar.  The only differences are:
	(1) CHP uses 'PrepareZ' while QASM uses 'Pz'
	(2) When specifying 2-qubit gates, CHP leaves a white space
	between the control and target qubits, while QASM separates
	them by a comma.
	(3) If a qubit is ancillary, QASM writes a lowercase 'a' 
	after the qubit number. 
	(4) In CHP, the qubit numbering starts at 0, while in QASM
	it starts at 1.
	'''
	chp_file = open(chp_filename, 'r')
	chp_data = chp_file.readlines()
	chp_file.close()

	qasm_data = ''
	i = 0
	for line in chp_data:
	    print 'i =', i
	    if line[0] != '#':
	        if len(line) == 1:
	            qasm_line = line
	        else:
	            print 'line', line
	            gate = line.split()
	            if gate[0] == 'PrepareZ':
	                qasm_gate = ['Pz']
	            elif gate[0] == 'PrepareX':
	                qasm_gate = ['Px']
	            elif gate[0] == 'MeasureZ':
	                qasm_gate = ['Mz']
	            elif gate[0] == 'MeasureX':
	                qasm_gate = ['Mz']
	            else:
	                qasm_gate = [gate[0]]
	            qubit_list = []               
	            for qubit in gate[1:]:
	                if int(qubit) < int(n_data_qubits):
	                    qubit_list += [str(int(qubit) + 1)]
	                else:
	                    qubit_list += [str(int(qubit) + 1) + 'a']

	            qasm_gate += [','.join(qubit_list)]
	            qasm_line = ' '.join(qasm_gate) + '\n'
	        qasm_data += qasm_line
	    i += 1
	qasm_file = open(qasm_filename, 'w')
	qasm_file.write(qasm_data)
	qasm_file.close() 

	return None



class Printer:
	'''
	This class contains various methods to print quantum circuits
	on text files or the screen.
	'''
	@classmethod
	def circuit_serial_output_string(cls, circuit):
		''' Returns a string of circuit in sufficient ordering of gates.
		'''
		s = ['%s %r'%(g.gate_name, g.qubits) for g in circuit.read_gates(without_t=True)]
		return '\n'.join(s)

	@classmethod
	def circuit_serial_output_string(cls, circuit):
		''' Returns a strig of circuit in a parallel order assuming each gate uses t=1.
		'''
		s = ['%3d %r'%(t, gates) for t,gates in circuit.read_gates(without_t=False)]
		return 'time, gates\n' + '\n'.join(s)

	@classmethod
	def ascii_per_qubit(cls, circuit):
		def _fill(q_dict, timewidth=4):
			maxt = max([len(gates) for gates in q_dict.values()])
			for gates in q_dict.values():
				gates += (['-' * timewidth] * (maxt - len(gates)))
				
		s = dict([(q,[]) for q in circuit.qubits()])
		for t, gates in circuit.read_gates(without_t=False):
			for gate in gates: 	

				if isinstance(gate,Container_Gate):
					for index, sub_circuit in enumerate(gate.circuit_list):
						sub_s = cls.ascii_per_qubit(sub_circuit)
						sub_circuit_label = index if len(gate.circuit_list)>1 else ' '
						for qubit in sub_s.keys():
							s[qubit] += (['--|'+str(sub_circuit_label)] + sub_s[qubit] + ['- |-'])
					#sub_s = cls.ascii_per_qubit(gate.sub_circuit)	
					#for qubit in sub_s.keys():
					#	s[qubit] += (['--| '] + sub_s[qubit] + ['- |-'])
				else:
					name = gate.gate_name
					for q, qubit in enumerate(gate.qubits):
						char = name[q] if len(name)>q else name[0]
						char = '-('+char+')' if (len(gate.qubits)>1) else '--'+char+'-'
						s[qubit] += [char]
			_fill(s)
		return s
	
	@classmethod
	def circuit_console_drawing(cls, circuit):
		qubit_s = cls.ascii_per_qubit(circuit)
		qubits = circuit.qubits()
		for q in qubits:
			s = '%3r'%q.qubit_id +q.qubit_type[0]+' '+ ''.join(qubit_s[q]) + '-\n'
			print s

	@classmethod
	def print_gates_on_qubits(cls, circuit):
		''' This is good for debugging. No error comes from read_gates method.
		'''
		qubits = circuit.qubits()
		for q in qubits:
			print '%4r'%q.qubit_id,
			for g in circuit.qubit_gates_map[q]:
				print '--%s'%g.gate_name + '%r--'%[q.qubit_id for q in g.qubits],
			print ''


	@classmethod
	def print_circuit(cls, circuit, only_physical=False, f=False, file_name=None):
		'''
		The keyword arguments are:
		
		*  circuit -- a `quantum circuit` to be serialized
		*  only_physical -- if True, only the physical gates are printed.  If False, the logical gates are also printed.  
		*  f -- if True, the output is printed on a text file.  If False, it is printed on the screen.
		*  file_name -- if the previous argument is True, then the user needs to specify a name for the output text file.  
		'''
		n_data = len(circuit.data_qubits())
		if f:
			circ_file = open(file_name, 'w')
			if only_physical:
				cls.print_physical_gates(circuit, n_data, True, circ_file)
			else:
				cls.print_all_gates(circuit, n_data, True, circ_file)
                        circ_file.close()
		else:
			if only_physical:
				cls.print_physical_gates(circuit=circuit, n_data=n_data)
			else:
				cls.print_all_gates(circuit=circuit, n_data=n_data)

		return None

	
	@classmethod
	def print_all_gates(cls, circuit, n_data, f=False, name=None):
		for gate in circuit.gates:
			if (hasattr(gate, 'circuit_list')):
				if f:
					instr = '\n' + gate.gate_name + ':' + '\n' + '-----------------------' + '\n'
					name.write(instr)
				else:
					print gate.gate_name, ':', '\n'
				cls.print_all_gates(gate.circuit_list[0], f, name)
				if f:
					division = '-----------------------' + '\n'				
					name.write(division)
				else:
					print '\n'
				 
			else:
				id_list = ' '
				type_list = ' '
				for qubit in gate.qubits:
					if qubit.qubit_type == 'ancilla':
						id_list += str(qubit.qubit_id+n_data)
					else:
						id_list += str(qubit.qubit_id)
					id_list += ' ' 
					type_list += qubit.qubit_type
					type_list += ' '			
				if f:
					if len(gate.gate_name) > 10:
						instr = gate.gate_name + id_list + '\t\t' + type_list + '\n'
					else:
						instr = gate.gate_name + id_list + '\t\t\t' + type_list + '\n'				
					name.write(instr)
				else:
					print gate.gate_name, id_list				
					#print gate.gate_name, ' '.join(qubits_ids)	# Correct
					

	@classmethod
	def print_physical_gates(cls, circuit, n_data, f=False, name=None):
		print '\n\ncircuit with %i data qubits:\n' %n_data
		for gate in circuit.gates:
			if (hasattr(gate, 'circuit_list')):
				cls.print_physical_gates(gate.circuit_list[0], n_data, f, name)
			else:
				qubits_ids = []
				print gate.gate_name
				for qubit in gate.qubits:
					print qubit.qubit_id, qubit.qubit_type
					if qubit.qubit_type == 'ancilla':
						qubits_ids += [str(qubit.qubit_id+n_data)]
					else:	
						qubits_ids += [str(qubit.qubit_id)]
				if f:
					instr = gate.gate_name + ' ' + ' '.join(qubits_ids) +'\n'
					name.write(instr)
				else:
					print gate.gate_name, ' '.join(qubits_ids)
		return None
		
