import sys
import os
import numpy as np
import scipy as sc
import sympy as sp
import matplotlib.pyplot as plt
#sys.path += ['/home/mau/MUSIQC/new_musiqc/']
import circuit as c
import faultTolerant.correction as cor
import faultTolerant.steane as st
#import visualizer.browser_vis as brow
#import Simulation_real_circuits as sim
import Simulation_classes_cleaning_notOO as sim
#import sympy.physics.quantum as quant
import sympy.matrices as mat
from copy import deepcopy
from decimal import Decimal
import math as m
from math import sqrt, pi
import time as t
import functions as fun
import json
import multiprocessing as mp
#import qutip as qt
import gc

#import visualizer.browser_vis as browser
#import visualizer.visualizer as vis

#number_gammas = 400
#top_gamma = 0.5
#list_gammas = []
#for exp in [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1]:
#   for i in range(1,10):
#       list_gammas += [i*(10**exp)]


#def pz(g):
#   return 0.0344617*g**4 + 0.00592929*g**3 + 0.0330055*g**2 + 0.14574*g

#def px(g):
#   return -0.00216689*g**4 - 0.000863701*g**3 - 0.00335448*g**2 + 0.523581*g

TMR_stabilizers = [
            ['Z','Z','I'],
            ['I','Z','Z']]


Shor_stabilizers = [
            ['Z','Z','I','I','I','I','I','I','I'],
            ['I','Z','Z','I','I','I','I','I','I'],
            ['I','I','I','Z','Z','I','I','I','I'],
            ['I','I','I','I','Z','Z','I','I','I'],
            ['I','I','I','I','I','I','Z','Z','I'],
            ['I','I','I','I','I','I','I','Z','Z'],
            ['X','X','X','X','X','X','I','I','I'],
            ['I','I','I','X','X','X','X','X','X']]


BS_stabilizers = [
            ['Z','Z','I','I','I','I','I','I','I'],
            ['I','I','I','Z','Z','I','I','I','I'],
            ['I','I','I','I','I','I','Z','Z','I'],
            ['I','Z','Z','I','I','I','I','I','I'],
            ['I','I','I','I','Z','Z','I','I','I'],
            ['I','I','I','I','I','I','I','Z','Z'],
            ['X','I','I','X','I','I','I','I','I'],
            ['I','X','I','I','X','I','I','I','I'],
            ['I','I','X','I','I','X','I','I','I'],
            ['I','I','I','X','I','I','X','I','I'],
            ['I','I','I','I','X','I','I','X','I'],
            ['I','I','I','I','I','X','I','I','X']]


BS_stabilizers_redun = [
            ['Z','Z','I','I','I','I','I','I','I'],
            ['Z','I','Z','I','I','I','I','I','I'],
            ['I','Z','Z','I','I','I','I','I','I'],
            ['I','I','I','Z','Z','I','I','I','I'],
            ['I','I','I','Z','I','Z','I','I','I'],
            ['I','I','I','I','Z','Z','I','I','I'],
            ['I','I','I','I','I','I','Z','Z','I'],
            ['I','I','I','I','I','I','Z','I','Z'],
            ['I','I','I','I','I','I','I','Z','Z'],
            ['X','I','I','X','I','I','I','I','I'],
            ['X','I','I','I','I','I','X','I','I'],
            ['I','I','I','X','I','I','X','I','I'],
            ['I','X','I','I','X','I','I','I','I'],
            ['I','X','I','I','I','I','I','X','I'],
            ['I','I','I','I','X','I','I','X','I'],
            ['I','I','X','I','I','X','I','I','I'],
            ['I','I','X','I','I','I','I','I','X'],
            ['I','I','I','I','I','X','I','I','X']]


BS_stabilizers_redun = [
            ['X','I','I','X','I','I','I','I','I'],
            ['X','I','I','I','I','I','X','I','I'],
            ['I','I','I','X','I','I','X','I','I'],
            ['I','X','I','I','X','I','I','I','I'],
            ['I','X','I','I','I','I','I','X','I'],
            ['I','I','I','I','X','I','I','X','I'],
            ['I','I','X','I','I','X','I','I','I'],
            ['I','I','X','I','I','I','I','I','X'],
            ['I','I','I','I','I','X','I','I','X']]


BS_stabilizers_4 = [
                    ['Z','Z','I','Z','Z','I','Z','Z','I'],
            ['I','Z','Z','I','Z','Z','I','Z','Z'],
            ['X','X','X','X','X','X','I','I','I'],
            ['I','I','I','X','X','X','X','X','X']]

BS_logicals = {'X': ['X','X','X','I','I','I','I','I','I'],
           'Y': ['Y','X','X','Z','I','I','Z','I','I'],
           'Z': ['Z','I','I','Z','I','I','Z','I','I']}


BS_logical_opers = {}
BS_logical_opers_eigensystem = {}
for key in BS_logicals:
    output_tuple = fun.oper_and_eigensystem(BS_logicals[key])   
    BS_logical_opers[key], BS_logical_opers_eigensystem[key] = output_tuple




"""
Because we were never able to find out a symbolic expression
for the best approximation to the ADC using the PC, we are taking
the numerical data from a text file and converting it into a dictionary.
Here, the key is the value of gamma and the value is a list [px,py,pz].
"""
#Pauli_folder = '/home/mau/ApproximateErrors/Simulation/'
#Pauli_file = Pauli_folder + 'Pauli_coefficients.txt'
#dic = open(Pauli_file, 'r')
#dic_Pauli = {}
#for line in dic.readlines():
#   split_line = line[:-1].split()
#   dic_Pauli[float(split_line[0])] = [float(split_line[i]) for i in [2,3,1]]   
#dic.close()



def dif(x,a,b,c,d,e,f,g):    # Rewrite this in a more elegant and versatile way.
    return a*x + b*x**2 + c*x**3 + d*x**4 + e*x**5 + f*x**6 + g*x**7



# Original create non-FT error correction step.
# Currently not used.
#def create_nonFT_EC(number_qubits, stabilizers):
#   """
#   Original create non-FT error correction step.
#   Currently not used.
#   """
#   n = number_qubits
#   circ = c.Circuit()
#   for stab in stabilizers:
#       if 'X' in stab:
#           circ.add_gate_at([n], 'PrepareXPlus')
#           for i in range(len(stab)):
#               if stab[i] != 'I':
#                   circ.add_gate_at([n,i], 'CX')
#           circ.add_gate_at([n], 'MeasureX')
#       if 'Z' in stab:
#           circ.add_gate_at([n], 'PrepareZPlus')
#           for i in range(len(stab)):
#               if stab[i] != 'I':
#                   circ.add_gate_at([i,n], 'CX')
#           circ.add_gate_at([n], 'MeasureZ')
#   circ.to_ancilla([n])
#   return circ


def create_nonFT_EC(n_data, stabilizers, with_initial_I=True,
            rot=None, rot_where='errors'):
    '''
    New option to create a non-FT error correction step
    The preparation step is not included in the circuit.
    It has to be included independently in a previous step, and
    it should be |+>.
    rot refers to the 1-qubit rotation matrix to be applied
    on each qubit of the stabilizers.
    '''
    circ = c.Circuit()
    if with_initial_I:
        for i in range(n_data):
            circ.add_gate_at([i], 'I')

    for stab in stabilizers:
        if with_initial_I:
            circ.add_gate_at([n_data], 'I')
        for i in range(len(stab)):
            if stab[i] != 'I':
                gate = 'C'
                if rot == None or rot_where == 'errors':  
                    gate += stab[i]
                    dens = None
                else:
                    prim_dens = fun.gate_matrix_dic[stab[i]]
                    dens = (rot*prim_dens)*(rot.H)                  
		
                circ.add_gate_at([n_data,i], gate, dens)

        circ.add_gate_at([n_data], 'MeasureX')

    circ.to_ancilla([n_data])

    return circ



def create_nonFT_EC_for_Steane():
    """
    """
    return create_nonFT_EC(7, st.Code.stabilizer)



def create_nonFT_EC_for_Shor():
    '''
    '''
    return create_nonFT_EC(9, Shor_stabilizers)



def create_EC_for_BS():
    '''
    '''
    return create_nonFT_EC(9, BS_stabilizers)



def create_EC_for_BS_redun():
    '''
    '''
    return create_nonFT_EC(9, BS_stabilizers_redun)



def create_EC_for_BS_12_qubits(with_initial_I=True):
    ''' 
    '''
    circ = c.Circuit()
    if with_initial_I:
        for i in range(9):
            circ.add_gate_at([i], 'I')

    Z_pairs = [[[0,1],[3,4],[6,7]], [[1,2],[4,5],[7,8]]]
    X_pairs = [[[0,3],[1,4],[2,5]], [[3,6],[4,7],[5,8]]]

    #for pairs in Z_pairs:
    #   for i in range(9,12):
    #       if with_initial_I:  
    #           circ.add_gate_at([i], 'I')
    #       circ.add_gate_at([i, pairs[i-9][0]], 'CZ')
    #       circ.add_gate_at([i, pairs[i-9][1]], 'CZ')
    #       circ.add_gate_at([i], 'MeasureX')

    for pairs in X_pairs:
        for i in range(9,12):
            if with_initial_I:  
                circ.add_gate_at([i], 'I')
            circ.add_gate_at([i, pairs[i-9][0]], 'CX')
            circ.add_gate_at([i, pairs[i-9][1]], 'CX')
            circ.add_gate_at([i], 'MeasureX')
            
    circ.to_ancilla(range(9,12))

    return circ
    

def create_EC_for_BS_12_qubits_redun(with_initial_I=True):
    ''' 
    '''
    circ = c.Circuit()
    if with_initial_I:
        for i in range(9):
            circ.add_gate_at([i], 'I')

    Z_pairs = [[[0,1],[0,2],[1,2]], [[3,4],[3,5],[4,5]], [[6,7],[6,8],[7,8]]]
    X_pairs = [[[0,3],[0,6],[3,6]], [[1,4],[1,7],[4,7]], [[2,5],[2,8],[5,8]]]

    for pairs in Z_pairs:
        for i in range(9,12):
            if with_initial_I:  
                circ.add_gate_at([i], 'I')
            circ.add_gate_at([i, pairs[i-9][0]], 'CZ')
            circ.add_gate_at([i, pairs[i-9][1]], 'CZ')
            circ.add_gate_at([i], 'MeasureX')

    for pairs in X_pairs:
        for i in range(9,12):
            if with_initial_I:  
                circ.add_gate_at([i], 'I')
            circ.add_gate_at([i, pairs[i-9][0]], 'CX')
            circ.add_gate_at([i, pairs[i-9][1]], 'CX')
            circ.add_gate_at([i], 'MeasureX')
            
    circ.to_ancilla(range(9,12))

    return circ



def create_checking_circuit_for_BS(logical_state='Z', n_data=9):
    '''
    '''
    circ = c.Circuit()
    for stab in BS_stabilizers_4:
        for i in range(len(stab)):
            if stab[i] != 'I':
                gate = 'C' + stab[i]
                circ.add_gate_at([n_data,i], gate)
        
        circ.add_gate_at([n_data], 'MeasureX')
    
    for i in range(len(BS_logicals[logical_state])):
        if BS_logicals[logical_state][i] != 'I':
            gate = 'C' + BS_logicals[logical_state][i]
            circ.add_gate_at([n_data,i], gate)

    circ.add_gate_at([n_data], 'MeasureX')

    circ.to_ancilla([n_data])

    return circ



def create_FT_EC_no_time(n_data, n_ancilla, stabilizers, redundancy=1,
             alternating=False, with_initial_I=True):
    '''
    This function creates a circuit that measures stabilizers
    assuming the ancilla is ALREADY prepared in a cat state.
    The preparation of the cat state will occur independently
    in a previous step.
    We also assume that CX and CZ are equally easy (or hard) to
    implement.  
    We still need to add the decoding option.
    '''
    n_stab = len(stabilizers)
    circ = c.Circuit()
    if with_initial_I:
        for i in range(n_data):
            circ.add_gate_at([i], 'I')

    new_stabs = []
    if alternating:
        for rep in range(redundancy):
                new_stabs += stabilizers
    
    else:
        for stab in stabilizers:
            new_stabs += [stab for i in range(redundancy)]

    for stab in new_stabs:
        i_a = 0
        for i in range(len(stab)):
            if stab[i] != 'I':
                if with_initial_I:
                    circ.add_gate_at([n_data+i_a], 'I')
                gate = 'C' + stab[i]
                circ.add_gate_at([n_data+i_a,i], gate)
                i_a += 1

        for i in range(i_a):
            circ.add_gate_at([n_data+i], 'MeasureX')

    for i in range(n_ancilla):  
        circ.to_ancilla([n_data+i])         

        # still need to add other option (decoding)
    return circ
                    


def create_FT_EC_time(n_data, n_ancilla, stabilizers):
    '''
    '''
    circ = c.Circuit()
    for i in range(n_data):
        circ.add_gate_at([i], 'I')
    
    for stab in stabilizers:    
        i_a = 0
        for i in range(len(stab)):
            if stab[i] != 'I':
                circ.add_gate_at([n_data+i_a], 'I')
                gate = 'C' + stab[i]
                ctrl = n_data + i_a
                circ.add_gate_at([ctrl,i], gate)
                all_qubits = range(n_data + n_ancilla)
                all_qubits.remove(ctrl)
                all_qubits.remove(i)
                for q in all_qubits:
                    circ.add_gate_at([q], 'I')
                i_a += 1
                    
        for i in range(i_a):
            circ.add_gate_at([n_data+i], 'MeasureX')
        for i in range(n_data):
            circ.add_gate_at([i], 'I')

    for i in range(i_a):    
        circ.to_ancilla([n_data+i])         

        # still need to add other option (decoding)
    return circ





def error(gate, n_qubits=7, qubits='All'):
    """
    """
    circ = c.Circuit()
    if qubits == 'All':
        for i in range(n_qubits):
            circ.add_gate_at([i], gate)
    else:
        for qubit in qubits:
            circ.add_gate_at([qubit], gate)
    return circ



def generate_list_error_gates(approx, list_ps, phi=None):
    '''
    For ADC or PolXY
    '''
    list_error_gates = []

    #if approx == 'PCapproxAD_cons':
    #   values = [' '.join(map(str, dic_Pauli[p])) for p in list_ps]
    #else:
    #   values = map(str, list_ps)

    # The PC_cons approximation to the ADC is the only approximation
    # for which we don't have an exact symbolic expression for the 
    # coefficients.  However, the polynomial fit is almost perfect 
    # (see figure), so instead of fetching the numerical values from 
    # the text file and getting an error if the particular damping
    # strength doesn't exist, let's just use the polynomial fit. 
    values = map(str, list_ps)  

    if approx == 'Dep_noise':
        list_error_gates = [' '.join(['Pauli', value, value, value]) 
                    for value in values]
    
    else:
        if phi==None:
            list_error_gates = [' '.join([approx, value]) 
                        for value in values] 
        else:   
            list_error_gates = [' '.join([approx, value, str(phi)]) 
                        for value in values] 
    
    return list_error_gates



def run_maj_final(outcomes, data_dens, ancilla_dens, next_circs, 
          take_final_outcome=False, next_outcomes='already_defined'):
        '''
        '''
        n_stabs = len(outcomes)/2
        out_list1, out_list2 = fun.translate_maj_vote_key(outcomes, n_stabs)

        if len(out_list2) == 0:
                out_s = ''.join(out_list1)

                return {out_s:data_dens}

        else:
                if take_final_outcome:
                        circs_to_apply = next_circs
                        len_circs = n_stabs

                else:
                        circs_to_apply = [next_circs[i] for i in out_list2]
                        len_circs = len(out_list2)

                n_data_qs = int(m.log(len(data_dens), 2))
                n_anc_qs = int(m.log(len(ancilla_dens), 2))
                n_qs = n_data_qs + n_anc_qs
        
        if next_outcomes == 'already_defined':
            outcomes_temp = ['already_defined' for i in range(len_circs)]
        else:
            pre_outcomes_temp = [['0'] for i in range(n_anc_qs-1)] + [['0','1']] 
            outcomes_temp = [pre_outcomes_temp for i in range(len_circs)]
               
        #print 'next_outcomes =', next_outcomes
        #print 'outcomes_temp =', outcomes_temp

        circ_sim = sim.Whole_Circuit(circs_to_apply, outcomes_temp,
                                             data_dens, ancilla_dens, n_qs,
                                             n_anc_qs)

        next_states = circ_sim.run_n_subcircs(len_circs, outcomes_temp)
        next_states = circ_sim.convert_dic_keys(next_states)
        n_ancilla = circ_sim.n_ancilla_qs
        norm_factor = (2**(n_ancilla-1))**(len_circs)
        next_states = fun.normalize_states(next_states, norm_factor)
        next_states = circ_sim.get_new_dic_maj_vote(out_list1, next_states,
                                                    take_final_outcome)

        return next_states
    


def run_final_and_corr(states_dict, ancilla_dens, next_circs,
               take_final_outcome, next_outcomes, n_data=7,
               sym=False, code='Steane', length=6, error='Z', 
               rot=None):
    '''
    function is defined like this for the sole purpose of 
    parallelizing it.
    '''
    entries = range(2**n_data)
    pre_matrix = [[complex(0.,0.) for i in entries] for j in entries]

    if sym:
        final_dens = mat.Matrix(pre_matrix)
    else:
        final_dens = np.matrix(pre_matrix)
        
    for key in states_dict: 
        #print key

        next_states = run_maj_final(key, states_dict[key], ancilla_dens,
                        next_circs, take_final_outcome,
                        next_outcomes)
        corr_states = fun.apply_correction_to_every_state(next_states,
                              sym, code,
                              length, error,
                              rot)
        final_dens += sum(corr_states.values())

    return final_dens


def run_maj_vote_total(n_data, n_ancilla, stabs, redun, alternating, with_initial_I,
               error_gate, error_kind, log_dens, code, error='X', 
               length=3, sym=False, take_final_outcome=True, parallel=True,
               n_proc=4, saved_dens=False, last_subcirc_num='final',
               prob_limit=1.e-10, corr='Shor'):
    '''
    n_proc: number of processors to be used in case parallel == True
    saved_dens was an option that I included when testing the function.
    The idea was, because the parallel part is only on the final 3
    subcircuits, to already at this point.  In general, we will
    not use this option.  
    Notice that all of this only applies to the old Steane code, which
    will be running on madiba and possibly dirac during the summer of
    2015.  It doesn't apply to the color code, which will use the new
    general parallel code.
    '''

    ##### Defining basic stuff #####

    n_stabs = len(stabs)
    n_tot_circ = 3*n_stabs
    n_first_circ = 2*n_stabs
    #phys_dens, log_dens = fun.initial_state_general(theta, phi, code)
    
    ##### Defining the circuit and inserting the errors #####

    circ = create_FT_EC_no_time(n_data, n_ancilla, stabs, redun,
                                    alternating, with_initial_I)
    
    #brow.from_circuit(circ, True)

    #fun.insert_errors(circ, error_gate, ['I','H','CX','CZ'], error_kind)
    gate0 = circ.gates[0]
    new_g = circ.insert_gate(gate0, gate0.qubits, '', 'Z', False)    
    new_g.is_error = True

    #brow.from_circuit(circ, True)
    #sys.exit(0)    
    
    ##### Running the first two stabilizer rounds  #####

    t1 = t.clock()
    print 'Running first two rounds of stabilizer measurements ...'

    pre_outcomes = [['0'] for i in range(n_ancilla-1)] + [['0','1']]
    outcomes = [pre_outcomes for i in range(n_tot_circ)]

    ancilla_dens = fun.cat_state_dens_matr(n_ancilla)
    circ_sim = sim.Whole_Circuit(circ, outcomes, log_dens, ancilla_dens)

    circ_list = circ_sim.sub_circuits
    initial_state = circ_sim.initial_state_data
    
    if last_subcirc_num == 'final':
        last_subcirc_num = len(circ_sim.sub_circuits)

    
    print 'Starting to run n_subcircs...'   

    # first we use a single processor to run the circuit until we get
    # enough branches to send at least one to every processor.
    inter_states_dict = circ_sim.run_initial_subcircs_tree(pre_outcomes, 
                                n_proc, 'None', 'None', None, None,
                                prob_limit, sym, corr, last_subcirc_num)

    
    t2 = t.clock()
    print 'Total time for the first part: %f seconds' %(t2-t1)

    print 'Dict length = %i' %len(inter_states_dict)
    print 'Dict keys ='
    for key in inter_states_dict:
        print key

    # Thought the garbage collector might help to free up some
    # memory, but it didn't.  MGA 2/8/2016.
    #gc.collect()
    
    #print inter_states_dict

    #sys.exit(0)

    ##### Running final round #####

    print 'Running last part'

    # the next circuit list starts wherever the tree stopped.
    # We can get that from the length of one of the keys.
    len_key = len(inter_states_dict.keys()[0])
    n_spaces = inter_states_dict.keys()[0].count(' ')
    initial_k = len_key - n_spaces

    print 'initial_k =', initial_k
    print 'keys =', inter_states_dict.keys()


    # if we already reached the end of the circuit
    if initial_k == last_subcirc_num:
        corr_results = []
        for key in inter_states_dict:
            correction = syndrome_function(key)
            corr_results += [fun.apply_operation(
                            inter_states_dict[key], correction)]    
        
        return sum(corr_results)

        


    count = 1

    n_keys = len(states)
    n_per_proc = n_keys // n_proc
    lists_keys = [states.keys()[i*n_per_proc:(i+1)*n_per_proc] 
                   for i in range(n_proc-1)]
    lists_keys += [states.keys()[(n_proc-1)*n_per_proc:]]
    list_dicts = [dict((k, states[k]) for k in list_key) 
              for list_key in lists_keys]

    print n_keys
    print n_per_proc
    print lists_keys
    print 'n proc =', n_proc
    #sys.exit(0)


    #run_final_and_corr(list_dicts[0], ancilla_dens, circ_list[n_first_circ:],
    #          take_final_outcome, 'not_defined', n_data, sym, code, length,
    #          error)

    
    pool = mp.Pool(n_proc)
    results = [pool.apply_async(run_final_and_corr, (list_dicts[i],
                              ancilla_dens,
                              circ_list[n_first_circ:],
                              take_final_outcome,
                              'not_defined', n_data,
                              sym, code, length,
                              error))
                    for i in range(n_proc)]

    pool.close()
    pool.join()
    
    results_list = [r.get() for r in results]
    
    return sum(results_list)
        



    #for key in states:
    #   print 'Running circuit %i.  Key = %s' %(count,key)

    #   t3 = t.clock()

    #   next_states = run_maj_final(key, states[key], ancilla_dens,
        #                                circ_list[n_first_circ:], take_final_outcome,
    #               'not_defined')
        #   corr_states = fun.apply_correction_to_every_state(next_states, sym, 
    #                             code, 3, error)
    #   count += 1
            #final_dens_list += [sum(corr_states.values())]
    #   final_dens += sum(corr_states.values())

    #   t4 = t.clock()

    #   print 'Total time for this key: %f seconds'     


    #t5 = t.clock()
    #print 'Total time for the last part: %f seconds' %(t5-t2)

    #final_dens = sum(final_dens_list)
    
    #return final_dens


def run_maj_vote_total_old(n_data, n_ancilla, stabs, redun, alternating, with_initial_I,
               error_gate, error_kind, log_dens, code, error='X', 
               length=3, sym=False, take_final_outcome=True, parallel=True,
               n_proc=4, saved_dens=False):
    '''
    n_proc: number of processors to be used in case parallel == True
    saved_dens was an option that I included when testing the function.
    The idea was, because the parallel part is only on the final 3
    subcircuits, to already at this point.  In general, we will
    not use this option.  
    Notice that all of this only applies to the old Steane code, which
    will be running on madiba and possibly dirac during the summer of
    2015.  It doesn't apply to the color code, which will use the new
    general parallel code.
    '''
        
    ##### Defining basic stuff #####

    n_stabs = len(stabs)
    n_tot_circ = 3*n_stabs
    n_first_circ = 2*n_stabs
    #phys_dens, log_dens = fun.initial_state_general(theta, phi, code)
    
    ##### Defining the circuit and inserting the errors #####

    circ = create_FT_EC_no_time(n_data, n_ancilla, stabs, redun,
                                    alternating, with_initial_I)
    
    #brow.from_circuit(circ, True)

    fun.insert_errors(circ, error_gate, ['I','H','CX','CZ'], error_kind)
    
    #brow.from_circuit(circ, True)
    #sys.exit(0)    
    
    ##### Running the first two stabilizer rounds  #####

    t1 = t.clock()
    print 'Running first two rounds of stabilizer measurements ...'

    pre_outcomes = [['0'] for i in range(n_ancilla-1)] + [['0','1']]
    outcomes = [pre_outcomes for i in range(n_tot_circ)]

    ancilla_dens = fun.cat_state_dens_matr(n_ancilla)
    circ_sim = sim.Whole_Circuit(circ, outcomes, log_dens, ancilla_dens)

    circ_list = circ_sim.sub_circuits
    initial_state = circ_sim.initial_state_data
    


    ##############################################################3

    #fold = '/home/mau/ApproximateErrors/Simulation/partial_dens/'
    
    #states = {}
    
    #if saved_dens:
    #   for i in range(64):
    #       bin_number = bin(i)[2:]
    #       extra_zeroes = 6 - len(bin_number)
    #       bin_number = '0'*extra_zeroes + bin_number
    #       npy_filename = fold + bin_number + '.npy'
    #       states[bin_number] = np.load(npy_filename)  

    #else:  


    print 'Starting to run n_subcircs...'   

    states = circ_sim.run_n_subcircs(n_first_circ, outcomes[:n_first_circ],
                                  initial_state, circ_list[:n_first_circ],
                  None, False, None, True)
    states = circ_sim.convert_dic_keys(states)
    norm_factor = (2**(n_ancilla-1))**(n_first_circ)
    states = fun.normalize_states(states, norm_factor)


    #fold = '/home/mau/ApproximateErrors/Simulation/partial_dens/'
    #for key in states:
    #   np.save(fold+key+'.npy', states[key])

    t2 = t.clock()
    print 'Total time for the first part: %f seconds' %(t2-t1)


    ##### Running final round #####

    print 'Running last part'

    count = 1

    n_keys = len(states)
    n_per_proc = n_keys // n_proc
    lists_keys = [states.keys()[i*n_per_proc:(i+1)*n_per_proc] 
                   for i in range(n_proc-1)]
    lists_keys += [states.keys()[(n_proc-1)*n_per_proc:]]
    list_dicts = [dict((k, states[k]) for k in list_key) 
              for list_key in lists_keys]

    print n_keys
    print n_per_proc
    print lists_keys
    print 'n proc =', n_proc
    #sys.exit(0)


    #run_final_and_corr(list_dicts[0], ancilla_dens, circ_list[n_first_circ:],
    #          take_final_outcome, 'not_defined', n_data, sym, code, length,
    #          error)

    
    pool = mp.Pool(n_proc)
    results = [pool.apply_async(run_final_and_corr, (list_dicts[i],
                              ancilla_dens,
                              circ_list[n_first_circ:],
                              take_final_outcome,
                              'not_defined', n_data,
                              sym, code, length,
                              error))
                    for i in range(n_proc)]

    pool.close()
    pool.join()
    
    results_list = [r.get() for r in results]
    
    return sum(results_list)
        



    #for key in states:
    #   print 'Running circuit %i.  Key = %s' %(count,key)

    #   t3 = t.clock()

    #   next_states = run_maj_final(key, states[key], ancilla_dens,
        #                                circ_list[n_first_circ:], take_final_outcome,
    #               'not_defined')
        #   corr_states = fun.apply_correction_to_every_state(next_states, sym, 
    #                             code, 3, error)
    #   count += 1
            #final_dens_list += [sum(corr_states.values())]
    #   final_dens += sum(corr_states.values())

    #   t4 = t.clock()

    #   print 'Total time for this key: %f seconds'     


    #t5 = t.clock()
    #print 'Total time for the last part: %f seconds' %(t5-t2)

    #final_dens = sum(final_dens_list)
    
    #return final_dens



def run_maj_vote_total_2(n_data, n_ancilla, stabs, redun, alternating, with_initial_I,
             error_gate, error_kind, log_dens, code, error='X', 
             length=3, sym=False, take_final_outcome=True):
    '''
    '''
    ##### Defining basic stuff #####

    n_stabs = len(stabs)
    n_tot_circ = 3*n_stabs
    n_first_circ = 2*n_stabs
    #phys_dens, log_dens = fun.initial_state_general(theta, phi, code)

    ##### Defining the circuit and inserting the errors #####

    circ = create_FT_EC_no_time(n_data, n_ancilla, stabs, redun,
                                    alternating, with_initial_I)
    fun.insert_errors(circ, error_gate, ['I','H','CX','CZ'], error_kind)
    

    ##### Running the first two stabilizer rounds  #####

    t1 = t.clock()
    print 'Running first two rounds of stabilizer measurements ...'

    pre_outcomes = [['0'] for i in range(n_ancilla-1)] + [['0','1']]
    outcomes = [pre_outcomes for i in range(n_tot_circ)]

    ancilla_dens = fun.cat_state_dens_matr(n_ancilla)
    circ_sim = sim.Whole_Circuit(circ, outcomes, log_dens, ancilla_dens)

    circ_list = circ_sim.sub_circuits
    initial_state = circ_sim.initial_state_data
    n_qs = len(circ_list[0].qubits())
    
    # Define State_and_Operations for every subcircuit, but without
    # initial state

    compiled_circ_list = []

    print 'Compiling the circuits...'
    ta = t.clock()  

    for i in range(len(circ_list)):

        tb = t.clock()

        compiled_circ_list += [sim.State_and_Operations(circ_list[i], 
                           n_qs, None, outcomes[i], sym)]

        tc = t.clock()
        print 'Time to compile circ %i = %f' %(i, tc-tb)
    
    td = t.clock()
    print 'Time to compile all the circuits = %f' %(td-ta)

    desired_outcomes = ['already_defined' for i in range(2*n_stabs)]
    states = circ_sim.run_n_subcircs(n_first_circ, 
                     desired_outcomes,
                                     initial_state, 
                     compiled_circ_list[:n_first_circ])
    states = circ_sim.convert_dic_keys(states)
    norm_factor = (2**(n_ancilla-1))**(n_first_circ)
    states = fun.normalize_states(states, norm_factor)

    t2 = t.clock()
    print 'Total time for the first part: %f seconds' %(t2-t1)


    ##### Running final round #####

    print 'Running last part'

    count = 1

    # The way to make it less RAM-intensive is actually extremely easy.
    # Instead of saving each one of the  matrices in a list 
    # (64 in the case of the Steane code), just store the partial sum.
    #final_dens_list = []

    entries = range(2**n_data)
    pre_matrix = [[complex(0.,0.) for i in entries] for j in entries]

    if sym:
        final_dens = mat.Matrix(pre_matrix)
    else:
        final_dens = np.matrix(pre_matrix)

    for key in states:
        print 'Running circuit %i.  Key = %s' %(count,key)

        t3 = t.clock()

        next_states = run_maj_final(key, states[key], ancilla_dens,
                                        circ_list[n_first_circ:], take_final_outcome)
        corr_states = fun.apply_correction_to_every_state(next_states, sym, 
                                  code, 3, error)
        count += 1
            #final_dens_list += [sum(corr_states.values())]
        final_dens += sum(corr_states.values())

        t4 = t.clock()

        print 'Total time for this key: %f seconds'     


    t5 = t.clock()
    print 'Total time for the last part: %f seconds' %(t5-t2)

    #final_dens = sum(final_dens_list)
    
    return final_dens



def run_maj_vote_total_3(n_data, n_ancilla, stabs, redun, alternating, with_initial_I,
             error_gate, error_kind, log_dens, code, error='X', 
             length=3, sym=False, take_final_outcome=True):
    '''
    '''
    ##### Defining basic stuff #####

    n_stabs = len(stabs)
    n_tot_circ = 3*n_stabs
    n_first_circ = 2*n_stabs
    #phys_dens, log_dens = fun.initial_state_general(theta, phi, code)

    ##### Defining the circuit and inserting the errors #####

    circ = create_FT_EC_no_time(n_data, n_ancilla, stabs, redun,
                                    alternating, with_initial_I)
    fun.insert_errors(circ, error_gate, ['I','H','CX','CZ'], error_kind)
    

    ##### Running the first two stabilizer rounds  #####

    t1 = t.clock()
    print 'Running first two rounds of stabilizer measurements ...'

    pre_outcomes = [['0'] for i in range(n_ancilla-1)] + [['0','1']]
    outcomes = [pre_outcomes for i in range(n_tot_circ)]

    ancilla_dens = fun.cat_state_dens_matr(n_ancilla)
    circ_sim = sim.Whole_Circuit(circ, outcomes, log_dens, ancilla_dens)

    circ_list = circ_sim.sub_circuits
    initial_state = circ_sim.initial_state_data
    n_qs = len(circ_list[0].qubits())
    
    states = circ_sim.run_n_subcircs(n_first_circ, 
                     outcomes[:n_first_circ],
                                     initial_state, 
                     circ_list[:n_first_circ])
    states = circ_sim.convert_dic_keys(states)
    norm_factor = (2**(n_ancilla-1))**(n_first_circ)
    states = fun.normalize_states(states, norm_factor)

    t2 = t.clock()
    print 'Total time for the first part: %f seconds' %(t2-t1)


    ##### Running final round #####

    print 'Running last part'

    count = 1

    # The way to make it less RAM-intensive is actually extremely easy.
    # Instead of saving each one of the  matrices in a list 
    # (64 in the case of the Steane code), just store the partial sum.
    #final_dens_list = []

    entries = range(2**n_data)
    pre_matrix = [[complex(0.,0.) for i in entries] for j in entries]

    if sym:
        final_dens = mat.Matrix(pre_matrix)
    else:
        final_dens = np.matrix(pre_matrix)


    # Define State_and_Operations for every subcircuit, but without
    # initial state

    ###################################################################
    compiled_circ_list = []

    #print 'Compiling the circuits...'
    #ta = t.clock() 

    for i in range(n_first_circ, n_tot_circ):

    #   tb = t.clock()

        compiled_circ_list += [sim.State_and_Operations(circ_list[i], 
                           n_qs, None, outcomes[i], sym)]

    #   tc = t.clock()
    #   print 'Time to compile circ %i = %f' %(i, tc-tb)
    
    #td = t.clock()
    #print 'Time to compile all the circuits = %f' %(td-ta)

    #desired_outcomes = ['already_defined' for i in range(2*n_stabs)]
    
    ###################################################################

    for key in states:
        print 'Running circuit %i.  Key = %s' %(count,key)

        t3 = t.clock()

        next_states = run_maj_final(key, states[key], ancilla_dens,
                                        compiled_circ_list, take_final_outcome)
        corr_states = fun.apply_correction_to_every_state(next_states, sym, 
                                  code, 3, error)
        count += 1
            #final_dens_list += [sum(corr_states.values())]
        final_dens += sum(corr_states.values())

        t4 = t.clock()

        print 'Total time for this key: %f seconds' %(t4-t3)        


    t5 = t.clock()
    print 'Total time for the last part: %f seconds' %(t5-t2)

    #final_dens = sum(final_dens_list)
    
    return final_dens



def run_maj_vote_total_4(n_data, n_ancilla, stabs, redun, alternating, with_initial_I,
             error_gate, error_kind, log_dens, code, error='X', 
             length=3, sym=False, take_final_outcome=True, n_parallel=4):
    '''
    '''
    ##### Defining basic stuff #####

    n_stabs = len(stabs)
    n_tot_circ = 3*n_stabs
    n_first_circ = 2*n_stabs
    #phys_dens, log_dens = fun.initial_state_general(theta, phi, code)

    ##### Defining the circuit and inserting the errors #####

    circ = create_FT_EC_no_time(n_data, n_ancilla, stabs, redun,
                                    alternating, with_initial_I)
    fun.insert_errors(circ, error_gate, ['I','H','CX','CZ'], error_kind)
    

    ##### Running the first two stabilizer rounds  #####

    t1 = t.clock()
    print 'Running first two rounds of stabilizer measurements ...'

    pre_outcomes = [['0'] for i in range(n_ancilla-1)] + [['0','1']]
    outcomes = [pre_outcomes for i in range(n_tot_circ)]

    ancilla_dens = fun.cat_state_dens_matr(n_ancilla)
    circ_sim = sim.Whole_Circuit(circ, outcomes, log_dens, ancilla_dens)

    circ_list = circ_sim.sub_circuits
    initial_state = circ_sim.initial_state_data
    n_qs = len(circ_list[0].qubits())
    
    states = circ_sim.run_n_subcircs(n_first_circ, 
                     outcomes[:n_first_circ],
                                     initial_state, 
                     circ_list[:n_first_circ])
    states = circ_sim.convert_dic_keys(states)
    norm_factor = (2**(n_ancilla-1))**(n_first_circ)
    states = fun.normalize_states(states, norm_factor)

    t2 = t.clock()
    print 'Total time for the first part: %f seconds' %(t2-t1)


    ##### Running final round #####

    print 'Running last part'

    compiled_circ_list = []

    #print 'Compiling the circuits...'
    #ta = t.clock() 

    for i in range(n_first_circ, n_tot_circ):

    #   tb = t.clock()

        compiled_circ_list += [sim.State_and_Operations(circ_list[i], 
                           n_qs, None, outcomes[i], sym)]
    #   tc = t.clock()
    #   print 'Time to compile circ %i = %f' %(i, tc-tb)
    
    #td = t.clock()
    #print 'Time to compile all the circuits = %f' %(td-ta)

    #desired_outcomes = ['already_defined' for i in range(2*n_stabs)]
    
    ###################################################################

    def run_one_state(key_and_states=[]):
        '''
        function to be used by parfor
        '''
        binary_key = key_and_states[0]
        data_density = key_and_states[1]

        next_states = run_maj_final(binary_key, data_density, ancilla_dens,
                        compiled_circ_list, take_final_outcome)
        corr_states = fun.apply_correction_to_every_state(next_states, sym, 
                                  code, 3, error)
        return sum(corr_states.values())


    # Define final density matrix

    entries = range(2**n_data)
    pre_matrix = [[complex(0.,0.) for i in entries] for j in entries]

    if sym:
        final_dens = mat.Matrix(pre_matrix)
    else:
        final_dens = np.matrix(pre_matrix)

    ######################################################################
    
    
    # Run the parfors

    states_in_list_format = [[key, states[key]] for key in states]

    seq_runs = int(64/n_parallel)
    for i in range(seq_runs):
        
        dens_list = qt.parfor(run_one_state, 
                          states_in_list_format[i*n_parallel:(i+1)*n_parallel])
    
        final_dens += sum(dens_list)

    ######################################################################

    return final_dens



def distance(phys_dens, log_dens, strength, error_gate, output='distance',
         code='Steane', error_kind=1, EC_type='pseudo', logical_operator=None,
         majority_vote=True, alternating=True, averaging=True, 
         save_final_dens_matr=True, dens_matr_folder=None, stab_rotation=None,
         rot_where='errors'):
    '''
    Inputs: - phys_dens: 1-qubit density matrix
        - log_dens: its encoded version
        - error_gate: - the error_gate as a string (i.e. 'AD 0.1')
                  - If the error_gate is different for the physical
                and logical levels, then it is a list or a tuple 
                of two strings.
        - output: either 'distance' or 'overlap'    

        - logical_operator: used only if the code is BS.
                    This is 'Z' if the initial state is |0>,
                            'X' if the initial state is |+>, and so on
         Eventually logical_operator will be a matrix (the 10-qubit control-
         logical operator)
    
        - stab_rotation: a 1-qubit rotation operator (numpy matrix)
        - rot_where:  'errors' or 'stabs'

    Outputs: - The physical and logical distances or overlaps. 
    Right now, it's specific for the Steane code and only for numerical
    calculations.
    '''

    ##### Assigning the physical and logical gates #####

    # if the error gate is a list
    if type(error_gate) == type([]) or type(error_gate) == type(()):
        error_gate_phys = error_gate[0]
        error_gate_log = error_gate[1]
    else:
        error_gate_phys = error_gate
        error_gate_log = error_gate 

    if stab_rotation != None and rot_where == 'errors':
        error_gate_log += ' r'      

    ##### Defining the physical circuit #####

    circ = c.Circuit()
    circ.add_gate_at([0], error_gate_phys)
    circ_sim = sim.State_and_Operations(circ, 1, phys_dens, [None], False)
    circ_sim.apply_all_operations()
    final_state1 = circ_sim.current_state.density_matrix


    ##### Defining the logical circuit #####

    # Still need to change BS #

    if code == 'BS':
        circZ = create_nonFT_EC(9, BS_stabilizers[:6])
        circX = create_nonFT_EC(9, BS_stabilizers[6:], False)
        fun.insert_errors(circZ, error_gate_log, ['I','H','CX','CZ'], 
                  error_kind)
        fun.insert_errors(circX, error_gate_log, ['I','H','CX','CZ'], 
                  error_kind)
        des_outcomesZ = [[['0','1']] for i in range(6)]
        des_outcomesX = [[['0','1']] for i in range(6)]
        ancilla_ket = fun.state_vector_dic['PrepareXPlus']
        ancilla_dens = ancilla_ket*(ancilla_ket.H)

        #browser.from_circuit(circX, True)

        circ_simZ = sim.Whole_Circuit(circZ, des_outcomesZ, 
                          log_dens, ancilla_dens)
        statesZ = circ_simZ.run_all_in_tree()
        corr_statesZ = fun.apply_correction_to_every_state_BS(statesZ, 'X')
        final_stateZ = sum(corr_statesZ.values())

        circ_simX = sim.Whole_Circuit(circX, des_outcomesX, final_stateZ, 
                          ancilla_dens)
        statesX = circ_simX.run_all_in_tree()
        corr_statesX = fun.apply_correction_to_every_state_BS(statesX, 'Z')
        final_stateX = sum(corr_statesX.values())

        circ_check = create_nonFT_EC(9, BS_stabilizers_4)
        des_outcomes_check = [[['0']] for i in range(4)]
        circ_check_sim = sim.Whole_Circuit(circ_check, des_outcomes_check, 
                           final_stateX, ancilla_dens)
        checked_state1 = circ_check_sim.run_all_in_tree().values()[0]

        checked_state2 = fun.measure_logical_stabilizer(9, checked_state1,
                                    logical_operator)

        final_state2 = checked_state2

        fidelity = np.trace(checked_state2).real

        overlap1 = fun.overlap(phys_dens, final_state1)

        if output == 'distance':      out1, out2 = 1-overlap1, 1-fidelity
        elif output == 'overlap':     out1, out2 = overlap1, fidelity
        
        
        # Implement BS with None EC


    elif code == 'Steane':

        if EC_type == 'pseudo':
            circ2 = create_nonFT_EC(7, st.Code.stabilizer, True,
                        stab_rotation, rot_where)
            
            #brow.from_circuit(circ2, True)         

            fun.insert_errors(circ2, error_gate_log, 
                      ['I','H','CX','CZ','C','dens'], 
                      error_kind)
            
            #brow.from_circuit(circ2, True)     
    
            # if stab_rotation != None, then we also have these
            # gates: 'C' and 'dens'.
            # If error_kind == 1, then it doesn't really matter.
            outcomes = [[['0','1']] for i in range(6)]
            ancilla_ket = fun.state_vector_dic['PrepareXPlus']
            ancilla_dens = ancilla_ket*(ancilla_ket.H)

            if stab_rotation == None or rot_where == 'stabs':
                rot_errors = None
                corr_rotation = stab_rotation
            else:
                # The errors are rotated by the inverse.
                rot_errors = stab_rotation.H
                corr_rotation = None
         
            circ_sim2 = sim.Whole_Circuit(circ2, outcomes, 
                              log_dens, ancilla_dens,
                              None, None, False,
                              rot_errors)
            states = circ_sim2.run_all_in_tree()
           
            corrected_states = fun.apply_correction_to_every_state(
                            states, False, code,
                            6, None, corr_rotation)
            # only if it is symbolic:
            #final_state2 = sim.sum_matrices(corrected_states)
            final_state2 = sum(corrected_states.values())

        elif EC_type == 'real':
            if majority_vote:
                n_data, n_ancilla = 7, 4
                stabs = st.Code.stabilizer
                X_stabs, Z_stabs = stabs[:3], stabs[3:]
                error1, error2 = 'Z', 'X'
                redun = 3
                with_initial_I = True
            
                densX = run_maj_vote_total(n_data, n_ancilla, X_stabs,
                               redun, alternating, 
                               with_initial_I,
                               error_gate_log, 
                               error_kind,
                               log_dens, code, error1)

                #np.save('/home/mau/test_partial.npy', densX)
                #sys.exit(0)

                if save_final_dens_matr:
                    if not os.path.exists(dens_matr_folder):
                        os.makedirs(dens_matr_folder)
                    filename = ''.join([dens_matr_folder, 
                                str(strength), 
                                'partial.npy'])
                    np.save(filename, densX)
                    
    
                final_state2 = run_maj_vote_total(n_data, n_ancilla, 
                                  Z_stabs, redun, 
                                  alternating, 
                                  with_initial_I,
                                  error_gate_log, 
                                  error_kind,
                                  densX, code, 
                                  error2)

                #circ2 = create_FT_EC_no_time(7,4,st.Code.stabilizer,
                #                 3, alternating)
                #fun.insert_errors(circ2, error_gate, 
                #          ['I','H','CX','CZ'], error_kind)
                # The outcomes are specified later on in the process
                # We just defined it as a placeholder.
                #outcomes = [None for i in range(18)]
                #ancilla_dens = fun.cat_state_dens_matr()
            
                #circ2_subcircs = fun.split_circuit(circ2, 4)
                #circX = circ2_subcircs[:9]
                #circZ = circ2_subcircs[9:]

                #print 'Running circuit X'
    
                #circX_sim = sim.Whole_Circuit(circX, outcomes[:9],
                #                 log_dens, ancilla_dens)

                #circ_list = circX_sim.sub_circuits
                #initial_state = circX_sim.initial_state_data
    
                #states = circX_sim.run_n_subcircs(circ_list[:6], 
                #                  initial_state)
                #states = circX_sim.convert_dic_keys(states)
                #states = fun.normalize_states(states, 8**6)

                

                ###########################################

                #circ2 = create_FT_EC_no_time(7,4,st.Code.stabilizer,
                #                 3)
                #fun.insert_errors(circ2, error_gate, 
                #          ['I','H','CX','CZ'], error_kind)
                # The outcomes are specified later on in the process
                # We just defined it as a placeholder.
                #outcomes = [None for i in range(18)]
                #ancilla_dens = fun.cat_state_dens_matr()
            
                #circ2_subcircs = fun.split_circuit(circ2, 4)
                #circX = circ2_subcircs[:9]
                #circZ = circ2_subcircs[9:]

                #print 'Running circuit X'

                #circX_sim = sim.Whole_Circuit(circX, outcomes[:9], 
                #                 log_dens, ancilla_dens)
                #final_states_X = circX_sim.run_3_stab_majority_vote()
                #corrected_states_X = 
                #   fun.apply_correction_to_every_state(
                #       final_states_X, False, 'Steane',
                #       3, 'Z')
            
                #final_state_X = sum(corrected_states_X.values())
            

                #print 'Running circuit Z'

                #circZ_sim = sim.Whole_Circuit(circZ, outcomes[9:], 
                #                  final_state_X, 
                #                  ancilla_dens,
                #                      False, 11)
                #final_states_Z = circZ_sim.run_3_stab_majority_vote()
                #corrected_states_Z = 
                #   fun.apply_correction_to_every_state(
                #       final_states_Z, False, 'Steane',
                #       3, 'X')
                #final_state2 = sum(corrected_states_Z.values())

            else:
                if averaging:
                    print 'No majority averaged'
                    circ2 = create_FT_EC_no_time(7,4,
                                st.Code.stabilizer)
                    fun.insert_errors(circ2, error_gate_log, 
                        ['I','H','CX','CZ'], error_kind)
                    outcomes = [[['0'],['0'],['0'],['0','1']] 
                            for i in range(6)]
                    ancilla_dens = fun.cat_state_dens_matr()
            
                    #brow.from_circuit(circ2, True)
                    
                    circ2_subcircs = fun.split_circuit(circ2, 4)
                    circX = circ2_subcircs[:3]
                    circZ = circ2_subcircs[3:]
                
                    print 'Running circuit X'
    
                    circX_sim = sim.Whole_Circuit(circX, 
                            outcomes[:3], log_dens, 
                            ancilla_dens)
                    final_states_X = circX_sim.run_all_in_tree()
                    corrected_states_X = fun.apply_correction_to_every_state(
                            final_states_X, False, 
                            'Steane', 3, 'Z')
                                        
                    final_state_X = sum(corrected_states_X.values())
                
                    # Because for each one of the 3 stabilizers we are just
                    # considering 2 out of the 16 possible outcomes (1/8),
                    # we need to renormalize the final density matrix by 8**3
                    final_state_X = (8**3)*final_state_X        
    
                    print 'Trace of the final X state =', np.trace(final_state_X)


                    print 'Running circuit Z'

                    circZ_sim = sim.Whole_Circuit(circZ, 
                                outcomes[3:], 
                                final_state_X, 
                                ancilla_dens, 
                                False, 11)
                    final_states_Z = circZ_sim.run_all_in_tree()
                    corrected_states_Z = fun.apply_correction_to_every_state(
                            final_states_Z, False, 
                            'Steane', 3, 'X')
                    final_state2 = sum(corrected_states_Z.values())
                    
                    # Same as for the X matrix
                    final_state2 = (8**3)*final_state2

                    print 'Trace of the final Z state =', np.trace(final_state2)
                    

                else:
                    circ2 = create_FT_EC_no_time(7,4,
                                st.Code.stabilizer) 
                    fun.insert_errors(circ2, error_gate_log, 
                           ['I','H','CX','CZ'], error_kind)
                    outcomes = [[['0'],['0'],['0'],['0','1']] 
                             for i in range(6)]
                    ancilla_dens = fun.cat_state_dens_matr()
                    circ2_sim = sim.Whole_Circuit(circ2, 
                            outcomes, log_dens, 
                            ancilla_dens)
                    states = circ2_sim.run_all_in_tree()
                    norm_states = fun.normalize_cat_state_matrices(states)
                    corrected_states = fun.apply_correction_to_every_state_cat_ancilla(norm_states)
                    final_state2 = sum(corrected_states.values())

    
            
        elif EC_type == None:
            circ2 = error(error_gate_log)
            circ2_sim = sim.State_and_Operations(circ, 7, 
                            log_dens, [None], False)
            circ2_sim.apply_all_operations()
            final_state2 = circ2_sim.current_state.density_matrix

        if output=='distance':
            out1 = fun.trace_distance(phys_dens, final_state1)
            out2 = fun.trace_distance(log_dens, final_state2)
    
        else: 
            if output == 'overlap':  fid = False
            elif output == 'fidelity':  fid = True 
            out1 = 1 - fun.overlap(final_state1, [phys_dens], fid) 
            log_dens_matrs = fun.get_one_error_dens_matrs(log_dens, 7)
            out2 = 1 - fun.overlap(final_state2, log_dens_matrs, fid)

    
    #np.save('/home/mau/test_final.npy', final_state2)


    if save_final_dens_matr:
        if not os.path.exists(dens_matr_folder):
            os.makedirs(dens_matr_folder)
        dens_filename = dens_matr_folder + str(strength) + '.npy'
        np.save(dens_filename, final_state2)
        # Just to compare size of .npy vs. .gz files
        # Nevermind. These .gz files don't seem to be able to store
        # imaginary values.
        #alt_dens_filename = dens_matr_folder + str(strength) + '.gz'
        #np.savetxt(alt_dens_filename, final_state2)            
            
    return out1, out2



def distances_for_various_noise_strengths(phys_dens, log_dens, list_strengths,
            list_error_gates, serialize_results=True, results_folder='.', 
        json_filename='distances.json', error_kind=1, EC_type='pseudo', 
        output='distance', code='Steane', logical_operator=None, 
        majority_vote=True, averaging=True, save_final_dens_matr=True, 
        dens_matr_folder=None, stab_rotation=None, rot_where='errors'):
    '''
    Inputs: - phys_dens: 1-qubit density matrix
        - log_dens: its encoded version
        - list_strengths: list of noise strengths
        - list_error_gates: list of error gates (as many as strengths)
                    if we want the physical level to have different
                    gates from the logical level, then it would be
                    a list of 2 lists:
                    [['AD 0', 'AD 0.1'], ['PCa 0', 'PCa 0.1']]
        - error_kind: 1, 2, or 3 (with the same meaning as in the BV circuits)
        - EC_type: 'pseudo' for 1-qubit ancilla
               'real' for cat-state ancilla
        - output: 'distance' or 'overlap'
        - code: QECC to use ('Steane' or 'BS')
        - serialize_results: whether or not we want to dump the results on a
                             json file to store them for future use
        - json_filename: name of the json file where the results are stored
        - stab_rotation: 1-qubit rotation matrix to be applied to the 
                 stabilizers.
        - rot_where: whether we want to apply the rotation to the 'errors'
                 or 'stabs'.  It's much faster to apply it to the
                 'errors'.
    '''
    dist_dict = {}
    list_phys_dist, list_log_dist = [], []

    # if list_error_gates has already a list of physical errors and 
    # logicals errors, then just 
    if type(list_error_gates[0]) == type([]):
        error_gates = zip(*list_error_gates)

    for i in range(len(list_strengths)):
        print 'strength =', list_strengths[i]
        error_gate = error_gates[i]
        phys_dist, log_dist = distance(phys_dens, log_dens, 
                           list_strengths[i], error_gate, 
                           output, code, error_kind, EC_type, 
                           logical_operator, majority_vote, 
                           True, averaging, save_final_dens_matr,
                           dens_matr_folder, stab_rotation,
                           rot_where)
        real_phys_dist = phys_dist.real
        real_log_dist = log_dist.real
        list_phys_dist += [real_phys_dist]
        list_log_dist += [real_log_dist]
        dist_dict[list_strengths[i]] = {'p':real_phys_dist, 'l':real_log_dist}

    if serialize_results:
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        json_filename = results_folder + json_filename
        json_file = open(json_filename, 'w')
        json.dump(dist_dict, json_file, indent=4, 
              separators=(',', ' : '), sort_keys=True)
        json_file.close()       
    
    return list_phys_dist, list_log_dist
    


def apply_error(init_dens, error_gate, phys_or_log='phys', 
        code='Steane', rot_errors=None):
    '''
    '''
    if phys_or_log == 'phys':  n_qs = 1
    elif phys_or_log == 'log':
        if code == 'Steane':  n_qs = 7
        elif code == '5qubit': n_qs = 5 
    
    circ = c.Circuit()
    for i in range(n_qs):
        circ.add_gate_at([i], error_gate)
    circ_sim = sim.State_and_Operations(circ, n_qs, init_dens,
                        [None], False, rot_errors)
    circ_sim.apply_all_operations()

    return circ_sim.current_state.density_matrix
        


def correct_Steane(initial_dens):
    '''
    '''
    circ_correct = create_nonFT_EC_for_Steane()
    outcomes = [[['0','1']] for i in range(6)]
    ancilla_ket = fun.state_vector_dic['PrepareXPlus']
    ancilla_dens = ancilla_ket*(ancilla_ket.H)
    circ_correct_sim = sim.Whole_Circuit(circ_correct, 
                         outcomes, 
                         initial_dens,
                         ancilla_dens)
    states = circ_correct_sim.run_all_in_tree()
    corrected_states = fun.apply_correction_to_every_state(
                            states)
    # only if it is symbolic:
    #final_state2 = sim.sum_matrices(corrected_states)
    corrected_log_dens = sum(corrected_states.values())
        
    # Let's normalize the final dens matrix just in case
    trace = np.trace(corrected_log_dens).real
    corrected_log_dens = (1./trace)*corrected_log_dens
    
    return corrected_log_dens



def perfect_EC(initial_dens, code='Steane', syndrome='average', ns=6):
    '''
    Function used to quickly calculate the final states
    after perfect EC.  Written during Joel Wallman's visit
    3/30/2016.  :)

    syndrome: 'average' if we want to average every syndrome branch
              'postselect' if we want to only take the error-free syndrome
    '''

    if syndrome == 'average':
        outcomes = [[['0','1']] for i in range(ns)]
    elif syndrome == 'postselect':
        outcomes = [[['0']] for i in range(ns)]

    if code == 'Steane':
        circ_correct = create_nonFT_EC(7, st.Code.stabilizer)

    elif code == '5qubit':
        circ_correct = create_nonFT_EC(5, fivequbit.Code.stabilizer)
       
    ancilla_ket = fun.state_vector_dic['PrepareXPlus']
    ancilla_dens = ancilla_ket*(ancilla_ket.H)
    circ_correct_sim = sim.Whole_Circuit(circ_correct, 
                         outcomes, 
                         initial_dens,
                         ancilla_dens)
    states = circ_correct_sim.run_all_in_tree()
    corrected_states = fun.apply_correction_to_every_state(
                            states, False, code)
    
    corrected_log_dens = sum(corrected_states.values())
        
    # Let's normalize the final dens matrix just in case
    trace = np.trace(corrected_log_dens).real
    corrected_log_dens = (1./trace)*corrected_log_dens
    
    return corrected_log_dens, trace
    




def compute_final_phys_and_log_states(theta, phi, errors, strength,
                      theta_vector=0., phi_vector=0.,
                      angle=None, code='Steane',
                      rot_error=None, to_phys_too=False):
    '''
    errors can be a string if we want to apply the same error to both the
    physical and the logical circuits or a list or tuple if we want to
    apply different errors
    '''
    # First we calculate the final physical state.
    phys_dens, log_dens = fun.initial_state_general_different_basis(
                theta_vector, phi_vector, theta, phi, code)

    if type(errors) != type([]) and type(errors) != type(()):
        errors = [errors, errors] 

    error_gates = []
    for error in errors:
        if error == 'Pauli' and type(strength) != list:
            strengths = map(str,[strength for i in range(3)])
            error_and_strengths = [error] + strengths 
            error_gate = ' '.join(error_and_strengths)
        else:
            error_gate = error + ' ' + str(strength)
            if angle != None:   
                error_gate += ''.join([' ', str(angle)])
        
        error_gates += [error_gate]         
    
    # uncorrected logical state
    final_uncorr_log_dens = apply_error(log_dens, error_gates[1], 
                        'log', code, rot_error)

    # physical state
    if to_phys_too: 
        final_phys_dens = apply_error(phys_dens, error_gates[0], 
                          'phys', None, rot_error)
    else:
        final_phys_dens = apply_error(phys_dens, error_gates[0],
                          'phys')


    return (phys_dens, final_phys_dens, 
        log_dens, final_uncorr_log_dens)
   

 
def open_npy_and_calculate_distance(folder, theta, phi, code, output, error,
                                    strength, angle=None, npy_filename='',
                                    theta_vector=0., phi_vector=0.,
                                    correct=False, save_corrected_dens=True,
                                    rot_error=None, to_phys_too=False):
    '''
    folder:    the absolute path to the folder where the npy file is located
    theta:
    phi:       the parameters that represent the state on the Bloch sphere
    code:      the QECC ('Steane' or 'BS')
    output:    'distance' or 'overlap': 'overlap' will be preferably used
               because of consistency (I haven't figured a way to do the BS
               or the error3 comparison with the ditance.)
    error:     name of the error, i.e., 'AD', 'CMCapproxAD_cons', etc.
    strength:  the noise strength (the name of the npy file will be gamma.npy)
    angle:     None if working with amplitude damping.
               If working with PolXY, angle is the angle of the polarization 
               axis.
        
    For the 'distance' case we have the following outputs:
    phys:        trace distance between initial and final physical 
                 (unencoded) states.
    uncorr_log:  trace distance between initial logical state and 
                 uncorrected logical state (initial logical state with 
                 a physical error channel applied on each physical qubit).
    log:         trace distance between initial logical state and logical 
                 state after EC (perfect EC for CC and faulty EC for CB).
    corr_log:    trace distance between initial logical state and logical
                 state after EC (which may or may not be faulty) and 
                 perfect EC.  Notice that for the CB case, log and 
                 corr_log have the same value because the first EC was 
                 already perfect.
    '''
 
    (phys_dens,
     final_phys_dens,
     log_dens, 
     final_uncorr_log_dens) = compute_final_phys_and_log_states(theta, 
                                    phi, 
                                    error, 
                                    strength,
                                    theta_vector,
                                    phi_vector, 
                                    angle,
                                    code,
                                    rot_error,
                                    to_phys_too)
    if save_corrected_dens:

        if type(npy_filename) == type([]):
            uncorr_filename = ''.join([folder, '',
                                   str(strength),
                                   'uncorrected.npy'])  
        else:
            uncorr_filename = ''.join([folder, npy_filename,
                                   str(strength),
                                   'uncorrected.npy'])  

        np.save(uncorr_filename, final_uncorr_log_dens)

             
    # Then we load the final logical state.
    if type(npy_filename) == type([]):
        dens_perf_file = folder + str(strength) + npy_filename[0] + '.npy'
        dens_fault_file = folder + str(strength) + npy_filename[1] + '.npy'
        dens_perf = fun.convert_mat(np.load(dens_perf_file), code, False)	
        dens_fault = fun.convert_mat(np.load(dens_fault_file), code, False)	


    else:
        dens_filename = folder + npy_filename + str(strength) + '.npy'
        final_log_dens = np.load(dens_filename)

        # Let's normalize the final dens matrix
        trace = np.trace(final_log_dens).real
        final_log_dens = (1./trace)*final_log_dens
    
    if correct:

        
        if type(npy_filename) == type([]):
            corr_filenames_dict = {
                        'corrected_perf': dens_perf, 
                        'corrected_fault': dens_fault
                         }
        
        else:

            corr_filenames_dict = {
                        'corrected_perf': correct_Steane(
                                          final_uncorr_log_dens), 
                        'corrected_fault': correct_Steane(
                                           final_log_dens)
                          }
        
            if save_corrected_dens:
            
                for corr_filename in corr_filenames_dict:
                    corr_filepath = ''.join([folder, 
                                             npy_filename, 
                                             str(strength), 
                                             corr_filename,
                                             '.npy']) 
            
                    np.save(corr_filepath, 
                            corr_filenames_dict[corr_filename])
            


    # Finally we calculate the physical and logical distances or 
    # overlaps.
    if output == 'distance':
        print 'Doing distance'
                
        phys = fun.trace_distance(phys_dens, final_phys_dens)
        uncorr_log = fun.trace_distance(log_dens, 
                             final_uncorr_log_dens)
        
        if type(npy_filename) == type([]):
            log = ''
        else:    
            log = fun.trace_distance(log_dens, final_log_dens)
        
        if correct:
            corr_log_perf = fun.trace_distance(log_dens, 
                    corr_filenames_dict['corrected_perf'])
            corr_log_fault = fun.trace_distance(log_dens,
                    corr_filenames_dict['corrected_fault'])

            return phys, uncorr_log, log, corr_log_perf, corr_log_fault 

        else:
            return phys, uncorr_log, log



    else: 
        if output == 'overlap':  fid = False
        elif output == 'fidelity':  fid = True
        
        print 'Doing %s' %output

        phys = 1 - fun.overlap(phys_dens, [final_phys_dens], fid)
        uncorr_log = 1 - fun.overlap(log_dens, 
                        [final_uncorr_log_dens], fid)
        

        if type(npy_filename) == type([]):
            log = ''
        else:    
            log = 1 - fun.overlap(log_dens, [final_log_dens], fid)

        if correct:
            corr_log_perf = 1 - fun.overlap(log_dens, 
                [corr_filenames_dict['corrected_perf']], fid)
            corr_log_fault = 1 - fun.overlap(log_dens,
                [corr_filenames_dict['corrected_fault']], fid)

            return phys, uncorr_log, log, corr_log_perf, corr_log_fault 

        else:
            return phys, uncorr_log, log

        
        #fid = fun.overlap(final_phys_dens, [phys_dens]).real
                #out_phys = 1 - fid

        # still need to figure out how to do it for BS
                #log_dens_matrs = fun.get_one_error_dens_matrs(log_dens, 
        #                       7, code)

        # We create the orthogonal logical state to log_state
        # and all the 1-error density matrices.
        # We just want to check that the total overlap is 1.
        #new_theta = math.pi - theta
        #if theta == 0. or theta == math.pi:
        #        new_phi = phi
        #else:
        #        new_phi = math.pi + phi

        #log_dens_ortho = fun.initial_state_general(new_theta,new_phi,'Steane')[1]
        #log_dens_matrs_ortho = fun.get_one_error_dens_matrs(log_dens_ortho, 7)

        #print 'Logical overlap'
        
        #over_log1 = fun.overlap(final_log_dens, log_dens_matrs).real

        #print 'Logical overlap 2'

        #over_log2 = fun.overlap(final_log_dens, log_dens_matrs_ortho).real
        #out_log = 1 - over_log1

        #print 'Theta =', theta, ',', 'Phi =', phi
        #print 'Strength =', strength
        #print 'Physical =', out_phys
        #print 'Logical1 =', over_log1
        #print 'Logical2 =', over_log2
        #print 'Trace = ', np.trace(final_log_dens)

        #over = np.trace(log_dens*log_dens_ortho)
        #print 'Overlap', over

        #print '\n'

        #eigvals = np.linalg.eigvals(final_log_dens)
        #for val in eigvals:
        #       if val.real < -1.e-10 or val.real > 1.e-10:  
        #               if val.real < 0:  print 'PROBLEMS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'

        #return out_phys, out_log



def look_for_threshold_iteratively(theta, phi, error, 
                   theta_vector=0., phi_vector=0.,
                   angle=None, code='Steane',
                   min_strength=0., max_strength=1.,
                   accuracy=1.e-10, strength=None,
                   step=None, output='distance',
                   forward=False, rot_error=None,
                   to_phys_too=False):
    '''
    forward refers to the kind of algorithm:
    (1) forward == False:  we start in the median of min_strength and 
                   max_strength and we reduce the stepsize by
                   half after each repetition. The problem with
                   this approach is that it might get trapped
                   in a second (high error) threshold.
    (2) forward == True:   we start at a low error rate and move up by
                   the same amount until we the first threshold
                   is reached.  At that point we take the first
                   algorithm. This second algorithm could be 
                   further refined.  Right now we are assuming
                   that the initial strength is the same as the
                   step (0.05).
    error can be a string or a list or tuple
    '''
    dif = 1
    if strength == None:  strength = 0.5*(min_strength + max_strength)
    if step == None:      step = 0.5*(strength - min_strength)
    
    iteration = 1

    while (abs(dif) > accuracy):        

        # final physical and logical states
        (phys_dens,
         final_phys_dens,
         log_dens,
         uncorr_log_dens) = compute_final_phys_and_log_states(theta, 
                                phi, 
                                error, 
                                strength,
                                    theta_vector,
                                phi_vector, 
                                    angle,
                                code,
                                rot_error,
                                to_phys_too)
        
        corr_log_dens = correct_Steane(uncorr_log_dens)
        trace = np.trace(corr_log_dens).real
        corr_log_dens *= (1./trace)
        
        # distances
        if output == 'distance':  
            phys_dist = fun.trace_distance(phys_dens, 
                               final_phys_dens)
            log_dist = fun.trace_distance(log_dens, 
                              corr_log_dens)
        else:
            if output == 'fidelity':  fid = True
            else: fid = False
            phys_dist = 1-fun.overlap(phys_dens, 
                          [final_phys_dens], fid)
            log_dist = 1-fun.overlap(log_dens, 
                         [corr_log_dens], fid)
        
        dif = log_dist - phys_dist
    
        threshold = strength 

        if dif < 0:
            print 'Going up ...'  
            strength += step
        else:
            print 'Going down ...'
            if forward:
                forward = False
                step *= 0.5   
            strength -= step

        if not forward:
            step *= 0.5

        print 'output =', output
        print 'theta =', theta
        print 'phi =', phi
        print 'iteration =', iteration
        print 'physical distance =', phys_dist
        print 'logical distance =', log_dist
        print 'threshold =', threshold
        print 'forward =', forward
        print '\n'
        iteration += 1
        
        #t.sleep(5)

        # to avoid an infinite loop in case the threshold is larger
        # than the maximum strength
        if abs(threshold - max_strength) < accuracy:  break

    return threshold



def open_several_npys_calc_dist_and_save_in_json(folder, theta, phi, code, 
                         output, error, list_strengths,
                         angle=None, serialize=True, 
                         json_filename='overlap.json', 
                         npy_filename='', 
                         theta_vector=0., 
                         phi_vector=0., correct=False,
                         save_corrected_dens=True,
                         rot_error=None, 
                         to_phys_too=False):
    '''
    folder:    the absolute path to the folder ABOVE which the npy file 
               is located, excluding the theta and the phi
    theta:
    phi:       the parameters that represent the state on the Bloch sphere
    code:      the QECC ('Steane' or 'BS')
    output:    'distance' or 'overlap': 'overlap' will be preferably used
               because of consistency (I haven't figured a way to do the BS
               or the error3 comparison with the ditance.)
    error:     name of the error, i.e., 'AD', 'CMCapproxAD_cons', etc.
    strength:  the noise strength (the name of the npy file will be 
               gamma.npy)
    angle:     None if working with amplitude damping.
               If working with PolXY, angle is the angle of the 
               polarization axis.
        
    This function opens all the npy files that correspond to the different 
    noise strengths for a particular initial Bloch sphere point and 
    creates a json file with all the overlaps.
    '''

    new_folder = folder + '/theta' + str(theta) + '_phi' + str(phi) + '/'
    #new_folder = folder
    dist_dict = {}
    for i in range(len(list_strengths)):

        if correct:
            (phys, 
             uncorr_log, 
             log, 
             log_corr_perf,
             log_corr_fault) = open_npy_and_calculate_distance(
                                new_folder, theta, 
                                phi, code, output, 
                                error, list_strengths[i], 
                                angle, npy_filename, 
                                theta_vector, phi_vector,
                                correct, save_corrected_dens,
                                rot_error, to_phys_too)
            
            dist_dict[list_strengths[i]] = {'p': phys,
                                            'lnc': uncorr_log, 
                                            'l': log, 
                                            'lcp': log_corr_perf,
                                            'lcf': log_corr_fault
                                           }

        else:           
            (phys, 
             uncorr_log, 
             log) = open_npy_and_calculate_distance(new_folder, 
                                 theta, phi, code, output, 
                                 error, list_strengths[i], 
                                 angle, npy_filename, 
                                 theta_vector, 
                                 phi_vector,
                                 correct,
                                 False,
                                 rot_error,
                                 to_phys_too)
                
            dist_dict[list_strengths[i]] = {'p': phys,
                                            'lnc': uncorr_log, 
                                            'l': log
                                           }


    if serialize:
        json_path = new_folder[:-1] + '_' + json_filename
        json_file = open(json_path, 'w')
        json.dump(dist_dict, json_file, indent=4, separators=(',', ':'),
                          sort_keys=True)
        json_file.close()

    return None



def generate_distances_and_save_in_json(Bloch_points, code, EC_type, error, 
                    error_kind, folder, list_strengths, 
                    list_error_gates=None, output='overlap', 
                    angle=None, majority_vote=True,
                    averaging=True, 
                    save_final_dens_matr=True,
                        theta_vector=0., phi_vector=0.,
                    stabs_rot=None, rot_where='errors'):
    '''
    error can be a single error if we want to apply the same error to the 
    physical and logical circuits or a 2-entry list if we want to apply 
    different errors
    stabs_rot is None if no rotation is to be applied or the rotation.
    rot_where is 'stabs' if we want the rotation to be applied to the 
    QECC or 'errors' if we want the rotation to be applied to the errors.
    '''

    if type(error) == type([]) or type(error) == type(()):
        error_phys, error_log = error[0], error[1]
    else:
        error_phys, error_log = error, error
        
    if list_error_gates == None:
        error_gates_phys = generate_list_error_gates(error_phys, 
                                 list_strengths, 
                                 angle)
        error_gates_log = generate_list_error_gates(error_log,
                                list_strengths,
                                angle)
    elif len(list_error_gates) == 1:
        error_gates_phys = list_error_gates
        error_gates_log = list_error_gates
    
    elif len(list_error_gates) == 2:
        error_gates_phys = list_error_gates[0]
        error_gates_log = list_error_gates[1]

    
    for theta in Bloch_points:
        for phi in Bloch_points[theta]:
     
            print 'processor time = %f' %t.clock()
            
            if code == 'BS':
                Z_stab = BS_logical_opers['Z']
                Z_eigsys = BS_logical_opers_eigensystem['Z']
                Y_eigsys = BS_logical_opers_eigensystem['Y']
                
                rot_theta = fun.rotate(Y_eigsys['vecs'],
                               Y_eigsys['vals'],
                               theta, Z_stab)

                rot_phi = fun.rotate(Z_eigsys['vecs'],
                             Z_eigsys['vals'],
                             phi, rot_theta)

                log_operator = rot_phi

                stabs_rot_list = [stabs_rot for i in range(9)]

            elif code == 'Steane': 
                log_operator = None
                stabs_rot_list = [stabs_rot for i in range(7)]
        
            if stabs_rot == None or rot_where == 'errors':
                stabs_rot_all_qs = None
            else:
                stabs_rot_all_qs = fun.tensor_product(stabs_rot_list)

            print 'Theta =', theta
            print 'Phi =', phi

            #if theta == 0.:  continue

            dens_matrs = fun.initial_state_general_different_basis(
                        theta_vector, phi_vector,
                        theta, phi, code, False,
                        stabs_rot_all_qs)
            phys_dens, log_dens = dens_matrs

            #dens_non_rot = fun.initial_state_general_different_basis(
            #           theta_vector, phi_vector,
            #           theta, phi, code)
            #log_dens_non_rot = dens_non_rot[1]

            #print 'distance =', fun.trace_distance(log_dens_non_rot,
            #                   log_dens)

            #filename = 'theta'+str(theta)+'_phi'+str(phi)+'maj_vote_'\
            #      +str(majority_vote)+'_aver_'+str(averaging)\
            #      +'.json'
            filename = ''.join(['theta', str(theta), '_phi', 
                                str(phi), '.json']) 
            #total_filename = folder+filename
            dens_matr_folder = ''.join([folder, 'theta',
                                        str(theta), '_phi',
                                        str(phi), '/'])

            error_gates = [error_gates_phys, error_gates_log]
            
            list_phys, list_log = distances_for_various_noise_strengths(
                                phys_dens, log_dens, list_strengths, error_gates,
                                True, folder, filename, error_kind, EC_type, output, 
                                code, log_operator, majority_vote, averaging,
                                save_final_dens_matr, dens_matr_folder, stabs_rot,
                                rot_where)

            print list_phys, list_log



def check_logical_honesty(folder, errors, EC_type, error_kind, theta, phi,
              strengths, output, basis='majority_voting', 
              limit=1.e-12, main_error='AD', want_dict=True,
              extra_folder=None):
    '''
    basis: 'majority_voting' or 'majority_voting_rotated_basis'
    want_dict:  True if we want to make theta substitutions
            False if we don't
    '''

    #if output == 'overlap':
    #    levels = ['p', 'l']
    #elif output == 'distance' or output == 'fidelity':
    if True: 
        if EC_type == 'pseudo':
            levels = ['p', 'lnc', 'l']
        elif EC_type == 'real':
            #levels = ['p', 'lnc', 'l', 'lcp', 'lcf']
            levels = ['p', 'lnc', 'lcp', 'lcf']

    theta_dict = {0.: 0.,
              m.pi/4: m.pi/4,
              m.pi/2: m.pi/2,
                  3*m.pi/4: m.pi/4,
                  m.pi: 0.}

    # For the CMC cases we don't need this dictionary, but it does not
    # affect us either.

    output_dict = {}
    for strength in strengths:
        #print 'Strength = ', strength
        dict1 = {}
        for level in levels:
            print 'Level =', level
            dict2 = {}
            for error in errors:
                print 'Error =', error
                if main_error == 'PolXY' and want_dict:
                    new_theta = theta_dict[theta]
                else:
                    cond1 = error[:1] == 'P'
                    cond2 = error[:3] == 'Dep'
                    if (cond1 or cond2) and want_dict:
                        new_theta = theta_dict[theta]
                    else:
                        new_theta = theta
                file_folder = '/'.join([folder,error,
                            EC_type,
                            str(error_kind), 
                            basis, 
                            ])
                if file_folder[-1] != '/':
                    file_folder += '/'
                if extra_folder != None:
                    file_folder += '/'.join([extra_folder,
                                 ''])
                filename = ''.join(['theta', 
                            str(new_theta), 
                            '_', 'phi', str(phi), 
                            '_', output, '.json'])
                filepath = file_folder + filename   
                json_file = open(filepath, 'r')
                json_dict = json.load(json_file)
                json_file.close()
                if error == 'CMCapproxRH_uncons' or error == 'CMCapproxRH_cons':
                    if strength < 1.e-3:  dist = json_dict['%.15f' %strength][level]
                    elif strength < 1.e-2:  dist = json_dict['%.14f' %strength][level]
                    elif strength < 1.e-1:  dist = json_dict['%.13f' %strength][level]
                else:
                    dist = json_dict[repr(strength)][level]
                dict2[error] = dist

                if error != 'AD' and error!= 'PolXY' and error != 'RZC' and error!= 'RHC':
                    if main_error == 'AD':
                        dist_error = dict2['AD']
                    elif main_error == 'PolXY':
                        dist_error = dict2['PolXY']
                    elif main_error == 'RZC':
                        dist_error = dict2['RZC']                   
                    elif main_error == 'RHC':
                        dist_error = dict2['RHC']                   

                    
                    dif = dict2[error] - dist_error
                    
                    if dif >= 0.-limit: 
                        pass #print 'honest'
                    else:  
                        #print 'Strength =',strength
                        #print 'Level =', level
                        #print 'Error =', error
                        #print 'not honest\n'
                        if '_cons' in error:
                            print 'Problems!!'
                            print error
                            print strength
                            print level
                            print dif   

            dict1[level] = dict2 
        output_dict[strength] = dict1               

    return output_dict  
    


def fit_logical_distances(input_dict, strengths, levels, channels,
              dict_of_fitting_funcs, main_channel='AD'):
    '''
    Notice:  This function works for any kind of distance, including
         both honesties and accuracies.
    input_dict:  a raw "honesty" dictionary.
                 See, for example, the raw files in summaries_pseudo1/distance/honesty/
    strengths:   a list of physical strengths we want to use as x values
             for our fit.
    levels:      list of levels. For example: ['p', 'lnc', 'l']
    channels:    list of channels.  For example: ['AD', 'PCapprox...', ...]
    dict_of...:  a dictionary of fitting functions. We do it this way
             because the functions to be used are different for some
             levels.
    '''
    
    dict2 = {}
    for level in levels:
        dict1 = {}
        for channel in channels:
            func = dict_of_fitting_funcs[level]
            if main_channel == 'AD' and channel == 'Dep_noise':
                new_strengths = [4*(m.sqrt(1-3*s) - (1 - 3*s))
                             for s in strengths]
            elif main_channel == 'PolXY' and channel == 'Dep_noise':
                new_strengths = [3.*s for s in strengths]
            else:
                new_strengths = strengths

            x_values = np.array(new_strengths)
            
            y_values = np.array([
                    input_dict[repr(strength)][level][channel] 
                    for strength in strengths
                    ])
            out = sc.optimize.curve_fit(func, x_values, y_values)

            coefficient = out[0][0]
            print type(coefficient)
            dict1[channel] = coefficient

        dict2[level] = dict1

    return dict2



def calculate_distance_final_states(folder, error, approxs, EC_type, 
                    error_kind, theta, phi, strengths, 
                    output, basis='', extra_folder=None,
                    theta_vector=0., phi_vector=0., 
                    angle=None, code='Steane', 
                    rot_errors=None):
    '''
    error = 'AD' or 'PolXY'
    approxs = ['CMCapproxAD_cons, ...]
    basis = '', 'majority_voting', or 'majority_voting_rotated_basis'
    '''
    if output == 'overlap':
        levels = ['p', 'l']
    elif output == 'distance':
        if EC_type == 'pseudo':
            levels = ['p', 'lnc', 'l']
        else:
            levels = ['p', 'lnc', 'l', 'lcp', 'lcf']

        # First we calculate the final physical state.
        phys_dens, log_dens = fun.initial_state_general_different_basis(
                theta_vector, phi_vector, theta, phi, code)

    #phys_dens, log_dens = fun.initial_state_general(theta, phi, code)

    phys_dict = {}
    lnc_dict = {}
    error_and_approxs = [error] + approxs
    for channel in error_and_approxs:
        dict1_phys = {}
        dict1_lnc = {}
        for strength in strengths:
                
            if channel == 'Dep_noise' and type(strength) != list:
                strens = map(str,[strength for i in range(3)])
                channel_and_strens = ['Pauli'] + strens 
                error_gate = ' '.join(channel_and_strens)
            else:
                error_gate = channel + ' ' + str(strength)
                if angle != None:   
                    error_gate += ''.join([' ', 
                            str(angle)])
            dict1_phys[strength] = apply_error(phys_dens, 
                                               error_gate, 
                                               'phys')
            dict1_lnc[strength] = apply_error(log_dens, 
                                              error_gate,
                                              'log', code,
                                              rot_errors)       
        phys_dict[channel] = dict1_phys
        lnc_dict[channel] = dict1_lnc
        

    
    output_dict = {}
    for strength in strengths:
        print 'Strength = ', strength
        dict1 = {}
        for level in levels:
            print 'Level =', level
    
            error_folder = '/'.join([folder, error, EC_type,
                         str(error_kind), 
                         basis])

            if error_folder[-1] != '/':
                error_folder += '/'
            if extra_folder != None:
                error_folder += '/'.join([extra_folder, ''])
        
            error_folder2 = ''.join(['theta', str(theta), '_',
                           'phi', str(phi)])
            
            error_folder += '/'.join([error_folder2, ''])

            
            if level == 'p':
                error_dens = phys_dict[error][strength]

            elif level == 'lnc':
                error_dens = lnc_dict[error][strength]          
            else:
                if level == 'l':
                    suffix = '.npy'
                elif level == 'lcp':
                    suffix = 'corrected_perf.npy'
                elif level == 'lcf':
                    suffix = 'corrected_fault.npy'

                error_filename = str(strength) + suffix
                error_path = error_folder + error_filename
                error_dens = np.load(error_path)
                trace = np.trace(error_dens).real
                error_dens = (1./trace)*error_dens

            dict2 = {}
            for approx in approxs:
                print 'Approx =', approx
                #if approx[:2] == 'PC' or approx[:3] == 'Dep':
                    #new_theta = theta_dict[theta]
                #   new_theta = theta   
                #else:
                #   new_theta = theta
        
                approx_folder = '/'.join([folder, approx, 
                              EC_type,
                              str(error_kind), 
                              basis])
            
                if approx_folder[-1] != '/':
                    approx_folder += '/'
                if extra_folder != None:
                    approx_folder += '/'.join([
                               extra_folder, 
                               ''])
            
                approx_folder += '/'.join([error_folder2, ''])

                if level == 'p':
                    approx_dens = phys_dict[approx][strength]
                
                elif level == 'lnc':
                    approx_dens = lnc_dict[approx][strength]

                else:
                    if level == 'l':  
                        suffix = '.npy'
                    elif level == 'lcp': 
                        suffix = 'corrected_perf.npy'
                    elif level == 'lcf':
                        suffix = 'corrected_fault.npy'

                    approx_filename = str(strength) + suffix
                    approx_path = approx_folder + approx_filename
                    approx_dens = np.load(approx_path)
                    trace = np.trace(approx_dens).real
                    approx_dens = (1./trace)*approx_dens

                if output == 'distance':
                    dist = fun.trace_distance(error_dens, 
                                  approx_dens)
                elif output == 'overlap':
                    pass
                
                dict2[approx] = dist
            
            dict1[level] = dict2    

        output_dict[strength] = dict1

    return output_dict



def calculate_any_average(folder, Bloch_points, theta_dict, phi_dict,
                  strength, level, channel, stand_dev=True,
                  output='distance', extra=None, p_viol=False,
                  limit=None):
    '''
    '''
    if folder[-1] != '/':   folder += '/'
    values = []
    values_sq = []
    n_points = 0
    viol = 0

    for theta in Bloch_points:
        if theta_dict == None:  new_theta = theta
        else:    new_theta = theta_dict[theta]
        phis = Bloch_points[theta]
        n_phis = len(phis)
        n_points += len(phis)
        
        for phi in phis:
            if phi_dict == None:  new_phi = phi
            else:  new_phi = phi_dict[phi]

            json_filename = ''.join(['theta', str(new_theta),
                         '_', 'phi', str(new_phi),
                         '_', output])
            if extra == None:
                json_filename += '.json'
            else:
                json_filename += ''.join(['_', extra, 
                              '.json'])

            json_path = folder + json_filename
            json_file = open(json_path, 'r')
            json_dict = json.load(json_file)
            json_file.close()
            
            if strength != None:
                st = repr(strength)
                value = float(json_dict[st][level][channel])
            else:
                value = float(json_dict[level][channel])
                 
            if p_viol and value < limit:   viol += 1
            
            values += [value]
            values_sq += [value**2]

            if value < -0.0001:
                print 'Theta = %s' %theta
                print 'Phi = %s' %phi
                print 'Value = %s \n' %value


    global_min, global_max = min(values), max(values)
    global_average = sum(values)/float(n_points)
    # The variance is always positive, but could be a very little
    # negative number, so let's compute its absolute value.
    global_variance = abs(sum(values_sq)/float(n_points) - global_average**2)
    if stand_dev:
        global_variance = sqrt(global_variance)

    output_dict = {'ave': global_average, 'var': global_variance, 
               'min': global_min, 'max': global_max}
    
    if p_viol:
        output_dict['p_viol'] = viol/float(n_points)

    return output_dict  



def threshold(x_data, y_data, fit_func, n, p_f=1.0, n_small=1, list_ns=None):
    '''
    Inputs: - x_data: needs to be a list
        - y_data: needs to be a list
        - fit_func: function to do the fit
        - n: order of polynomial + 1 (7+1=8 for Steane)
        - p_f: upper limit for noise strength
    This function takes a set of points corresponding to 
    (strength of noise, difference in physical and logical distance)
    and calculates the threshold, i.e. the point where the curve
    intersects the x axis (y = 0).
    '''
    popt, pcov = sc.optimize.curve_fit(fit_func, np.array(x_data), np.array(y_data))
    print popt
    y = sp.Symbol('y', positive=True)
    #print sum([popt[k-1]*y**(k+n_small-1) for k in range(1,n)])
    if list_ns == None:
        eqn = sp.Eq(sum([popt[k-1]*y**(k+n_small-1) for k in range(1,n)]), 0)
    else:
        eqn = sp.Eq(sum([popt[list_ns.index(k)]*y**(k) for k in list_ns]), 0)
    print 'eqn = ', eqn
    sol_list = sp.solve(eqn, y, check=False)
    valid_sols = []
    print 'Valid solutions =', sol_list
    for sol in sol_list:
        if abs(sp.im(sol)) < 1.e-16:
            #if (sol >= -1.e-3 and sol <= p_f):
            if (sp.re(sol) >= -1.e-3 and sp.re(sol) <= 1.0):
                valid_sols += [sp.re(sol)]
    print 'Valid solutions =', valid_sols
    
    valid_sols.sort()
    thresholds = []
    antithresholds = []

    # The threshold might be 0.
    # We're assuming the first element in y_data is 0.
    # If the second is positive, then 0 is a threshold.
    # If the second is negative, then 0 is just a normal point.
    
    ############# Plotting ###############################
    #def solved_dif(x):
    #   return sum([popt[k-1]*x**k for k in range(1,n)])

    #x_points_func = np.linspace(0,p_f,100)
    #y_points_func = [solved_dif(x) for x in x_points_func]
    #fig, ax = plt.subplots()
    #ax.plot(np.array(x_points_func), np.array([0. for x in x_points_func]), 'k--')
    #ax.plot(np.array(x_points_func), np.array(y_points_func), 'k--')
    #ax.plot(np.array(x_data), np.array(y_data), 'ro')
    #ax.set_xlim((-0.05, 0.666))
    #ax.set_ylim((-0.02,0.1))
    #plt.show()
    ############# Plotting ###############################


    print 'x data =', x_data
    print 'y data =', y_data
    
    print 'valid solutions =', valid_sols

    #if y_data[1] > 0:
    #   thresholds += [x_data[0]]
   
    print 'valid sols =', valid_sols
 
    if abs(valid_sols[0]) < 1.e-9:
        valid_sols = valid_sols[1:] # because the first is always 0

    print 'valid solutions =', valid_sols

    print 'thresholds =', thresholds

    for sol in map(abs, valid_sols):
        print 'solution =', sol
        print 'thresholds =', thresholds
        for x in x_data:
            print 'x =', x
            if x >= sol:
                i = x_data.index(x)
                if y_data[i] > 0:
                    thresholds += [sol]
                    print 'Found it! Threshold'
                else:
                    antithresholds += [sol]
                    print 'Found it! Antithreshold'
                break

    # CORRECT THIS LATER ON!!!!!!!!

    print 'thresholds =', thresholds
    
    for sol in map(abs, valid_sols):
        #if (sol >= 0.5 and sol <= p_f):
        if (sol >= x_data[-1] and sol <= p_f):
            thresholds += [sol]

    # Hack added to avoid returning an empty list when the logical
    # distance is always lower than the physical distance

    if len(thresholds) == 0:
        thresholds += [p_f]

    
    print 'thresholds =', thresholds
    print 'antithresholds =', antithresholds

    # Check this break.  It should go out of only one for.

    #if len(valid_sols) > 1:
    #   valid_sols.sort()
    #   valid_sols = valid_sols[1:]

    return thresholds, antithresholds
    #return valid_sols, antithresholds


def threshold_from_distance_file(absolute_filename, func, n, p_f=1.0,
                 want_plot=False, output='overlap',
                 strengths=None, kind='old', x_min=1.e-5, 
                 x_max=5.e-3, log=True, n_small=1, list_ns=None):
    '''
    Reads distances from json file and computes threshold based
    on polynomial fitting.
    Assumes the distance file is a json file.
    '''
    strengths, phys_dists, log_dists = fun.read_distances_from_json(
                        absolute_filename,
                        output, strengths, 
                        kind)

    print 'strengths', strengths
    print 'phys dists', phys_dists
    print 'log dists', log_dists

    if want_plot:
        fig = plt.figure(figsize=(7,5), dpi=200)
        ax = plt.subplot(1,1,1)     
        ax.plot(np.array(strengths), np.array(phys_dists),
             color='blue', linewidth=2.5, linestyle='-',
             label='physical')
        ax.plot(np.array(strengths), np.array(log_dists),
             color='green', linewidth=2.5, linestyle='-',
             label='logical')
        plt.legend(loc='upper left')
        if log:
            ax.set_xscale('log')
            ax.set_yscale('log')
        plt.xlim(x_min, x_max)
        figure_name = absolute_filename[:-4] + 'png'
        plt.savefig(figure_name, dpi=200)
        plt.show()
        

    diffs = [y-x for x,y in zip(phys_dists, log_dists)]
    thresh, antithresh = threshold(strengths, diffs, func, n, p_f, n_small, list_ns)

    return thresh, antithresh





def threshold_for_one_point(theta, phase, n_gammas, error_gate, phi, 
                func, n, p_f, output, code, error_kind, 
                EC_type):
    '''
    phi is just used if the gate is PolXY.  For ADC it is 'None'.
    '''
    list_gammas = [p_f*float(i)/n_gammas for i in range(n_gammas+1)]    
    list_error_gates = generate_list_error_gates(error_gate, list_gammas, phi)

    phys_dens, log_dens = initial_state_general(theta, phase, code)
    phys_dist,log_dist = dist(phys_dens, log_dens, list_gammas, list_error_gates,
                  output, code, error_kind, EC_type)
    diffs = [y-x for x,y in zip(phys,log)]
    thresh, antithresh = threshold(list_gammas, diffs, func, n, p_f)
     
    return thresh, antithresh



def threshold_for_several_points(n_greenwich, n_gammas, error_gate, phi,
                func, n, p_f, output, code, error_kind,
                EC_type):
    '''
    '''
    Bloch_sphere_points = fun.points_on_Bloch_sphere(n_greenwich)
    threshold_dict = {}
    
    for theta in Bloch_sphere_points:
        phase_dict = {}
        
        for phase in Bloch_sphere_points[theta]:
            threshold = threshold_for_one_point(theta, phase, n_gammas,
                    error_gate, phi, func, n, p_f, output, code, 
                    error_kind, EC_type)
            phase_dict[phase] = threshold
        
        threshold_dict[theta] = phase_dict

    return threshold_dict



#threshold_dict = threshold_for_several_points(3, 10, 'AD', None, dif, 8, 0.5,
#                         'distance', 'Steane', 1, 'pseudo')
#threshold_json = json.dumps(threshold_dict, sort_keys=True, indent=4, 
#               separators=(',', ': '))
#f = open('thresholds_ADC.json', 'w')
#f.write(threshold_json)
#f.close()



#########################################################################################
#                                           #
#    All the fllowing functions are currently not being used.  I decided to put them    #
#    all at the end cause it was getting too confusing.  MGA 12/10/13.          #
#                                           #
#########################################################################################











def create_logical_state_Shor(state='zero'):
    zeros = sim.create_ket(['zero','zero','zero'])
    ones = sim.create_ket(['one','one','one'])
    plus = (1./sqrt(2))*(zeros + ones)
    minus = (1./sqrt(2))*(zeros - ones)
    if state == 'zero':
        return sim.tensor_product([plus,plus,plus])
    elif state == 'one':
        return sim.tensor_product([minus,minus,minus])
    else:
        raise NameError('Only zero or one')




def create_initial_state_Steane(initial_state='zero'):
    """
    Creates a circuit that generates a logical zero, one, plus, or minus
    for the Steane code.
    """
    if (initial_state == 'zero' or initial_state == 'one'):
        circ = cor.Steane_Correct.encoded_zero_Steane()
        if initial_state == 'one':
            for j in range(3):
                circ.add_gate_at([j], 'X')
        
    elif (initial_state == 'plus' or initial_state == 'minus'):
        circ = cor.Steane_Correct.encoded_plus_Steane()
        if initial_state == 'minus':
            for j in range(3):
                circ.add_gate_at([j], 'Z')
    
    return circ
    








def create_initial_state_plus_error(initial_state_or_circ, gate):
    """
    """
    if type(initial_state_or_circ) == str:
        circ = create_initial_state_Steane(initial_state_or_circ)
    else:
        circ = initial_state_or_circ

    for i in range(7):
        circ.add_gate_at([i], gate)
    
    return circ





def create_circuit_with_EC_Steane(initial_state_or_circ, gate, qubits='All', FT=False):
    """
    It can take either an initial state or an already defined circuit
    and adds 7 gates (one to each qubit) and an error correction step.
    If the FT option is False, then there's only 1 ancilla.
    If it's True, there are 4 ancillae (cat state).
    """
    if type(initial_state_or_circ) == str:
        circ = create_initial_state_Steane(initial_state_or_circ)
    else:
        circ = initial_state_or_circ
    
    circ_error = error(gate, 7, qubits)
    circ.join_circuit(circ_error, False)    

    if FT:  
        circ2 = cor.Cat_Correct.cat_syndrome_4(st.Code.stabilizer, 1, False, False).unpack()
    else:
        circ2 = create_nonFT_EC_for_Steane()
    circ.join_circuit(circ2, False) 

    return circ





def final_dens_matr_and_prob_one_meas_outcome(initial_state='zero', gate='AD 0.1', qubits='All', desired_outcomes=[[0] for i in range(6)], code='Steane'):
    """
    Creates a circuit with the preparation of the initial state, an error gate, and stabilizer 
    measurement with the desired outcomes specified.
    Currently only numerical.  The symbolic option is not implemented yet.
    It returns the final density matrix and the total probability of this occuring.  
    """
    #circ0 = create_initial_state_Steane(initial_state)
    #circ0_st = sim.State_and_Operations(circ0, 7)
    #circ0_st.apply_all_operations()
    #initial_dens_matrix = circ0_st.current_state.density_matrix
    
    if code == 'Steane':
        circ = create_circuit_with_EC_Steane(initial_state, gate, qubits)
        circ_sim = sim.Whole_Circuit(circ, desired_outcomes)

    elif code == 'Shor':
        circ = error(gate, 9)
        circ2 = create_nonFT_EC(9, Shor_stabilizers)
        circ.join_circuit(circ2, False)
        circ_sim = sim.Whole_Circuit(circ, desired_outcomes, False, initial_state)
        
    output = circ_sim.run_all_subcircuits()
    final_dens_matrix = output[0]
    #print desired_outcomes
    correction = get_syndrome(desired_outcomes)
    corrected_final_dens_matrix = correction*final_dens_matrix*(correction.H)
    
    total_prob = 1
    for prob in output[2]:
        total_prob *= prob
    
    #return [str(Decimal(trace_distance)), str(Decimal(total_prob))]
    #return [str(trace_distance), str(total_prob)]

    return corrected_final_dens_matrix, total_prob


#res = final_dens_matr_and_prob_one_meas_outcome(initial_state='one')
#print np.trace(res[0])
#print res[1]

def compute_final_dens_matr(initial_state='zero', gate='AD 0.1', n_qubits=7, qubits='All', code='Steane'):
    """
    Calculates the final density matrix takes into account all the possible syndrome outcomes, 
    which are 2**6 for the Steane code.
    """
    total_prob = 0
    final_dens_matr = np.matrix([[(0. + 0.j) for i in range(2**n_qubits)] for j in range(2**n_qubits)])
    for i in range(2**(n_qubits-1)):
        dens, prob = final_dens_matr_and_prob_one_meas_outcome(initial_state, gate, qubits, sim.binary(i, n_qubits-1), code)
        #final_dens_matr += prob*dens
        final_dens_matr += dens
        total_prob += prob

    return final_dens_matr, prob





#zero_ket = sim.create_logical_state_Steane('zero', True)
#one_ket = sim.create_logical_state_Steane('one', True)
#theta = sp.Symbol('theta', real=True)
#phi = sp.Symbol('phi', real=True)
#enc_ket = sp.cos(theta/2)*zero_ket + sp.exp(sp.I*phi)*sp.sin(theta/2)*one_ket
#enc_dens = enc_ket*(quant.Dagger(enc_ket))

#ket = mat.Matrix([[sp.cos(theta/2)],[sp.exp(sp.I*phi)*sp.sin(theta/2)]])
#dens = ket*(quant.Dagger(ket))

def symbolic_distance(dens, enc_dens, error_gate):
    """
    error needs to be a string like 'AD g'.
    Currently only implemented for Steane
    """
    t1 = t.clock()
    print 'Gate =', error_gate
    circ = c.Circuit()
    circ.add_gate_at([0], error_gate)
    circ_sim = sim.State_and_Operations(circ, 1, dens, [None], True)
    circ_sim.apply_all_operations()
    final_state1 = circ_sim.current_state.density_matrix
    overlap1 = sp.simplify(sim.overlap(dens, final_state1, True)) 
    print 'physical overlap =', overlap1

    circ1 = error(error_gate)
    circ2 = create_nonFT_EC_for_Steane()
    circ1.join_circuit(circ2, False)
    circ_sim1 = sim.Whole_Circuit(circ1, [['All'] for i in range(6)], True, enc_dens)
    states = circ_sim1.run_all_in_tree()
    corrected_states = apply_correction_to_every_state(states, True)
    final_state2 = sim.sum_matrices(corrected_states, True)
    overlap2 = sp.simplify(sim.overlap(enc_dens, final_state2, True))
    difference = sp.simplify(overlap1-overlap2)
    print 'logical overlap =', overlap2
    print 'difference =', difference    

    filename1 = 'Final_density_matrix_%s.txt' %error_gate
    filename2 = 'difference_%s.txt' %error_gate

    string = ''
    for i in range(len(final_state2)):
        string += str(final_state2[i]) + '\n'
    f = open(filename1, 'w')
    f.write(string)
    f.close()
    f = open(filename2, 'w')
    f.write((str(difference)).replace("**", "^"))
    f.close()

    t2 = t.clock()
    print 'This took me %f s' %(t2-t1)

    return overlap1, overlap2, difference


#overlap1, overlap2, difference = symbolic_distance(dens, enc_dens, 'AD g')
#overlap1, overlap2, difference = symbolic_distance(dens, enc_dens, 'CMCapproxAD_cons g')

#g = sp.Symbol('g', real=True)
#theta = sp.Symbol('theta', real=True)
#phi = sp.Symbol('phi', real=True)
#physical = -g*sin(theta/2)**4 - g*cos(theta/2)**4 + g*cos(theta/2)**2 - 2*(-g + 1)**(1/2)*cos(theta/2)**4 + 2*(-g + 1)**(1/2)*cos(theta/2)**2 + sin(theta/2)**4 + cos(theta/2)**4

#from sympy import exp

#logical = (6.0*g**7*exp(2*I*phi)*sin(theta/2)**4 + 6.0*g**7*exp(2*I*phi)*cos(theta/2)**4 - 6.0*g**7*exp(2*I*phi)*cos(theta/2)**2 - 21.0*g**6*exp(2*I*phi)*sin(theta/2)**4 - 21.0*g**6*exp(2*I*phi)*cos(theta/2)**4 + 21.0*g**6*exp(2*I*phi)*cos(theta/2)**2 + 1.3125*g**5*(-g + 1)**(1/2)*exp(4*I*phi)*cos(theta/2)**4 - 1.3125*g**5*(-g + 1)**(1/2)*exp(4*I*phi)*cos(theta/2)**2 - 2.625*g**5*(-g + 1)**(1/2)*exp(2*I*phi)*cos(theta/2)**4 + 2.625*g**5*(-g + 1)**(1/2)*exp(2*I*phi)*cos(theta/2)**2 + 1.3125*g**5*(-g + 1)**(1/2)*cos(theta/2)**4 - 1.3125*g**5*(-g + 1)**(1/2)*cos(theta/2)**2 + 31.5*g**5*exp(2*I*phi)*sin(theta/2)**4 + 31.5*g**5*exp(2*I*phi)*cos(theta/2)**4 - 31.5*g**5*exp(2*I*phi)*cos(theta/2)**2 - 3.9375*g**4*(-g + 1)**(1/2)*exp(4*I*phi)*cos(theta/2)**4 + 3.9375*g**4*(-g + 1)**(1/2)*exp(4*I*phi)*cos(theta/2)**2 + 7.875*g**4*(-g + 1)**(1/2)*exp(2*I*phi)*cos(theta/2)**4 - 7.875*g**4*(-g + 1)**(1/2)*exp(2*I*phi)*cos(theta/2)**2 - 3.9375*g**4*(-g + 1)**(1/2)*cos(theta/2)**4 + 3.9375*g**4*(-g + 1)**(1/2)*cos(theta/2)**2 - 26.25*g**4*exp(2*I*phi)*sin(theta/2)**4 - 36.75*g**4*exp(2*I*phi)*cos(theta/2)**4 + 31.5*g**4*exp(2*I*phi)*cos(theta/2)**2 + 5.25*g**3*(-g + 1)**(1/2)*exp(4*I*phi)*cos(theta/2)**4 - 5.25*g**3*(-g + 1)**(1/2)*exp(4*I*phi)*cos(theta/2)**2 - 12.0*g**3*(-g + 1)**(1/2)*exp(2*I*phi)*cos(theta/2)**4 + 12.0*g**3*(-g + 1)**(1/2)*exp(2*I*phi)*cos(theta/2)**2 + 5.25*g**3*(-g + 1)**(1/2)*cos(theta/2)**4 - 5.25*g**3*(-g + 1)**(1/2)*cos(theta/2)**2 + 14.0*g**3*exp(2*I*phi)*sin(theta/2)**4 + 35.0*g**3*exp(2*I*phi)*cos(theta/2)**4 - 24.5*g**3*exp(2*I*phi)*cos(theta/2)**2 - 2.625*g**2*(-g + 1)**(1/2)*exp(4*I*phi)*cos(theta/2)**4 + 2.625*g**2*(-g + 1)**(1/2)*exp(4*I*phi)*cos(theta/2)**2 + 9.75*g**2*(-g + 1)**(1/2)*exp(2*I*phi)*cos(theta/2)**4 - 9.75*g**2*(-g + 1)**(1/2)*exp(2*I*phi)*cos(theta/2)**2 - 2.625*g**2*(-g + 1)**(1/2)*cos(theta/2)**4 + 2.625*g**2*(-g + 1)**(1/2)*cos(theta/2)**2 - 5.25*g**2*exp(2*I*phi)*sin(theta/2)**4 - 15.75*g**2*exp(2*I*phi)*cos(theta/2)**4 + 10.5*g**2*exp(2*I*phi)*cos(theta/2)**2 - 1.0*g*(-g + 1)**(1/2)*exp(2*I*phi)*cos(theta/2)**4 + 1.0*g*(-g + 1)**(1/2)*exp(2*I*phi)*cos(theta/2)**2 - 2.0*(-g + 1)**(1/2)*exp(2*I*phi)*cos(theta/2)**4 + 2.0*(-g + 1)**(1/2)*exp(2*I*phi)*cos(theta/2)**2 + 1.0*exp(2*I*phi)*sin(theta/2)**4 + 1.0*exp(2*I*phi)*cos(theta/2)**4)*exp(-2*I*phi)

#print logical.subs(phi,0)
#print logical.subs(phi,pi/2)
#print logical.subs(phi,pi)


    


 


def basis_vectors_XY(phi):
    '''
    Takes in an angle phi, which defines a vector in the XY plane.
    The angle is defined with respect to the X axis, so that:
    -phi = 0 is +X
    -phi = pi/2 is +Y
    and so on.
    The output is the +/- eigenvectors with this angle, both physical
    and logical.
    Currently only for the Steane code.  
    '''
    phase = complex(np.cos(phi),np.sin(phi))
    phys_a = sqrt(0.5)*(zero + phase*one)
    phys_b = sqrt(0.5)*(zero - phase*one)
    log_0 = sim.create_logical_state_Steane('zero')
    log_1 = sim.create_logical_state_Steane('one')
    log_a = sqrt(0.5)*(log_0 + phase*log_1)
    log_b = sqrt(0.5)*(log_0 - phase*log_1)
    
    return [phys_a, log_a], [phys_b, log_b] 


def initial_state_XY(theta, phi):
    '''
    '''
    zeros, ones = basis_vectors_XY(phi)
    phys_ket = np.cos(theta/2)*zeros[0] + np.sin(theta/2)*ones[0]
    log_ket = np.cos(theta/2)*zeros[1] + np.sin(theta/2)*ones[1]
    phys_dens = phys_ket*(phys_ket.H)
    log_dens = log_ket*(log_ket.H)

    return phys_dens, log_dens



def thresholds(n_theta, list_gammas, phi, list_error_gates, func, n, p_f=1.0):
    '''
    zeros: list of [physical zero, logical zero]
    ones: list of [physical one, logical one]
    '''
    list_solutions = []
    for i in range(n_theta+1):
        pre_theta = float(i)/float(n_theta) 
        theta = pre_theta*np.pi
        phys_dens, log_dens = initial_state_XY(theta, phi)  
    
        print 'Point %i of %i' %(i+1, n_theta+1)
        print 'Theta/pi =', pre_theta
    
        phys,log = dist(phys_dens, log_dens, list_gammas, list_error_gates)
        diffs = [y-x for x,y in zip(phys,log)]
    
        print 'list gammas =', list_gammas
        print 'diffs =', diffs
    
        thresh, antithresh = threshold(list_gammas, diffs, func, n, p_f)
        list_solutions += [[pre_theta, theta, thresh, antithresh]]
        
    return list_solutions


def distance_plot_XY(theta, phi, n_gammas, upper_gamma, approx):
    '''
    ''' 
    phys_dens, log_dens = initial_state_XY(theta, phi)
    list_gammas = [upper_gamma*i/n_gammas for i in range(n_gammas+1)]
    list_error_gates = generate_list_error_gates(approx, list_gammas, phi)      
    phys_dist, log_dist = dist(phys_dens, log_dens, list_gammas, list_error_gates) 
    
    outlines = [' '.join(map(str,[list_gammas[i],phys_dist[i],log_dist[i]])) for i in range(n_gammas+1)]
    out_string = '\n'.join(outlines)
    file_name = '_'.join(['Dist',approx,str(round(theta,2)),str(round(phi,2))]) + '.txt'
    out_file = open(file_name, 'w')
    out_file.write(out_string)
    out_file.close()

    return None


def distance_plot_ADC(theta, phis, n_gammas, upper_gamma, approx):
    '''
    '''
    list_gammas = [upper_gamma*i/n_gammas for i in range(n_gammas+1)]
    list_error_gates = generate_list_error_gates(approx, list_gammas)       
    out_info = [map(str,list_gammas)]
    for phi in phis:
        print 'phi =', phi
        phys_dens, log_dens = initial_state_general(theta, phi)
        phys_dist,log_dist = dist(phys_dens,log_dens,list_gammas,list_error_gates)
        out_info += [log_dist]
    out_lines = []
    for i in range(n_gammas):
        out_lines += [' '.join([str(l[i]) for l in out_info])]
    out_string = '\n'.join(out_lines)
    file_name = '_'.join(['Dist',approx,str(round(theta,2)),'test']) + '.txt'
    out_file = open(file_name, 'w')
    out_file.write(out_string)
    out_file.close()

    return None     



def thresholds_AD(n_theta, n_gamma, phi, func, n, approximation, p_f=1.0):
    '''
    '''
    t1 = t.clock()
    log_0 = sim.create_logical_state_Steane('zero')
    log_1 = sim.create_logical_state_Steane('one')
    zeros, ones = [zero, log_0], [one, log_1]
    
    list_gammas = [0.5*(float(i)/n_gamma) for i in range(n_gamma+1)]

    list_error_gates = generate_list_error_gates(approx, list_gammas)

    sols = thresholds(n_theta, list_gammas, phi, list_error_gates, zeros, ones, func, n, p_f)
    #ser_sols = []
    #for sol in sols:
    #   ser_sols += [[sol[0], sol[1], map(str(sol[2]))]]
    output_string = '\n'.join(' '.join(map(str,sol)) for sol in sols)
    out_file = open(approximation + '.txt', 'w')
    out_file.write(output_string)
    out_file.close()
    t2 = t.clock()
    print 'This took me %f s' %(t2-t1)

    return None 



def thresholds_PolXY(n_theta, n_p, phi, func, n, approx, p_f=(2./3.)):
    '''
    '''
    t1 = t.clock()
    zeros, ones = basis_vectors_XY(phi)

    list_ps = [(2./3.)*(float(i)/n_p) for i in range(n_p+1)]
    
    list_error_gates = []
    ser_ps = map(str,list_ps)
    ser_phis = [str(phi) for i in range(n_p+1)]

    list_error_gates = [' '.join([approx, ser_ps[i], ser_phis[i]]) for i in range(n_p+1)] 

    sols = thresholds(n_theta, list_ps, phi, list_error_gates, zeros, ones, func, n, p_f)
    ser_sols = []
    for sol in sols:
        one_sol = [sol[0], sol[1]] + sol[2] + sol[3]
        ser_sols += [one_sol]
    output_string = '\n'.join([' '.join(map(str,sol)) for sol in ser_sols])
    out_file = open(approx + str(round(phi,3)) + '.txt', 'w')
    out_file.write(output_string)
    out_file.close()
    t2 = t.clock()
    print 'This took me %f s' %(t2-t1)

    return None
    

#approximations = ['AD', 'CMCapproxAD_cons', 'PCapproxAD_cons']
#n_theta = 5
#n_pi2 = 30
#n_gammas = 10
#upper_gamma = 0.5

#thetas = [i*np.pi/n_theta for i in range(n_theta+1)]
#for approx in approximations:
#   for theta in thetas:
#       print 'theta =', theta
#       n = int(float(n_pi2*np.sin(theta)))
#       phis = [i*2*np.pi/n for i in range(n)]
#       distance_plot_ADC(theta, phis, n_gammas, upper_gamma, approx)       
        
            





#n_theta = 31 
#n_gamma = 50    # This one needs to be a factor of 200 because that's how many points we have in Pauli_coefficients.txt.

#approx = 'PCapproxAD_cons'
#thresholds_AD(n_theta, n_gamma, dif, 8, approx)

#approximations = ['PolXY', 'CMCapproxPolXY_cons', 'PCapproxPolXY_cons']
#phis = [pi*i/32 for i in range(17)]
#thetas = [pi*i/4 for i in range(5)]

#for phi in phis:
#   print 'phi =', phi
#   for theta in thetas:
#       print 'theta =', theta
#       for approx in approximations:
#           print 'approximation =', approx
#           distance_plot_XY(theta, phi, n_gamma, 0.5, approx)
#for phi in phis:
#   print 'phi =', phi
#   for approx in approximations:
#       print 'approx =', approx
#       thresholds_PolXY(n_theta, n_gamma, phi, dif, 8, approx)
