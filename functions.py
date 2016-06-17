import sys
import os
sys.path += ['/home/mau/MUSIQC/new_musiqc']
from circuit import *
from math import sqrt, sin, cos, pi, exp, acos
import numpy as np
import scipy as sc
import sympy as sp
import sympy.matrices as mat
import sympy.functions as sympy_func
#import sympy.physics.quantum as quant
import random as rd
import collections as col
import Approx_Errors as ap
#import visualizer.visualizer as vis
import time as t
import json
import copy
import faultTolerant.steane as st
import faultTolerant.fivequbit as fivequbit

zero = np.matrix([[1.],[0.]])
one = np.matrix([[0.],[1.]])


# Functions for fitting the honesties and accuracies
# in the limit of small noise strength (p -> 0).

def linear(x,a):   return a*x
def quadratic(x,a):   return a*x**2
def tert(x,a):   return a*x**3
def quartic(x,a):   return a*x**4
def linear_and_quadratic(x,a,b):  return a*x + b*x**2
def quadratic_and_tert(x,b,c):    return b*x**2 + c*x**3
def tert_and_quartic(x,c,d):      return c*x**3 + d*x**4
def quartic_and_quintic(x,d,e):   return d*x**4 + e*x**5

def three_halves(x,a):  return a*x**(1.5)


# The next functions will be used by Conor to fit the entries
# in the process matrices.
def hd1(x,a,b,c):   return a*x + b*x**2 + c*x**3
def hd2(x,a,b,c):   return a*x**2 + b*x**3 + c*x**4
def hd3(x,a,b,c):   return a*x**3 + b*x**4 + c*x**5
def hd4(x,a,b,c):   return a*x**4 + b*x**5 + c*x**6



def px(g):
    """
    Polynomial approximation to the px coefficient of the PC
    approximation to the ADC
    """
    return -0.00216689*g**4 - 0.000863701*g**3 - 0.00335448*g**2 + 0.523581*g



def pz(g):
    """
    Polynomial approximation to the pz coefficient of the PC
    approximation to the ADC
    """
    return 0.0344617*g**4 + 0.00592929*g**3 + 0.0330055*g**2 + 0.14574*g



def binary(num, length_list=6):
    """Binary representation of a number"""
    binary = []
    for i in range(length_list):
        binary.insert(0, [num&1])
        num = num >> 1  
    return binary


def bit_flip_code_interpreter(number=0):
    """Interpretation of the bit flip code"""
    if number == 0:    return 'n'
    elif number == 1:  return 2
    elif number == 2:  return 0
    elif number == 3:  return 1
    else: raise ValueError('Invalid number')


"""Dictionaries"""
gate_matrix_dic = {'I': ap.I,
           'X': ap.X,
           'Y': ap.Y,
           'Z': ap.Z,
           'H': ap.H,
           'S': ap.S,
           'Sinv': ap.Sinv,          
           'MeasureZ': [np.matrix([[1.,0.],[0.,0.]]), 
                np.matrix([[0.,0.],[0.,1.]])],
           'MeasureX': [0.5*np.matrix([[1.,1.],[1.,1.]]), 
                0.5*np.matrix([[1.,-1.],[-1.,1.]])]}


gate_matrix_dic_sym = {'I': sp.Matrix([[1.,0.],[0.,1.]]),
               'X': sp.Matrix([[0.,1.],[1.,0.]]),
               'Y': sp.Matrix([[0.,-sp.I],[sp.I,0.]]),
               'Z': sp.Matrix([[1.,0.],[0.,-1.]]),
               'H': (1./sp.sqrt(2))*sp.Matrix([[1.,1.],[1.,-1.]]),
               'S': sp.Matrix([[1.,0.],[0.,-sp.I]]),           
               'MeasureZ': [sp.Matrix([[1.,0.],[0.,0.]]), 
                    sp.Matrix([[0.,0.],[0.,1.]])],
               'MeasureX': [0.5*sp.Matrix([[1.,1.],[1.,1.]]), 
                    0.5*sp.Matrix([[1.,-1.],[-1.,1.]])]}


symbol_word_dic = {'0': 'zero',
           '1': 'one',
           '+': 'plus',
           '-': 'minus',}



name_gate_dic = {'zero': 'PrepareZPlus',
         'one': 'PrepareZMinus',
         'plus': 'PrepareXPlus',
         'minus': 'PrepareXMinus'}



state_vector_dic = {'PrepareZPlus': np.matrix([[1.],[0.]]),
            'PrepareZMinus': np.matrix([[0.],[1.]]),
            'PrepareXPlus': (1./sqrt(2.))*np.matrix([[1.],[1.]]),
                'PrepareXMinus': (1./sqrt(2.))*np.matrix([[1.],[-1.]]),
            'PrepareYPlus':  (1./sqrt(2.))*np.matrix([[1.],[1.j]]),
            'PrepareYMinus':  (1./sqrt(2.))*np.matrix([[1.],[-1.j]])}



state_vector_dic_sym = {'PrepareZPlus': sp.Matrix([[1.],[0.]]),
                'PrepareZMinus': sp.Matrix([[0.],[1.]]),
                'PrepareXPlus': (1./sp.sqrt(2))*sp.Matrix([[1.],[1.]]),
                    'PrepareXMinus': (1./sp.sqrt(2))*sp.Matrix([[1.],[-1.]]),
                'PrepareYPlus':  (1./sp.sqrt(2))*sp.Matrix([[1.],[sp.I]]),
                'PrepareYMinus':  (1./sp.sqrt(2))*sp.Matrix([[1.],[-sp.I]])}



theta_dict_4 = {  0.:      0.,
          pi/4:    pi/4,
          pi/2:    pi/2,
          3*pi/4:  pi/4,
          pi:      0.
           }

phi_dict_4 = {  0.:      0.,
            2*pi/5: 2*pi/5,
        4*pi/5: 4*pi/5, 
        6*pi/5: 4*pi/5,
            8*pi/5: 2*pi/5,
        pi/4:   pi/4,
        pi/2:   pi/2,
        3*pi/4: pi/4,
        pi:     0.,
        5*pi/4: pi/4,
        3*pi/2: pi/2,
        7*pi/4: pi/4
         }


Bloch_points_ADC = { 0.: [0.],
             pi/4: [0., 2*pi/5, 4*pi/5],
             pi/2: [0., pi/4, pi/2],
             3*pi/4: [0., 2*pi/5, 4*pi/5],
             pi: [0.]
           }


Bloch_points_PolXY = {  0.: [0.],
            pi/4: [0., 2*pi/5, 4*pi/5],
            pi/2: [0., pi/4, pi/2]
             }
    

Minimization_groups = {0:  [[0,2,12,14], [], 'Pauli'],
               1:  [[0,2,12,14] + range(24,30), [], 'Pauli + m'],
               2:  [range(24), [], 'Clifford'],
               3:  [range(30), [], 'complete group'],
               4:  [[0,24], [], 'I + measZp'],
               5:  [[0,12,15], [], 'I + X + HXY'],
               6:  [[0,2,12,15], [], 'I + Z + X + HXY'],
               7:  [[0,2,12,14], [[12,14]], 'Pauli constrained px py'],
               8:  [[0,2,12,14], [[2,12], [12,14]], 'Depolarizing channel'],
               9:  [[0,1], [], 'I + S'],
              10:  [[0,4], [], 'H'],
	      11:  [[0,1,7,20,23], [], 'I + S + HSt + StH + StHSt'],
              12:  [[0, 1, 23], [], 'I + S + StHSt']
	      }


"""General functions"""
def points_on_Bloch_sphere(n_greenwich, thetas=None, output_dens_matrix=False):
    '''
    Generates uniformly distributed points on the Bloch sphere surface.
    n_greenwich is the number of points on the greenwich meridian, i.e.
    the half circle between |0> (north pole) and |1> (south pole) that
    crosses |+>.
    the total number of points is given by 4*n_greenwich**2/pi  
    Although set by default to None, the values of theta can also be 
    specified explicitly
    '''
    # thetas are the angles from the north pole to the south pole.
    if thetas == None:
        thetas = [i*pi/n_greenwich for i in range(n_greenwich + 1)]

    output = {}
    for theta in thetas:
        nphis = int(2*sin(theta)*n_greenwich)
        if nphis == 0: nphis = 1
        phis = [2*i*pi/nphis for i in range(nphis)]
        
        if output_dens_matrix:
            phis_dic = {}
        
            for phi in phis:
                phase = complex(cos(phi), sin(phi))
                ket = cos(theta/2)*zero + phase*sin(theta/2)*one
                dens_matrix = np.outer(ket,ket.H)
                phis_dic[phi] = dens_matrix
        
            output[theta] = phis_dic

        else:
            output[theta] = phis

    return output




def points_on_Bloch_sphere_random(n):
    '''
    Generates random points on the Bloch sphere surface
    Based on mathworld.wolfram.com/SpherePointPicking.html
    '''
    output = []
    for i in range(n):
        phi = 2*pi*rd.random()
        theta = acos(2*rd.random() - 1)
        phase = complex(cos(phi), sin(phi))
        ket = cos(theta/2)*zero  + phase*sin(theta/2)*one
        dens_matrix = np.outer(ket,ket.H)
        output += [[theta, phi, dens_matrix]]
        
    return output



def test_Bloch_sphere(n_greenwich, random=False):
    '''
    Test to determine which method to generate points on the 
    Bloch sphere surface is better.  
    The non-random seems to be better in the sense that for 
    a given number of points, the resulting average density 
    matrix is closer to the Identity (maximally mixed state). 
    '''
    if random:  
        angles = points_on_Bloch_sphere_random(int(4*n_greenwich**2/pi))
    else:
        angles = points_on_Bloch_sphere(n_greenwich)

    average_dens = np.matrix([[0.j,0.j],[0.j,0.j]])
    for angle in angles:
        average_dens += angle[2]
    average_dens /= len(angles)

    return [average_dens]



'''
Functions taken from Approx_Errors.py
'''

"""Basic definitions"""
x = sp.Symbol('x[0]', real=True)
y = sp.Symbol('x[1]', real=True)
z = sp.Symbol('x[2]', real=True)
gamma = sp.Symbol('gamma', positive=True)
pm = 0.5*(1+gamma-sp.sqrt(1-gamma))
p = sp.Symbol('p', positive=True)
phi = sp.Symbol('phi', positive=True)
p1 = (p/7.)*(3. + 4*sp.cos(2*phi) - 3*sp.sin(2*phi))
p2 = (p/7.)*(3. - 3*sp.cos(2*phi) + 4*sp.sin(2*phi))

paX, paY = sp.Symbol('paX', positive=True), sp.Symbol('paY', positive=True)
paZ, paX, paH = sp.Symbol('paZ', positive=True), sp.Symbol('paX', positive=True), sp.Symbol('paH', positive=True)

I = np.matrix([[1.,0.],[0.,1.]])
H = (1./sqrt(2.))*np.matrix([[1.,1.],[1.,-1.]])
S = np.matrix([[1.,0.],[0.,1.j]])
Sinv = S.H
X = np.matrix([[0.,1.],[1.,0.]])
Y = np.matrix([[0.,-1.j],[1.j,0.]])
Z = S*S
HXY = sqrt(1./2.)*np.matrix([[0.,1.-1.j],[1.+1.j,0.]])

Isym = sp.Matrix([[1.,0.],[0.,1.]])
Xsym = sp.Matrix([[0.,1.],[1.,0.]])
Ysym = sp.Matrix([[0.,-1.j],[1.j,0.]])
Zsym = sp.Matrix([[1.,0.],[0.,-1.]])
HXYsym = sp.sqrt(1./2.)*sp.Matrix([[0.,1.-1.j],[1.+1.j,0.]])
rhosym = 0.5*(Isym + x*Xsym + y*Ysym + z*Zsym)
Pauli_channel = [sp.sqrt(1-paX-paY-paZ)*Isym, sp.sqrt(paX)*Xsym, sp.sqrt(paY)*Ysym, sp.sqrt(paZ)*Zsym]
AD = [sp.Matrix([[1.,0.],[0.,sp.sqrt(1-gamma)]]), sp.Matrix([[0.,sp.sqrt(gamma)],[0.,0.]])]
ApprxAD = [sp.sqrt(1-pm)*Isym, sp.sqrt(pm)*sp.Matrix([[1.,0.],[0.,0.]]), sp.sqrt(pm)*sp.Matrix([[0.,1.],[0.,0.]])]
ApproxAD_Pauli = [sp.sqrt(1-gamma)*Isym, sp.sqrt(gamma/2)*Xsym, sp.sqrt(gamma/2)*Ysym]
PolC = [sp.sqrt(1-p)*Isym, sp.sqrt(p)*(sp.cos(phi)*Xsym + sp.sin(phi)*Ysym)]
ApproxPolC = [sp.sqrt(1-p1-p2)*Isym, sp.sqrt(p1)*Xsym, sp.sqrt(p2)*HXYsym]

ApproxPolC_constr = [sp.sqrt(1-paZ-paX-paH)*Isym, sp.sqrt(paZ)*Zsym, sp.sqrt(paX)*Xsym, sp.sqrt(paH)*HXYsym]

worst_dens_CC = (1./2.)*(Isym + sp.cos(5.*pi/8.)*Xsym + sp.sin(5.*pi/8.)*Ysym)
worst_dens_PC = (1./2.)*(Isym + sp.cos(6.*pi/8.)*Xsym + sp.sin(6.*pi/8.)*Ysym)




"""Basis for 2X2 matrices (in numpy and sympy matrix forms)"""
Pauli_num = [I,X,Y,Z]
Norm_Pauli_num = [(1./sqrt(2))*element for element in Pauli_num]

Pauli_sym = [Isym, Xsym, Ysym, Zsym]
Norm_Pauli_sym = [(1/sp.sqrt(2))*basis_element for basis_element in Pauli_sym]



"""Basis for 4X4 matrices"""
Pauli_2q_num = [np.kron(P1, P2) for P1 in Pauli_num for P2 in Pauli_num]
Norm_Pauli_2q_num = [(1./2.)*element for element in Pauli_2q_num]




def change_matrix_type(matrix, num_to_sym=True):
        """
        Change a matrix or array from numpy to sympy and a sympy matrix
        to numpy matrix.
        """
        if num_to_sym:
                n = len(matrix)
                new_matrix = sp.Matrix([[matrix[i,j] for j in range(n)] for i in range(n)])
        else:
                n = int(sqrt(len(matrix)))
                new_matrix = np.matrix([[complex(sympy_func.re(matrix[i,j]), sympy_func.im(matrix[i,j])) for j in range(n)] for i in range(n)])
        return new_matrix



def decomposition(A, q=1):
        """
        The decomposition function takes a numpy matrix and returns its decomposition in the normalized Pauli basis.
        Currently, it is only implemented for 1-qubit (2X2) and 2-qubit (4X4) matrices.
        """
        if q == 1:
                return [np.trace(A*element) for element in Norm_Pauli_num]
        elif q == 2:
                return [np.trace(A*element) for element in Norm_Pauli_2q_num]



def decomposition_sympy(A):
        """The same function but for a sympy array instead of a numpy matrix."""
        return [(element*A).trace() for element in Norm_Pauli_sym]



"""1-qubit Clifford group"""
Cliff = [[I,S,S*S,S*S*S],[H,H*S,H*S*S,H*S*S*S],[S*H,S*H*S,S*H*S*S,S*H*S*S*S],[H*S*S*H,H*S*S*H*S,H*S*S*H*S*S,H*S*S*H*S*S*S],[S*S*H,S*S*H*S,S*S*H*S*S,S*S*H*S*S*S],[Sinv*H,Sinv*H*S,Sinv*H*S*S,Sinv*H*S*S*S]]

"""Pauli measurements"""
meas_Zp = [np.matrix([[1.,0.],[0.,0.]]), np.matrix([[0.,1.],[0.,0.]])]  # translation to |0>
meas_Xp = [H*A*H for A in meas_Zp]                                      # translation to |+>
meas_Yp = [S*A*Sinv for A in meas_Xp]                                   # translation to |+i>
meas_Zn = [H*Z*A*Z*H for A in meas_Xp]                                  # translation to |1>
meas_Xn = [Z*A*Z for A in meas_Xp]                                      # translation to |->
meas_Yn = [Sinv*A*S for A in meas_Xp]                                   # translation to |-i>



def process_matrix(operators, q=1, output_sympy=True, input_numpy=True):
        """
        This takes a list of operators (2x2 numpy matrices or arrays) 
        and returns the corresponding process matrix in the normalized Pauli basis
        as a sympy matrix or numpy array. Currently, only the sympy option is used.
        """
        n = len(operators)
        d = 2**q
        l = d**2
        if output_sympy:
                PM = sp.Matrix([[0.+0.j for x in range(l)] for y in range(l)])
        else:
                PM = np.array([[0.+0.j for x in range(l)] for y in range(l)])
        amplitudes = []
        if input_numpy:
                for operator in operators:
                        amplitudes.append(decomposition(operator, q))
        else:
                for operator in operators:
                        amplitudes.append(decomposition_sympy(operator))
        for i in range(l):
                for j in range(l):
                        a = np.array([amplitudes[k][i] for k in range(n)])
                        b = np.array([amplitudes[k][j].conjugate() for k in range(n)])
                        if output_sympy:

                                PM[i,j] = np.dot(a,b)
                        else:
                                PM[i][j] = np.dot(a,b)
        return PM



def Liouvillian(channel, d=2, basis='Pauli', sym=False):
    '''
    Calculates the Liouvillian of a quantum channel.
    The default basis is the orthonormal Pauli basis.
    So far, specific for 1 or 2 qubits (d = 2 or 4).
    '''
    if sym:
        pass
	
    else:
        Liou_mat = np.matrix([[complex(0.,0.) for i in range(d**2)] 
				  for j in range(d**2)])

        if d == 2:    basis_set = Norm_Pauli_num
        elif d == 4:  basis_set = Norm_Pauli_2q_num

        for i in range(d**2):
            for j in range(d**2):
                Liou_final = final_dens_matrix_num(
				        channel, 
					basis_set[j], 
					d)
                Liou_mat[i,j] = np.trace(basis_set[i]*Liou_final)

    return Liou_mat	
					


def final_dens_matrix(channel, initial_dens_matrix='sym'):
    """
    This takes as input a channel (either a list of 2X2 matrices
    or a 4X4 matrix) and returns the final density matrix.
    The matrices can be numpy matrices, arrays, or sympy matrices.
    In case "sym" is chosen, then the initial density matrix is symbolic.
    """
    dens_final = sp.Matrix([[0.,0.],[0.,0.]])

    if initial_dens_matrix == "sym":
        dens = rhosym
    
    else:
        if (isinstance(initial_dens_matrix, np.matrix) or \
            isinstance(initial_dens_matrix, np.ndarray)):
            initial_dens_matrix = change_matrix_type(initial_dens_matrix)
        
        dens = initial_dens_matrix


    if isinstance(channel, list):
        for kraus in channel:
            if (isinstance(kraus, np.matrix) or isinstance(kraus, np.ndarray)):
                kraus = change_matrix_type(kraus)
            dens_final += (kraus*dens)*(kraus.H)
        
    else:
        if (isinstance(channel, np.matrix) or isinstance(channel, np.ndarray)):
            channel = change_matrix_type(channel)
        
        for i in range(4):
            for j in range(4):
                dens_final += (Norm_Pauli_sym[i]*dens)*(channel[i,j]*\
                              (Norm_Pauli_sym[j].H))
        
    return dens_final



def final_dens_matrix_num(channel, initial_dens_matrix, d=2):
    """
    This takes as input a channel (either a list of 2X2 matrices
    or a 4X4 matrix) and returns the final density matrix.
    Only works if the initial density matrix is a numpy matrix or
    array.
    """
    dens_final = np.matrix([[0.+0.j for i in range(d)]
			            for j in range(d)])
    dens = copy.deepcopy(initial_dens_matrix)

    if isinstance(channel, list):
        for kraus in channel:
            dens_final += (kraus*dens)*(kraus.H)
    
    else:
        if d == 2:    basis_set = Norm_Pauli_num
        elif d == 4:  basis_set = Norm_Pauli_2q_num 

        for i in range(d**2):
            for j in range(d**2):
                dens_final += (basis_set[i]*dens)*(channel[i,j]*\
                              (basis_set[j].H))
        
    return dens_final



"""
tot_group is a list that contains the 30 operations in our error model.
tot_Proc is a list that contains the process matrices for these 30 operations.
Notice that each process matrix is a sympy matrix.
"""
tot_group = [[A] for sublist in Cliff for A in sublist]
tot_group += [meas_Zp,meas_Xp,meas_Yp,meas_Zn,meas_Xn,meas_Yn]
tot_Proc = [process_matrix(A) for A in tot_group]


def coefficients(group=range(30), cons=[]):
        """
        This returns a list of the symbolic coefficients associated with each 
        operator in the approximation channel.  Each coefficient will be an 
        independent paramater in the distance minimization.  
        -group refers to the matrices that we want to include in our 
        approximation channel. For example, if we want to use the Pauli group, 
        then group = [0,2,12,14] because in the list of matrices, 
        0 = I, 2 = Z, 12 = X, 14 = Y.
        -cons refers to the constraints that we want to apply to the coefficients.
        It should be a list of 2-entry lists, each one specifying which coefficients 
        have the same value. For example, if want to use the Pauli group with the 
        constraint that X and Y have the same vaule, then group = [0,2,12,14] 
        and cons = [[12,14]]
        """
        coef = range(len(group)-1)

        if cons!=[]:

                for con in cons:
                        coef[group.index(con[1])-1] = coef[group.index(con[0])-1]

                ind = range(len(coef))
                for i in range(len(coef)-len(cons)):

                        n_list = [coef.index(coef[ind[0]])]
                        n = coef.count(coef[ind[0]])
                        new_coef = copy.deepcopy(coef)
                        while n > 1:
                                l = len(n_list)
                                new_coef.remove(coef[ind[0]])
                                n_list += [new_coef.index(coef[ind[0]])+l]
                                n -= 1
                        for j in n_list:
                                del ind[ind.index(j)]
                        if coef[n_list[0]] != i:
                                for j in n_list:
                                        coef[j] = i
        coef_symbols = []
        for index in coef:
                coef_symbols += [sp.Symbol('c['+str(index)+']', positive=True)]
        iden_coef = 1-sum(coef_symbols)
        coef_symbols.insert(0,iden_coef)
        return coef_symbols



def find_equivalent_Kraus_channel(channel):
	'''
	Calculates an equivalent Kraus channel to {E_k}: {F_j},
	following the procedure in Exercise 9.10 of Mike and Ike
	
	Input: channel needs to be a list of numpy matrices.
	''' 
	
	n_k = len(channel)
	W_matrix = np.matrix([[np.trace((channel[k].H)*channel[i]) for i in range(n_k)] 
                                                                   for k in range(n_k)])

	#print W_matrix	
	eigsystem = np.linalg.eigh(W_matrix)
	#print (eigsystem[1].H)*W_matrix*eigsystem[1]
	eq_channel = []
	for i in range(n_k):
		#eq_channel += [sum([eigsystem[0][i]*eigsystem[1][j,i]*channel[j] for j in range(n_k)])]
		eq_channel += [sum([eigsystem[1][j,i]*channel[j] for j in range(n_k)])]
	#print eigsystem[0][0], eigsystem[1][:,0] 
	#d_matrix = eigsystem[1]*W_matrix*eigsystem[1].H
	#print d_matrix
	
	return eq_channel



def HS_distance(target_process, group=range(30), cons=[]):
        """
        Hilbert-Schmidt distance between the target process matrix (the matrix we are 
        trying to approximate) and the process matrix of the group we choose.
        Notice that target_process should be a sympy matrix.
        """
        Approx = proc_mat_group(group, cons)
        Dif = (Approx - target_process)*(Approx - target_process)
        Dist = ((0.125)*(Dif.trace())).expand()
        return Dist



def ave_fidcons(target, approx=range(30), cons=[]):
        """Constraint: The minimization is performed forcing this expression to be > 0."""
        Approx = proc_mat_group(approx, cons)
        if isinstance(target, list):
                target_process = process_matrix(target)
        elif isinstance(target, mat.Matrix):
                target_process = target

        cons = target_process[0] - Approx[0]
        return cons



def trace_distance_ae(initial_density_matrix, target_process, approx, cons=[]):
        """
        Returns the trace distance between the two final density matrices.
        Notice that approx needs to be a list of integers, which correspond to the operations in the
        approximative group that will be used for the approximation and initial_density_matrix
        needs to be a sympy matrix.
        """
        approx_process = proc_mat_group(approx, cons)
        dens_final_target = final_dens_matrix(target_process, initial_density_matrix)
        dens_final_approx = final_dens_matrix(approx_process, initial_density_matrix)

        Dif = dens_final_target - dens_final_approx
        Eig = Dif.eigenvals().items()
        Dist = 0.5*sum([Eig[i][1]*sympy_func.Abs(Eig[i][0]) 
            for i in range(len(Eig))])
        return Dist



def proc_mat_group(group=range(30), cons=[]):
        """
        This returns the process matrix for the approximation channel we choose to do
        the minimization with.
        """
        coef = coefficients(group, cons)

        proc_matrices = [coef[i]*tot_Proc[elem] for i,elem in enumerate(group)]    

        proc_mat = proc_matrices[0]
        for i in range(1,len(group)):
                proc_mat += proc_matrices[i]
        return proc_mat


def fid(target_unitary, error_channel_operators, density_matrix, symbolic=1):
        """Fidelity between a unitary gate and a non-necessarily unitary gate,
        for a given initial density matrix. This is later used when calculating
        the worst case fidelity.
        Notice that the input format of the general channel is a list of Kraus
        operators instead of a process matrix.  The input format of the target
        unitary is just the matrix itself, not its process matrix.
        symbolic = 1 is the case when the the input matrices are sympy, 
        while symbolic = 0 is used when the input matrices are numpy.
        """
        V, K, rho = target_unitary, error_channel_operators, density_matrix
        if symbolic:
                Tra = (((V.H)*K[0])*rho).trace()
                fid = Tra*(fun.conjugate(Tra))
                for i in range(1,len(K)):
                        Tra = (((V.H)*K[i])*rho).trace()
                        fid += Tra*(fun.conjugate(Tra))
                return fid.expand()
        else:
                Tra = np.trace((V.H)*K[0]*rho)
                fid = Tra*(Tra.conjugate())
                for i in range(1,len(K)):
                        Tra = np.trace((V.H)*K[i]*rho)
                        fid += Tra*(Tra.conjugate())
                return fid



def worst_fidcons(target, approx=range(30), cons=[], dens_matr=meas_Zn[0]):
        """
        This defines the worst fidelity constraint.
        target needs to be expressed as a list of numpy matrices.
        approx is a list of integers, which represent the indexes for the operators
        that we want to use in our approximation. The default is all the operators.
        dens_matr is the initial state that we want to use. The default is |1><1|, 
        which corresponds to the worst initial point for the amplitude damping.
        It needs to be a numpy matrix.
        """
        fid1 = fid(I, target, dens_matr, symbolic=0)
        fid2 = fid(Isymb, Kraus_group(approx, cons), change_matrix_type(dens_matr), symbolic=1)
        cons = fid1 - fid2
        return cons




def constraints(target, approx=range(30), cons=[], cons_type=0, dens_matrs=[meas_Zn[0]]):
        """
        target needs to be a list of numpy matrices
        approx is a list of the operators in our group that we want to use for the approximation
        dens_matr is a list of all the density matrices to which the worst-fidelity constraint is run.
        """
        cons = [coefficients(approx,cons)[0]]
        if cons_type == 0:
                cons += [ave_fidcons(target, approx)]
        elif cons_type == 1:
                for dens_mat in dens_matrs:
                        cons += [worst_fidcons(target, approx, dens_mat)]
        return cons



def target_distance(target, x):
        """
        This function is used in the worst-distance minimization.
        target can be a list of Kraus operators or a process matrix.
        x refers to the 3 coefficients: cx, cy, cz, but it is only
        used as a dummy variable.
        """
        initial_rho = 0.5*(Isym + x[0]*Xsym + x[1]*Ysym + x[2]*Zsym)
        final_rho = final_dens_matrix(target, initial_rho)
        Dif = initial_rho - final_rho
        Eig = Dif.eigenvals().items()
        Dist = 0.5*sum([Eig[i][1]*sp.Abs(Eig[i][0]) for i in range(len(Eig))])
        return Dist



def target_distance_num(target, x):
        """
        This function is used in the worst-distance minimization.
        target can be a list of Kraus operators or a process matrix.
        x refers to the 3 coefficients: cx, cy, cz, but it is only
        used as a dummy variable.
        """
        initial_rho = 0.5*(I + x[0]*X + x[1]*Y + x[2]*Z)
        final_rho = final_dens_matrix_num(target, initial_rho)
      
        return trace_distance(initial_rho, final_rho)


# This is quicker. Notice that this constraint would be different if we were not 
# comparing the channels to the Identity.



# The abs acting on the numerical parameters are meant to avoid
# having very small negative values. It looks weird, but those
# parameters are not meant to be negative in the first place.


def Pauli(px,py,pz, sym=False):
    if sym:
        pass
    else:
        P1 = sqrt(abs(1-px-py-pz))*ap.I
        P2 = sqrt(abs(px))*ap.X
        P3 = sqrt(abs(py))*ap.Y
        P4 = sqrt(abs(pz))*ap.Z
        return [P1, P2, P3, P4] 



def bit_flip(p, sym=False):
    if sym:
        pass
    else:
        P1 = sqrt(abs(1-p))*ap.I
        P2 = sqrt(abs(p))*ap.X
        return [P1, P2]



def DC_1q(p, sym=False):
    if sym:
        pass
    else:
        P1 = sqrt(abs(1.-p))*ap.I
        P2 = sqrt(abs(p/3.))*ap.X
        P3 = sqrt(abs(p/3.))*ap.Y
        P4 = sqrt(abs(p/3.))*ap.Z
        return [P1, P2, P3, P4] 
        


def DC_2q(p, sym=False):
    if sym:
        pass
    else:
        P1 = [sqrt(abs(1.-p))*ap.I, ap.I]
        P2 = [sqrt(abs(p/15.))*ap.I, ap.X]
        P3 = [sqrt(abs(p/15.))*ap.I, ap.Y]
        P4 = [sqrt(abs(p/15.))*ap.I, ap.Z]
        P5 = [sqrt(abs(p/15.))*ap.X, ap.I]
        P6 = [sqrt(abs(p/15.))*ap.X, ap.X]
        P7 = [sqrt(abs(p/15.))*ap.X, ap.Y]
        P8 = [sqrt(abs(p/15.))*ap.X, ap.Z]
        P9 = [sqrt(abs(p/15.))*ap.Y, ap.I]
        P10 = [sqrt(abs(p/15.))*ap.Y, ap.X]
        P11 = [sqrt(abs(p/15.))*ap.Y, ap.Y]
        P12 = [sqrt(abs(p/15.))*ap.Y, ap.Z]
        P13 = [sqrt(abs(p/15.))*ap.Z, ap.I]
        P14 = [sqrt(abs(p/15.))*ap.Z, ap.X]
        P15 = [sqrt(abs(p/15.))*ap.Z, ap.Y]
        P16 = [sqrt(abs(p/15.))*ap.Z, ap.Z]
        full_list = [P1, P2, P3, P4, P5, P6, P7, P8, P9]
        full_list += [P10, P11, P12, P13, P14, P15, P16]
        return full_list


def CX_ion_trap(p, sym=False):
    if sym:
        pass
    else:
        P1 = [sqrt(abs(1.-p))*ap.I, ap.I]
        P2 = [sqrt(abs(p))*ap.Z, ap.X]
        return [P1, P2]


def T1_T2_Geller(gamma, lambd, sym=False):
    if sym:  
        E1 = sp.Matrix([[1,0], [0,sp.sqrt(1-gamma-lambd)]])
        E2 = sp.Matrix([[0,sp.sqrt(gamma)], [0,0]])
        E3 = sp.Matrix([[0,0], [0,sp.sqrt(lambd)]])
    else:    
        E1 = np.matrix([[1.,0.], [0.,sqrt(abs(1.-gamma-lambd))]])
        E2 = np.matrix([[0.,sqrt(abs(gamma))], [0.,0.]])
        E3 = np.matrix([[0.,0.], [0., sqrt(abs(lambd))]])
    return [E1, E2, E3]



def AD(gamma, sym=False, rot_errors=None):
    if sym:
        AD1 = sp.Matrix([[1,0],[0,sp.sqrt(1-gamma)]])
        AD2 = sp.Matrix([[0,sp.sqrt(gamma)],[0,0]])
    else:
        AD1 = np.matrix([[1.,0.],[0.,sqrt(abs(1.-gamma))]])
        AD2 = np.matrix([[0.,sqrt(abs(gamma))],[0.,0.]])
        # This is not the most elegant solution. But it will work
        # for what I wanna show.
        #rot = generate_rotation_operator(None, None, pi/4,
        #           'gate', ['Y'])
        #AD1 = (rot*AD1)*(rot.H)
        #AD2 = (rot*AD2)*(rot.H)
    return [AD1, AD2]


def CMCapproxAD_cons(pm, sym=False):
    if sym:
        A1 = sp.sqrt(1-pm)*sp.Matrix([[1,0],[0,1]])
        A2 = sp.sqrt(pm)*sp.Matrix([[1,0],[0,0]])
        A3 = sp.sqrt(pm)*sp.Matrix([[0,1],[0,0]])
    else:
        A1 = sqrt(abs(1-pm))*ap.I
        A2 = sqrt(abs(pm))*np.matrix([[1.,0.],[0.,0.]])
        A3 = sqrt(abs(pm))*np.matrix([[0.,1.],[0.,0.]])
    return [A1, A2, A3]


def CMCapproxAD_uncons(gamma, sym=False):
    if sym:
        pm = 0.5*(1 + gamma - sp.sqrt(1-gamma))
        A1 = sp.sqrt(1-pm)*sp.Matrix([[1,0],[0,1]])
        A2 = sp.sqrt(pm)*sp.Matrix([[1,0],[0,0]])
        A3 = sp.sqrt(pm)*sp.Matrix([[0,1],[0,0]])
    else:
        pm = 0.5*(1 + gamma - sqrt(1-gamma))
        A1 = sqrt(abs(1-pm))*ap.I
        A2 = sqrt(abs(pm))*np.matrix([[1.,0.],[0.,0.]])
        A3 = sqrt(abs(pm))*np.matrix([[0.,1.],[0.,0.]])
    return [A1, A2, A3]



#def PCapproxAD_cons(px, py, pz, sym=False):
#   if sym:
#       P1 = sp.sqrt(1-px-py-pz)*ap.Isym
#       P2 = sp.sqrt(px)*ap.Xsym
#       P3 = sp.sqrt(py)*sp.Ysym
#       P4 = sp.sqrt(pz)*ap.Zsym
#   else:
#       P1 = sqrt(abs(1-px-py-pz))*ap.I
#       P2 = sqrt(abs(px))*ap.X
#       P3 = sqrt(abs(py))*ap.Y
#       P4 = sqrt(abs(pz))*ap.Z
#   return [P1,P2,P3,P4]


def PCapproxAD_cons(gamma, sym=False):
    if sym:
        raise Exception('No symbolic PC_cons approx to the ADC')
    else:
        p_x, p_y = px(gamma), px(gamma)
        p_z = pz(gamma)
        P1 = sqrt(abs(1-2*p_x-p_z))*ap.I
        P2 = sqrt(abs(p_x))*ap.X
        P3 = sqrt(abs(p_y))*ap.Y
        P4 = sqrt(abs(p_z))*ap.Z
        return [P1, P2, P3, P4] 
        


def PCapproxAD_uncons(gamma, sym=False):
    if sym:
        px, py = 0.25*gamma, 0.25*gamma
        pz = -0.25*gamma + 0.5*(1-sp.sqrt(1-gamma))
        P1 = sp.sqrt(1-px-py-pz)*ap.Isym
        P2 = sp.sqrt(px)*ap.Xsym
        P3 = sp.sqrt(py)*ap.Ysym
        P4 = sp.sqrt(pz)*ap.Zsym
    else:
        px, py = 0.25*gamma, 0.25*gamma
        pz = -0.25*gamma + 0.5*(1-sqrt(1-gamma))
        P1 = sqrt(abs(1-px-py-pz))*ap.I
        P2 = sqrt(abs(px))*ap.X
        P3 = sqrt(abs(py))*ap.Y
        P4 = sqrt(abs(pz))*ap.Z
        #rot = generate_rotation_operator(None, None, pi/4,
        #           'gate', ['Y'])
        #P1 = (rot*P1)*(rot.H)
        #P2 = (rot*P2)*(rot.H)
        #P3 = (rot*P3)*(rot.H)
        #P4 = (rot*P4)*(rot.H)
    return [P1,P2,P3,P4]
    


def PauliapproxAD_uncons(gamma, sym=False):
    '''
    '''
    if sym:
        pass
    else:
        pc = (1./12.)*(gamma - 2*sqrt(1-gamma) + 2)
        P1 = sqrt(abs(1-3*pc))*ap.I
        P2 = sqrt(abs(pc))*ap.X
        P3 = sqrt(abs(pc))*ap.Y
        P4 = sqrt(abs(pc))*ap.Z
    return [P1,P2,P3,P4]
        



def PolXY(p, phi, sym=False):
    if sym:
        AD1 = sp.sqrt(1-p)*ap.Isym
        AD2 = sp.sqrt(p)*(sp.cos(phi)*ap.Xsym + sp.sin(phi)*ap.Ysym)
    else:
        AD1 = sqrt(abs(1-p))*ap.I
        AD2 = sqrt(abs(p))*(cos(phi)*ap.X + sin(phi)*ap.Y)
    return [AD1,AD2]



def CMCapproxPolXY_cons(p, phi, sym=False):
    if sym:
        p1 = (p/3)*(2*sp.cos(2*phi) - sp.sin(2*phi) + 1)
        p2 = (p/3)*(2*sp.sin(2*phi) - sp.cos(2*phi) + 1)
        p3 = (sp.sqrt(2)-1)*(p/6)*(sp.cos(2*phi) + sp.sin(2*phi) - 1)
        A0 = sp.sqrt(1-p1-p2-p3)*ap.Isym
        A1 = sp.sqrt(p1)*ap.Xsym
        A2 = sp.sqrt(p2)*ap.HXYsym
        A3 = sp.sqrt(p3)*ap.Zsym
    else:
        p1 = (float(p)/3.)*(2*cos(2*phi) - sin(2*phi) + 1)
        p2 = (float(p)/3.)*(2*sin(2*phi) - cos(2*phi) + 1)
        p3 = (sqrt(2)-1)*(float(p)/6.)*(cos(2*phi) + sin(2*phi) - 1)
        A0 = sqrt(abs(1-p1-p2-p3))*ap.I
        A1 = sqrt(abs(float(p1)))*ap.X
        A2 = sqrt(abs(float(p2)/2.))*(ap.X + ap.Y)
        A3 = sqrt(abs(float(p3)))*ap.Z  
    return [A0, A1, A2, A3]



def CMCapproxPolXY_uncons(p, phi, sym=False):
    if sym:
        p1 = (p/7)*(4*sp.cos(2*phi) - 3*sp.sin(2*phi) + 3)
        p2 = (p/7)*(4*sp.sin(2*phi) - 3*sp.cos(2*phi) + 3)
        A0 = sp.sqrt(1-p1-p2-p3)*ap.Isym
        A1 = sp.sqrt(p1)*ap.Xsym
        A2 = sp.sqrt(p2)*ap.HXYsym
    else:
        p1 = (float(p)/7.)*(4*cos(2*phi) - 3*sin(2*phi) + 3)
        p2 = (float(p)/7.)*(4*sin(2*phi) - 3*cos(2*phi) + 3)
        A0 = sqrt(abs(1-p1-p2))*ap.I
        A1 = sqrt(abs(float(p1)))*ap.X
        A2 = sqrt(abs(float(p2)/2.))*(ap.X + ap.Y)
    return [A0, A1, A2]



def PCapproxPolXY_cons(p, phi, sym=False):
    if sym:
        p1 = p*(sp.cos(phi))**2
        p2 = p*(sp.sin(phi))**2
        p3 = p*sp.sin(phi)*sp.cos(phi)
        A0 = sp.sqrt(1-p1-p2-p3)*ap.Isym
        A1 = sp.sqrt(p1)*ap.Xsym
        A2 = sp.sqrt(p2)*ap.Ysym
        A3 = sp.sqrt(p3)*ap.Zsym    
    else:
        p1 = p*(cos(phi))**2
        p2 = p*(sin(phi))**2
        p3 = p*sin(phi)*cos(phi)
        A0 = sqrt(abs(1-p1-p2-p3))*ap.I
        A1 = sqrt(abs(float(p1)))*ap.X
        A2 = sqrt(abs(float(p2)))*ap.Y
        A3 = sqrt(abs(float(p3)))*ap.Z  
    return [A0,  A1, A2, A3]



def PCapproxPolXY_uncons(p, phi, sym=False):
    if sym:
        A0 = sp.sqrt(1-p)*ap.Isym
        A1 = sp.sqrt(p)*sp.cos(phi)*ap.Xsym
        A2 = sp.sqrt(p)*sp.sin(phi)*ap.Ysym
    else:
        A0 = sqrt(abs(1-p))*ap.I
        A1 = sqrt(abs(p))*cos(phi)*ap.X
        A2 = sqrt(abs(p))*sin(phi)*ap.Y
    return [A0,  A1, A2]



def PauliapproxPolXY_uncons(p, phi, sym=False):
    '''
    '''
    if sym:
        pass
    else:
        pc = float(p)/3
        P1 = sqrt(abs(1-3*pc))*ap.I
        P2 = sqrt(abs(pc))*ap.X
        P3 = sqrt(abs(pc))*ap.Y
        P4 = sqrt(abs(pc))*ap.Z
    return [P1,P2,P3,P4]



def RZC(theta, sym=False):
    '''
    Rotation about the Z axis by an angle theta
    '''
    if sym:
        #RZ = mat.Matrix([[sp.exp(-0.5j*theta), 0.], [0., sp.exp(0.5j*theta)]])
        RZ = sp.cos(0.5*theta)*ap.Isym - 1.j*sp.sin(0.5*theta)*ap.Zsym
    else:
        #RZ = np.matrix([[exp(-0.5j*theta), 0.], [0., exp(0.5j*theta)]])
        RZ = cos(0.5*theta)*ap.I - 1.j*sin(0.5*theta)*ap.Z
    return [RZ]



def CMCapproxRZ_uncons(theta, sym=False):
    '''
    The CMCa approx to RZC has I, S, Z, and S^t.
    p_S = 0.5*(sin(theta) - cos(theta) + 1)
    '''
    if sym:
        pass
    else:
        # we first move theta to the range [0., 2*pi]
        if theta >= 0.:
            theta = theta%(2*pi)
        else:
            theta = -(-theta%(2*pi)) + 2*pi

        if theta >= 0. and theta < pi/2:
            pS = 0.5*(sin(theta) - cos(theta) + 1)
            A0 = sqrt(1-pS)*ap.I
            A1 = sqrt(pS)*ap.S
        elif theta >= pi/2 and theta < pi:
            pZ = sin(theta - pi/2)
            A0 = sqrt(1-pZ)*ap.S
            A1 = sqrt(pZ)*ap.Z      
        elif theta >= pi and theta < 3*pi/2:
            pZ = cos(theta - pi)
            A0 = sqrt(pZ)*ap.Z
            A1 = sqrt(1-pZ)*ap.Sinv
        elif theta >= 3*pi/2 and theta <= 2*pi:
            pI = 0.5*(sin(theta - 3*pi/2) - cos(theta - 3*pi/2) + 1)
            A0 = sqrt(pI)*ap.I
            A1 = sqrt(1-pI)*ap.Sinv
        
    return [A0, A1]



def CMCapproxRZ_cons(theta, sym=False):
    '''
    I have only obtained an expression for this channel in the interval
    0 <= theta <= pi/2.  In this interval CMCw = I + S
    p_S = sqrt(2)*sin(0.5*theta)
    '''
    if sym:
        pass
    else:
        # we first move theta to the range [0., 2*pi]
        if theta >= 0.:
            theta = theta%(2*pi)
        else:
            theta = -(-theta%(2*pi)) + 2*pi
        
    if theta >= 0. and theta <= pi/2:
        pS = sqrt(2)*sin(0.5*theta)
        A0 = sqrt(1-pS)*ap.I
        A1 = sqrt(pS)*ap.S
    elif theta >= pi/2 and theta < pi:
        pass
    elif theta >= pi and theta < 3*pi/2:
        pass
    elif theta >= 3*pi/2 and theta < 2*pi:
        pass

    return [A0, A1]



def PCapproxRZ_uncons(theta, sym=False):
    '''
    The PCa approx to RZC only has I and Z.
    p_Z = sin(theta/2.)**2
    '''
    if sym:
        pass
    else:
        A0 = cos(theta/2.)*ap.I
        A1 = sin(theta/2.)*ap.Z
    return [A0,  A1]
        


def PCapproxRZ_cons(theta, sym=False):
    '''
    The PCw approx to RZC only has I and Z.
    p_Z = sin(theta/2.)
    '''
    if sym:
        pass
    else:
        pZ = abs(sin(theta/2.))
        A0 = sqrt(abs(1-pZ))*ap.I
        A1 = sqrt(pZ)*ap.Z
    return [A0, A1]


def PCapproxRZ_geom_mean(theta, sym=False):
    '''
    '''
    if sym:
        pass
    else:
        pZ = abs(sin(theta/2.)**3)
        A0 = sqrt(abs(1-pZ))*ap.I
        A1 = sqrt(pZ)*ap.Z
    return [A0, A1]


def PauliapproxRZ_uncons(theta, sym=False):
    '''
    This and PauliapproxRH_uncons are the same channel.
    '''
    if sym:
        pass
    else:
        A0 = cos(theta/2.)*ap.I
        pc = (1./sqrt(3.))*sin(theta/2.)
        A1 = pc*ap.X
        A2 = pc*ap.Y
        A3 = pc*ap.Z
    return [A0, A1, A2, A3]



def RHC(theta, sym=False):
    '''
    Rotation about the Hadamard axis by an angle theta
    '''
    if sym:
        RH = sp.cos(0.5*theta)*ap.Isym - 1.j*sp.sin(0.5*theta)*ap.Hsym
    else:
        RH = cos(0.5*theta)*ap.I - 1.j*sin(0.5*theta)*ap.H
    return [RH]
    


def PCapproxRH_uncons(theta, sym=False):
    '''
    '''
    if sym:
        pass
    else:
        A0 = cos(theta/2.)*ap.I
        pc = (1./sqrt(2.))*sin(theta/2.)
        A1 = pc*ap.X
        A2 = pc*ap.Z
    return [A0,  A1, A2]



def PCapproxRH_cons(theta, sym=False):
    '''
    Meanwhile.  Need to change it later.  MGA 12/24/2015.
    '''
    if sym:
        pass
    else:
        if theta == 0.0:  return [ap.I]
        #if abs(theta - 0.000628318530718) < 1.e-13:
        #    pZ = 0.00015721402300582625
        #    pX = 0.00015694523718530709
        #    pY = 0.00015708046643604149
        #elif abs(theta - 0.00628318530718) < 1.e-13:
        #    pZ = 0.0015706116871571831
        #    pX = 0.0015709757987224071
        #    pY = 0.0015707937377372812
        #elif abs(theta - 0.0628318530718) < 1.e-13:
        #    pZ = 0.015706679100493566
        #    pX = 0.015704079977633515
        #    pY = 0.015705379508687292

        # new addition
        pX, pY, pZ = 0.25*theta, 0.25*theta, 0.25*theta
        
        A0 = sqrt(1-pX-pY-pZ)*ap.I
        A1, A2, A3 = sqrt(pX)*ap.X, sqrt(pY)*ap.Y, sqrt(pZ)*ap.Z

    return [A0, A1, A2, A3]


def PCapproxRH_geom_mean(theta, sym=False):
    '''
    Ken's idea of a Pauli channel that is the
    geometric mean between the PCa and the PCw
    March 12, 2016.  MGA
    '''
    if sym:
        pass
    else:
        p = 0.5*sqrt(abs(sin(theta/2.)**3))
        A0 = sqrt(abs(1-2*p))*ap.I
        A1 = sqrt(p)*ap.X
        A2 = sqrt(p)*ap.Z
    return [A0, A1, A2]


def CMCapproxRH_uncons(theta, sym=False):
    '''
    Meanwhile.  Need to change it later.  MGA 12/24/2015.
    '''
    if sym:
        pass
    else:
        if theta == 0.0:  return [ap.I]
        #if abs(theta - 0.000628318530718) < 1.e-13:
        #    p1 = 0.00017792664316328155
        #    p23 = 0.0001771052499374411
        #elif abs(theta - 0.00628318530718) < 1.e-13:
        #    p1 = 0.0017830289349599975
        #    p23 = 0.0017825839865563062
        #elif abs(theta - 0.0628318530718) < 1.e-13:
        #    p1 = 0.018353262000438565
        #    p23 = 0.018351854049169565

        p1, p23 = 0.283*theta, 0.283*theta
        A0 = sqrt(1-p1-p23)*ap.I
        A1 = sqrt(p1)*ap.S
        #A23 = sqrt(p23)*(ap.Sinv*ap.H*ap.Sinv)
        # Might be faster if we define this matrix explicitly.
        A23 = sqrt(p23)*(sqrt(0.5)*np.matrix([[1.+0.j, 0.-1.j], [0.-1j, 1.+0.j]]))

    return [A0, A1, A23]
        


def CMCapproxRH_cons2(theta, sym=False):
    '''
    Meanwhile.  Need to change it later.  MGA 12/24/2015.
    '''
    if sym:
        pass
    else:
        if theta == 0.0:  return [ap.I]
        if theta == 0.000628318530718:
            C1=np.matrix([[-4.01436550e-10-0.00452518j, 4.83886689e-03+0.00449831j],
 			    [ -4.83886689e-03+0.00449831j, -4.01478671e-10+0.00452518j]])
            C2=np.matrix([[ 5.29399332e-10-0.0034253j,  -6.38061944e-03+0.00341792j],
 		        [ 6.38061944e-03+0.00341792j,   5.29370754e-10+0.0034253j]])
            C3=np.matrix([[ 2.51685126e-06+0.00978445j,   1.26814885e-05+0.00982924j],
 		        [ -1.26814885e-05+0.00982924j,   2.51685126e-06-0.00978445j]])
            C4=np.matrix([[ 9.99839669e-01-0.00012801j,  -8.29544600e-08-0.00012859j],
 		        [ 8.29544600e-08-0.00012859j,   9.99839669e-01+0.00012801j]])


	elif theta == 0.00628318530718:
            C1=np.matrix([[-1.41348799e-08-0.01790645j, -3.19770446e-05+0.01790619j],
 		        [ 3.20053134e-05+0.01790619j, 1.41343243e-08+0.01790645j]])
            C2=np.matrix([[-2.01558721e-10+2.24507752e-05j, -2.53233659e-02-2.27915894e-05j],
 		        [2.53233658e-02-2.27915885e-05j, -2.38223656e-10-2.24507752e-05j]])
            C3=np.matrix([[-7.98967163e-05-0.03098775j, 4.17448542e-07-0.03098819j],
 		        [ -4.17448542e-07-0.03098819j,  -7.98967163e-05+0.03098775j]])
            C4=np.matrix([[ -9.98395525e-01+0.00128708j, -8.66934203e-09+0.0012871j],
 		        [  8.66934203e-09+0.0012871j,   -9.98395525e-01-0.00128708j]])


        elif theta == 0.0628318530718:
            C1=np.matrix([[  3.73500970e-08-0.00203757j,  -8.00210476e-02+0.00204316j],
 		        [  8.00210481e-02+0.00204316j,   3.78329390e-08+0.00203757j]])
            C2=np.matrix([[ -8.05371866e-09-0.05658316j,   2.88551857e-03+0.05658369j],
 		        [ -2.88550518e-03+0.05658369j,   5.34266273e-09+0.05658316j]])
            C3=np.matrix([[ -2.62571377e-03-0.09716455j,  -6.75505693e-06-0.09716329j],
 		        [  6.75505693e-06-0.09716329j,  -2.62571377e-03+0.09716455j]])
            C4=np.matrix([[ -9.83836691e-01+0.01329346j,   4.61839823e-07+0.01329329j],
 		        [ -4.61839823e-07+0.01329329j,  -9.83836691e-01-0.01329346j]])


    return [C1, C2, C3, C4]
                    


def CMCapproxRH_cons(theta, sym=False):
    '''
    Meanwhile.  Need to change it later.  MGA 12/24/2015.
    '''
    if sym:
        pass
    else:
        if theta == 0.0:  return [ap.I]
        #if theta == 0.000628318530718:
        #    p1 = 0.00012766920903233779
        #    p7 = 0.00012842002518564951
        #    p20 = 0.00012808836693458598
        #    p23 = 0.0001288408298481157

        #elif theta == 0.00628318530718:
        #    p1 = 0.0012825292239748579
        #    p7 = 0.0012825305090340109
        #    p20 = 0.0012825650196256577
        #    p23 = 0.0012825660273357777

        #elif theta == 0.0628318530718:
        #    p1 = 0.012823554614700236
        #    p7 = 0.01282426794429159
        #    p20 = 0.012822509343727452
        #    p23 = 0.012823222999367416

        p1, p7 = 0.204*theta, 0.204*theta
        p20, p23 = 0.204*theta, 0.204*theta

        A0 = sqrt(1-p1-p7-p20-p23)*ap.I
        A1 = sqrt(p1)*ap.S
        A7 = sqrt(p7)*(ap.H*ap.Sinv) 
        A20 = sqrt(p20)*(ap.Sinv*ap.H)
        A23 = sqrt(p23)*(ap.Sinv*ap.H*ap.Sinv)

    return [A0, A1, A7, A20, A23]



def PauliapproxRH_uncons(theta, sym=False):
    '''
    Same as PauliapproxRZ_uncons
    '''

    return PauliapproxRZ_uncons(theta, sym) 



def RTC(theta, sym=False):
    '''
    Rotation about the T axis (1/sqrt(3) (X + Y + Z)) by an angle theta
    '''
    if sym:
        pass
    else:
        RT = cos(0.5*theta)*ap.I - 1.j*sin(0.5*theta)*(1./sqrt(3))*(ap.X+ap.Y+ap.Z)
    return [RT]


def sum_matrices(matrix_list, sym=False, dim=[2**7, 2**7]):
    if sym:
        dens = sp.Matrix([[0. for i in range(dim[1])] 
                       for j in range(dim[0])])
        for matrix in matrix_list:
            dens += matrix
    else:
        dens = sum(matrix_list)
    return dens
    


def apply_operation(dens, oper, sym=False):
    """
    """
    if type(dens) == type(''):
        raise ValueError('Density matrix is a string!!')
    if len(dens) != len(oper):
        raise ValueError('Density matrix and operation do not have the same size')
    if sym:
        return (oper*dens)*(quant.Dagger(oper))
    else:
        return (oper*dens)*(oper.H)




def tensor_product(gates_list, sym=False):
    if len(gates_list) == 1:
        return gates_list[0]
    if sym:
        if len(gates_list) > 2:
            return quant.TensorProduct(gates_list[0], 
                tensor_product(gates_list[1:], sym))
        else:
            return quant.TensorProduct(gates_list[0], 
                           gates_list[1])

    else:   
        if len(gates_list) > 2:
            return np.kron(gates_list[0], 
                tensor_product(gates_list[1:], sym))
        else:
            return np.kron(gates_list[0], gates_list[1])



####################  create-states functions  ####################


def create_ket(list_states):
    list_matrices = [state_vector_dic[name_gate_dic[state]] 
                    for state in list_states]
    return tensor_product(list_matrices)



def create_logical_state_Shor(state='zero'):
    """
    Returns the vector corresponding to Shor's logical zero or one.
    """
    zeros = create_ket(['zero','zero','zero'])
    ones = create_ket(['one','one','one'])
    plus = (1./sqrt(2))*(zeros + ones)  
    minus = (1./sqrt(2))*(zeros - ones) 
    if state == 'zero':
        return tensor_product([plus,plus,plus])
    elif state == 'one':
        return tensor_product([minus,minus,minus])
    else:
        raise NameError('Only zero or one implemented')



def create_logical_state_422(state1='zero', state2='zero'):
    '''
    Incomplete for now
    '''
    if state1 == 'zero' and state2 == 'zero':
        keta = create_ket(['zero', 'zero', 'zero', 'zero'])
        ketb = create_ket(['one', 'one', 'one', 'one'])
        ket_total = (1./sqrt(2))*(keta + ketb)
        return ket_total
        



def create_logical_state_BS(state='zero'):
    '''
    In the Bacon-Shor[[9,1,3]] code, in order to have the logical
    Z be a string of physical Zs and likewise for the logica X,
    we take interchange |0> and |+> (and therefore |1> and |-> too) 
    '''
    plus = create_logical_state_Shor('zero')
    minus = create_logical_state_Shor('one')
    
    if state == 'zero':
        return (1./sqrt(2))*(plus + minus)
    elif state == 'one':
        return (1./sqrt(2))*(plus - minus)



def create_logical_state_Steane(state='zero', sym=False):
    """
    Returns the vector corresponding to Steane's logical zero or one.
    """
    if state != 'zero' and state != 'one':
        raise NameError('Only zero or one implemented.') 
    kets = [0,85,51,102,15,90,60,105]
    states = []
    
    for ket in kets:
        list_numbers = [str(num[0]) for num in binary(ket, 7)]
        list_states = [state_vector_dic[name_gate_dic[symbol_word_dic[number]]] 
                   for number in list_numbers]
        states += [tensor_product(list_states)]
    
    ket = sum(states)
    if state == 'one':
        X = tensor_product([gate_matrix_dic[gate] 
                for gate in ['X','X','X','X','X','X','X']])
        ket = X*ket
    
    if sym:
        return (1./sp.sqrt(8))*sp.Matrix([elem for elem in ket])
    else:
        return (1./sqrt(8))*ket 



def create_logical_state_5qubit(state='zero', sym=False):
    """
    Returns the vector corresponding to Steane's logical zero or one.
    """
    stabs = fivequbit.Code.stabilizer + [fivequbit.Code.logical['Z']]   
    
    list_states = [state_vector_dic['PrepareZPlus'] for i in range(5)]
    zero_ket = tensor_product(list_states)  
 
    I_matrix = tensor_product([gate_matrix_dic['I'] for i in range(5)])
    matrix_list = []
    for stab in stabs:
        stab_matrix = tensor_product([gate_matrix_dic[gate]
                for gate in stab])  
        matrix_list += [I_matrix + stab_matrix]

    rotation = matrix_list[0]*matrix_list[1]*matrix_list[2]*matrix_list[3]*matrix_list[4]
    ket = (1./sqrt(2**6))*rotation*zero_ket
    
    if state == 'one':
        X = tensor_product([gate_matrix_dic[gate] for gate in fivequbit.Code.logical['X']])
        ket = X*ket

    return ket



def create_logical_state_color(state='zero', sym=False):
    '''
    We start with the 'zero_ket' = |0000000>, and we apply the rotation
    (I + S1)*(I + S2)*(I + S3), where S1, S2, S3 are the X stabilizers.
    We then normalize the resulting ket.  If the state is 'one' we apply
    a logical X rotation.
    This also works for the Steane code, and we should merge these
    methods later on.
    '''
    stabs = st.Code.stabilizer_color_code[:3] 

    list_states = [state_vector_dic['PrepareZPlus'] for i in range(7)]
    zero_ket = tensor_product(list_states)  
    
    I_matrix = tensor_product([gate_matrix_dic['I'] for i in range(7)])
    matrix_list = []
    for stab in stabs:
        stab_matrix = tensor_product([gate_matrix_dic[gate]
                for gate in stab])  
        matrix_list += [I_matrix + stab_matrix]
    
    rotation = (1./sqrt(8.))*matrix_list[0]*matrix_list[1]*matrix_list[2]
    ket = rotation*zero_ket
    if state == 'one':
        X = tensor_product([gate_matrix_dic[gate] 
                for gate in ['X','X','X','X','X','X','X']])
        ket = X*ket

    return ket
        


def initial_state_general(theta, phase_angle, code='Steane', ket=False):
    '''
    Generates a physical and a logical 1-qubit density matrix,
    corresponding to the angles theta and phi and a particular
    QECC.
    '''
    phase = complex(np.cos(phase_angle), np.sin(phase_angle))
    phys_ket = np.cos(theta/2)*zero + phase*np.sin(theta/2)*one
    
    if code == 'color':
        log_0 = create_logical_state_color('zero')
        log_1 = create_logical_state_color('one')
    elif code == 'Steane':
        log_0 = create_logical_state_Steane('zero')
        log_1 = create_logical_state_Steane('one')
    elif code == 'BS':
        log_0 = create_logical_state_BS('zero')
        log_1 = create_logical_state_BS('one')
    elif code == '5qubit':  
        log_0 = create_logical_state_5qubit('zero')
        log_1 = create_logical_state_5qubit('one')
    elif code == 'TMR':
        log_0 = create_ket(['zero','zero','zero'])
        log_1 = create_ket(['one','one','one'])
    log_ket = np.cos(theta/2)*log_0 + phase*np.sin(theta/2)*log_1
        
    if ket:
        return phys_ket, log_ket    
    else:
        phys_dens = phys_ket*(phys_ket.H)
        log_dens = log_ket*(log_ket.H)
        
        return phys_dens, log_dens



def initial_state_general_different_basis(theta_vector, phi_vector,
                                          theta, phi, code='Steane',
                                          ket=False, stabs_rot=None):
    '''
    Takes in a point on the Bloch sphere defined by (theta_vector, 
    phi_vector).
    The idea is to use this point to define a new pair of ortho-
    normal vectors and then use them to define the point you want.
    So 'theta_vector' and 'phi_vector' are the angles used to define
    the new basis vectors.  
    'theta' and 'phi' are used to define the point taking the other
    points as reference.
    This function will be used to select specific points on the Bloch
    sphere to determine any symmetries on the threshold of the PolC. 
        
    For this function, stabs_rot is not a 1-qubit rotation operator,
    but rather a rotation operator that spans all the physical
    qubits (7 for the Steane code)
    '''

    # Define theta and phi for the |axis-> vector
    theta_vector_1 = pi - theta_vector
    # To not introduce a global phase on the one ket
    # that will later turn into a relative phase,
    # we do the following:
    if theta_vector == 0.0 or theta_vector == pi:  
        phi_vector_1 = 0.0
    else:
        phi_vector_1 = phi_vector + pi

    # Define the new basis
    zero_phys, zero_log = initial_state_general(theta_vector,
                                                phi_vector,
                                                code,
                                                True)

    one_phys, one_log = initial_state_general(theta_vector_1,
                                                phi_vector_1,
                                                code,
                                                True)

    # Define states
    phase = complex(np.cos(phi), np.sin(phi))
    phys_ket = np.cos(theta/2)*zero_phys + phase*np.sin(theta/2)*one_phys
    log_ket = np.cos(theta/2)*zero_log + phase*np.sin(theta/2)*one_log

    # Notice that stabs_rot only acts on the logical ket
    # The physical ket remains the same.
    if stabs_rot != None:
        log_ket = stabs_rot*log_ket
    if ket:
        return phys_ket, log_ket
    else:
        phys_dens = phys_ket*(phys_ket.H)
        log_dens = log_ket*(log_ket.H)
        return phys_dens, log_dens
        



def convert_mat(rho, code, log_to_phys=True):
    '''
    Converts from a physical to a logical state
    and viceversa
    '''
    log_0 = initial_state_general(0.,0.,code,True)[1]
    log_1 = initial_state_general(pi,0.,code,True)[1]
    if log_to_phys:
        zero_zero = (log_0.H*rho*log_0)[0,0]
        zero_one = (log_0.H*rho*log_1)[0,0]
        one_zero = (log_1.H*rho*log_0)[0,0]
        one_one = (log_1.H*rho*log_1)[0,0]
        small_mat = np.matrix([[zero_zero, zero_one],[one_zero, one_one]])
    
        return small_mat
    
    else:
        zero_zero = rho[0,0]*(log_0*(log_0.H))
        zero_one = rho[0,1]*(log_0*(log_1.H))
        one_zero = rho[1,0]*(log_1*(log_0.H))
        one_one = rho[1,1]*(log_1*(log_1.H))
        big_matrix = zero_zero + zero_one + one_zero + one_one

        return big_matrix



def cat_state_dens_matr(n=4):
    '''
    Returns the density matrix corresponding to a n-qubit cat state
    No symbolic option yet.
    '''
    zeroes = create_ket(['zero' for i in range(n)])
    ones = create_ket(['one' for i in range(n)])
    cat_ket = (1./sqrt(2.))*(zeroes + ones)
    cat_dens = cat_ket*(cat_ket.H)

    return cat_dens




####################  Cat state functions  ###################


def cat_state_no_verify(n=4, from_1=False):
        '''
        Creates an n-qubit cat state with no verification.
        '''
        circ = c.Circuit()
        for i in range(n):
                circ.add_gate_at([i], 'PrepareZPlus')
        circ.add_gate_at([0], 'H')
        if from_1:
                for i in range(n-1):
                        circ.add_gate_at([0,i+1], 'CX')
        else:
                for i in range(n-1):
                        circ.add_gate_at([i,i+1], 'CX')

        return circ



def cat_state_1_verify(n=4, from_1=False):
        '''
        Creates an n-qubit cat state with 1 verification step
        between the first and last qubits.
        Warning: If you try to visualize this circuit on the web browser
        using Yu's tool, be aware that it will mess up the CNOTs.  The 
        circuit is still OK; it's just that the order will be different
        '''
        circ = cat_state_no_verify(n, from_1)
        circ.add_gate_at([n], 'PrepareZPlus')
        circ.add_gate_at([0,n], 'CX')
        circ.add_gate_at([n-1,n], 'CX')
        circ.add_gate_at([n], 'MeasureZ')
        circ.to_ancilla([n])
        return circ



def cat_state_n_verify(n=4, from_1=False, n_verify=1):
        '''
        Creates an n-qubit cat state with n verification steps
        between the first and last qubits.
        Warning: If you try to visualize this circuit on the web browser
        using Yu's tool, be aware that it will mess up the CNOTs.  The 
        circuit is still OK; it's just that the order will be different.
        Notice that we are essentially measuring the stabilizers of the cat
        state.  The first (n-1) stabilizers consist of nearest-neighboring
        ZZ operators (Z1Z2, Z2Z3, etc.).  The last stabilizer consists of
        a string of X's.  In practice, we will never measure all stabilizers
        to verify a cat state.  I include this mainly for debugging purposes.
        '''
        if n_verify > n:
                raise ValueError('The number of verification steps cannot be \
                                 greater than the number of qubits in the \
                                 cat state.')

        circ = cat_state_no_verify(n, from_1)
        for i in range(n_verify):
                if i != 3:
                        circ.add_gate_at([n+i], 'PrepareZPlus')
                        circ.add_gate_at([i, n+i], 'CX')
                        circ.add_gate_at([i+1, n+i], 'CX')
                        circ.add_gate_at([n+i], 'MeasureZ')
        if n == n_verify:
                circ.add_gate_at([2*n - 1], 'PrepareZPlus')
                circ.add_gate_at([2*n - 1], 'H')
                for i in range(n):
                        circ.add_gate_at([2*n - 1, i], 'CX')
                circ.add_gate_at([2*n - 1], 'H')
                circ.add_gate_at([2*n - 1], 'MeasureZ')
        for i in range(n, n + n_verify):
                circ.to_ancilla([i])

        return circ



def normalize_cat_state_matrices(states_dict):
    '''
    When working with cat-state ancillae, we don't do the branching
    with ALL possibilities, because we only care about the outcomes
    with a definite parity. (This means that for a stabilizer the
    two outcomes '0000' and '1100' give rise to the same state with
    the same probability.)
    Because of this reduced branching, the final states are not
    normalized.
    '''
    total_trace = float(sum([np.trace(state).real for state in states_dict.values()]))
    if total_trace == 0.0:
        return states_dict
    new_states_dict = {}
    for outcome in states_dict:
        new_states_dict[outcome] = (1./total_trace)*states_dict[outcome]

    return new_states_dict



####################  end cat state functions  ####################




def overlap(dens_matr1, dens_matrs2, fidelity=False, sym=False):
    '''
    The second argument is now a list of density matrices
    The function returns the sum of the overlap between
    dens_matr1 and every density matrix in dens_matrs2.
    This was changed in order to implement Ken's suggestion
    regarding how to measure the overlap betweem the initial
    and the final states when we have errors after every gate.
    MGA: 4/1/14 (April Fools' Day!  HAHA!! But this is true.)
    
    If fidelity == True, the output is precisely the fidelity
    (the square root of the overlap)
    If fidelity == False, the output is just the overlap.
    Notice that this definition of fidelity is true only when
    one of the states is pure.
    
    CAUTION: Notice that this ONLY works if one of the density
    matrices (the first one) is PURE.
    '''
    if sym:
        # change the symbolic part later on.
        return (dens_matr1*dens_matr2).trace()
    else:
        traces = []
        for dens_matr in dens_matrs2:
            trace = np.trace(dens_matr1*dens_matr)
            traces += [trace]

        if fidelity:  return sqrt(abs(sum(traces)))
        else:         return abs(sum(traces))



def trace_distance(dens_matr1, dens_matr2, sym=False):
    Dif = dens_matr1 - dens_matr2
    if sym:
        Eig = Dif.eigenvals()
        first_key = Eig.keys()[0]
        Sum = sp.Abs(Eig[first_key]*first_key)
        if len(Eig) > 1:
            for key in Eig.keys()[1:]:
                Sum += sp.Abs(Eig[key]*key)
        return Sum/2
    
    else:
        Eig = np.linalg.eigvalsh(Dif)
        return 0.5*sum([abs(eigen) for eigen in Eig])



def comp_basis(dim, sym=False):
    """
    Generates all the computational states in the space of dimension dim.
    """
    if sym:
        basis = [quant.Dagger(sp.Matrix([[0 for i in range(2**dim)]])) 
                        for j in range(2**dim)]
        for i in range(2**dim):
            basis[i][i] = 1
    else:
        basis = [np.matrix([[float(0) for i in range(2**dim)]]).H 
                        for j in range(2**dim)]
        for i in range(2**dim):
            basis[i][i] = 1.
    return basis


### Functions related to the diamond distance ###


def diagonalize_proc_mat(proc_mat, basis='standard'):
    '''
    Diagonalizes a process matrix and returns a Kraus channel
    '''
    if basis == 'standard':
        basis = [sqrt(0.5)*oper for oper in [ap.I, ap.X, ap.Y, ap.Z]]


    proc_mat = np.array(proc_mat)
    eigvals, eigvecs = np.linalg.eig(proc_mat)
    
    new_Kraus = []
    dim = len(eigvals)
    for i in range(dim):
        eigval = eigvals[i]
        eigvec = eigvecs[:,i]
        if abs(eigval) < 1.e-15:
            continue
        
        factor = sqrt(abs(eigval))
        new_oper = factor*sum([eigvec[j]*basis[j] for j in range(dim)])
        new_Kraus += [new_oper]

    return new_Kraus
    


def write_Kraus_opers(operators):
	'''
	Writes the Kraus operators onto a text file
	'''
	number = len(operators)
        size = len(operators[0])
        filename = 'kraus_operators.txt'
	f = open(filename, 'w')
        f.write('%d %d \n' %(number, size))
        for op in operators:
                for i in range(size):
                        for j in range(size):
				#entry = op[i,j]
				#if abs(entry.real) < 1.e-11:
				#	entry_real = 0.
				#else:
				#	entry_real = entry.real
				#if abs(entry.imag) < 1.e-11:
				#	entry_imag = 0.
				#else:
				#	entry_imag = entry.imag
				
				entry = repr(op[i,j])
				#entry = repr(complex(entry_real,entry_imag))
				entry = entry.replace('(', '')
				entry = entry.replace(')', '')
				f.write(entry + ' ')
                        f.write('\n')
        f.close()


def get_diamond_distance(operators, folder):
	'''
	Calls the matlab script to calculate the diamond distance
	operators:  list of Kraus operators
	folder:     folder where the matlab script is located
	'''
	write_Kraus_opers(operators)
	command = ('matlab -nojvm < %srun_diamond.m' %folder)
	b = os.system(command)
        filename = 'kraus_operators.txt'
	last_line = file(filename, "r").readlines()[-1]
        
	return(float(last_line))
		



def trace_out_ancillary_subsystem(dens_matrix, num_data, num_ancilla, sym=False):
    """
    Assumes the ancillary subsystem is at the "end".
    """
    basis = comp_basis(num_ancilla, sym)
    list_results = []
    if sym:
        for i in range(len(basis)):
            pre_projector = [gate_matrix_dic_sym['I'] 
                    for j in range(num_data)] + [basis[i]]
            projector = tensor_product(pre_projector, sym)
            result = quant.Dagger(projector)*(dens_matrix)*projector
            list_results += [result]
        final_dens_matr = mat.zeros(2**num_data)
        for result in list_results:
            final_dens_matr += result
        return final_dens_matr
    else:
        for i in range(len(basis)):
            pre_projector = [gate_matrix_dic['I'] 
                for j in range(num_data)] + [basis[i]]
            projector = tensor_product(pre_projector, sym)
            result = (projector.H)*(dens_matrix)*(projector)
            list_results += [result]
        return sum(list_results)
            



def split_circuit(circ, number_ancilla):
    """
    This function is used in the class Whole_Circuit during initialization.
    We split the circuit into subcircuits that end with 'number_ancilla'
    ancilla-qubit measurements.
    """
    i = 0
    list_indexes = []
    for gate in circ.gates:
        cond1 = gate.gate_name[:7] == 'Measure'
        cond2 = gate.qubits[0].qubit_type == 'ancilla'
        if (cond1 and cond2):
            i += 1
            if i == number_ancilla:
                i = 0
                list_indexes += [circ.gates.index(gate)]
    
    list_subcircuits = []
    first_index = 0
    for index in list_indexes:
        subcircuit = Circuit(gates = circ.gates[first_index:index+1])
        subcircuit.update_map()
        list_subcircuits += [subcircuit]    
        first_index = index + 1     

    return list_subcircuits



def insert_errors(circ, error_gate, after_gates=['PrepareZPlus', 'H', 'CX'], kind=1):
        '''
        I had originally planned to use MUSIQC's monte tools to do this, but
        it turned out to be a little over-complicated.
        Inputs:  - circ: circuit where the errors are to be inserted
                 - error_gate: error gate to be inserted (string)
                 - after_gates: the gates after which we want to insert errors
                   Notice that they by default we don't insert errors after
                   measurement gates.
        We need to add two modifications to this function:
          (1) Instead of changing the initial circuit, create another circuit 
              instance and leave the initial circuit unchanged.
          (2) Add an option where you can add errors at each time step, not 
              necessarily after each gate.  Maybe this would be easier to do
              by creating circuits with 'I' gates where qubits are idle.
        '''
        for g in circ.gates[::-1]:
                if g.gate_name in after_gates:
                        if len(g.qubits) == 1:
                                if ((kind==1 or kind==2) and
                                     g.qubits[0].qubit_type=='ancilla'):
                                        continue
                                new_g = circ.insert_gate(g, g.qubits, '',
                                                         error_gate, False)
                                new_g.is_error = True
                        else:
                                q1, q2 = g.qubits[0], g.qubits[1]

                                if (kind==1 and (q1.qubit_type=='ancilla' or
                                    q2.qubit_type=='ancilla')):
                                        continue

                                elif kind==2:
                                        if q2.qubit_type=='data':
                                                new_g = circ.insert_gate(g, 
                                [q2], '',
                                                                error_gate, False)
                                                new_g.is_error=True
                                        if q1.qubit_type=='data':
                                                new_g = circ.insert_gate(g, 
                                [q1], '',
                                                                error_gate, False)
                                                new_g.is_error=True

                                else:
                                        for q in g.qubits[::-1]:
                                                new_g = circ.insert_gate(g, 
                                [q], '',
                                                                error_gate, False)
                                                new_g.is_error = True
        return None


def get_syndrome_Steane(outcomes, error='Z'):
    '''
    Just for one type of stabilizers
    outcomes needs to be a list of 3 bits
    '''
    # if the input is of the form ['0 1 1']
    # instead of [0, 1, 1], convert it.
    #print 'outcomes =', outcomes
    #print 'outcomes type =', type(outcomes)

    if len(outcomes) == 1:
        if type(outcomes) == type([]):
            outcomes = map(int, outcomes[0].split())
    
    if type(outcomes) == type(''):
        if outcomes.count(' ') == 0:
            outcomes = map(int, outcomes)
        else:
            outcomes = map(int, outcomes.split())

    #print 'outcomes =', outcomes
    qubit = 4*outcomes[0] + 2*outcomes[1] + outcomes[2]
    #print 'qubit', qubit

    syndrome = ['I' for i in range(7)]
    if qubit != 0:
        syndrome[qubit-1] = error
        
    corr_mat = tensor_product([gate_matrix_dic[gate] for gate in syndrome])
    
    return corr_mat



def get_syndrome(outcomes, sym=False, code='Steane', 
                rot=np.matrix([[1.,0.],[0.,1.]])):
    """
    """
    if type(outcomes) == type(''):
        outcomes = [int(outcome) for outcome in outcomes]

    if code == 'TMR':
        qubit = 2*outcomes[0] + outcomes[1]
        syndrome = ['I' for i in range(3)]
        if qubit == 1:
            syndrome[2] = 'X'
        elif qubit == 2:
            syndrome[0] = 'X'
        elif qubit == 3:
            syndrome[1] = 'X'

        return tensor_product([gate_matrix_dic[gate] 
                            for gate in syndrome])
    

    elif code == 'T_gate_Steane':
        qubit = 2*outcomes[0] + outcomes[1]
        if qubit == 0:
            syndrome = ['Sinv' for i in range(7)]
        elif qubit == 1:
            syndrome = ['Sinv','S','S','Sinv','Sinv','Sinv','Sinv']
        elif qubit == 2:
            syndrome = ['I','Z','I','I','I','I','I']
        elif qubit == 3:
            syndrome = ['I','I','Z','I','I','I','I']

        return tensor_product([gate_matrix_dic[gate] 
                            for gate in syndrome])
       
    elif code == 'T_gate_BS':
        if outcomes[0] == 1 and outcomes[1] == 1:
            syndrome = ['I','I','I','Z','I','I','I','I','I']
        elif outcomes[0] == 0 and outcomes[1] == 1:
            syndrome = ['I','I','I','I','I','I','Z','I','I']
        else:
            pass

        return tensor_product([gate_matrix_dic[gate] 
                            for gate in syndrome])
      

    elif code == 'Steane':
        X_meas = outcomes[:3]
        Z_meas = outcomes[3:]

        Z_qubit = 4*X_meas[0] + 2*X_meas[1] + X_meas[2]
        X_qubit = 4*Z_meas[0] + 2*Z_meas[1] + Z_meas[2]

        Z_syndrome = ['I' for i in range(7)]
        X_syndrome = ['I' for i in range(7)]
        if Z_qubit != 0:
            Z_syndrome[Z_qubit-1] = 'Z'
        if X_qubit != 0:
            X_syndrome[X_qubit-1] = 'X'

        #print 'Outcome =', outcomes
        #print 'Z sydnrome =', Z_syndrome
        #print 'X syndrome = ', X_syndrome


    elif code == 'Shor':
        Z_meas = [[entry[0] for entry in outcomes[2*i:2*i+2]] 
                            for i in range(3)]
        X_meas = [entry[0] for entry in outcomes[6:]]

        X_outcomes = []
        for meas in Z_meas:
            qubit = 2*meas[0] + meas[1]
            X_qubits += [bit_flip_code_interpreter(outcome)]
        qubit = 2*X_meas[0] + X_meas[1]
        Z_qubit = bit_flip_code_interpreter(Z_outcome)

        X_syndrome = ['I' for i in range(9)]
        Z_syndrome = ['I' for i in range(9)]

        if Z_qubit != 'n':
            for i in range(3*Z_qubit, 3*(Z_qubit+1)):
                            Z_syndrome[i] = 'Z'

        for i in range(3):
            if X_qubits[i] != 'n':
                    X_syndrome[3*i + X_qubits[i]] = 'X'




    # Haven't implemented the rotation for the symbolic case
    # MGA 11/4/14.
    if sym:
        corr_mat_Z = tensor_product([gate_matrix_dic_sym[gate] 
                         for gate in Z_syndrome], sym)
        corr_mat_X = tensor_product([gate_matrix_dic_sym[gate] 
                         for gate in X_syndrome], sym)

    else:
        corr_list_Z = [(rot*gate_matrix_dic[gate])*(rot.H)
                   for gate in Z_syndrome]
        corr_list_X = [(rot*gate_matrix_dic[gate])*(rot.H)
                   for gate in X_syndrome]
        
        corr_mat_Z = tensor_product(corr_list_Z)
        corr_mat_X = tensor_product(corr_list_X)

    return corr_mat_Z*corr_mat_X



# Original apply correction.
# Currently not used.
#def apply_correction_to_every_state(states):
#   '''
#   '''
#   corrected_states = [states[0]]
#       for i in range(1, len(states)):
#           outcomes = sim.binary(i, length)
#               correction = get_syndrome(outcomes, sym, code)
#               corrected_states += [sim.apply_operation(states[i], correction, sym)]
#        return corrected_states



def apply_correction_to_every_state(states_dict, sym=False, code='Steane', 
                    length=6, error='Z',
                    rot=None):
    """
        """
    if rot == None:
        rot = np.matrix([[1., 0.], [0., 1.]])
    new_states_dict = {}
    for outcome in states_dict:
        if ' ' in outcome:
            mod_outcome = map(int,outcome.split(' '))
        else:
            mod_outcome = [int(s) for s in outcome]
        
        if code == 'Steane':
            if length == 6:  
                correction = get_syndrome(mod_outcome,
                              sym, code, rot)
            else:  
                correction = get_syndrome_Steane(mod_outcome, 
                                 error) 

        else:
            correction = get_syndrome(mod_outcome, sym, code, rot)
        
        corrected_state = apply_operation(states_dict[outcome], 
                            correction, sym)
        new_states_dict[outcome] = corrected_state
 
    return new_states_dict



def apply_correction_to_every_state_BS(states_dict, error_detected='Z', sym=False):
    '''
    '''
    new_states_dict = {}

    #print 'Detecting error =', error_detected

    for outcome in states_dict:
        mod_outcome = map(int,outcome.split(' '))
        outcome_stab1, outcome_stab2 = mod_outcome[:3], mod_outcome[3:]
        parity1, parity2 = sum(outcome_stab1)%2, sum(outcome_stab2)%2
        if parity1 == 0:
            if parity2 == 0:  qubit_corr = None
            else:         qubit_corr = 2
        else:
            if parity2 == 0:  qubit_corr = 0
            else:         qubit_corr = 1

        if error_detected == 'Z' and qubit_corr != None: qubit_corr *= 3

        #print 'Outcome =', outcome
        #print 'Qubit =', qubit_corr

        corr_list = ['I' for i in range(9)]
        if qubit_corr != None:  corr_list[qubit_corr] = error_detected
        corr_mat = tensor_product([gate_matrix_dic[gate] for gate in corr_list])
        corr_state = apply_operation(states_dict[outcome], corr_mat, sym)
        new_states_dict[outcome] = corr_state

    return new_states_dict      
    



def apply_correction_to_every_state_cat_ancilla(states_dict, sym=False):
    '''
    only for Steane code with 4-qubit cat ancillae.
    '''
    new_states_dict = {}
    for outcome in states_dict:
        simp_outcome = [(sub_outcome.count('0'))%2 for sub_outcome in outcome.split(' ')]
        correction = get_syndrome(simp_outcome, sym, 'Steane')
        corrected_state = apply_operation(states_dict[outcome], correction, sym)
        print 'Trace =', np.trace(states_dict[outcome])
        new_states_dict[outcome] = corrected_state

    return new_states_dict



def oper_and_eigensystem(list_gate_names_or_oper=['Z']):
    '''
    input: a list of gates, for example ['Z', 'Z', 'Z'] or the operator
    outputs:  (a) the operator corresponding to that list of gates
              (b) the eigvals and eigvecs of that operator 
    '''
    if type(list_gate_names_or_oper) == type([]):
        list_gates = [gate_matrix_dic[gate] 
                      for gate in list_gate_names_or_oper]
        oper = tensor_product(list_gates)
    else:
        oper = list_gate_names_or_oper
    eigvals, eigvecs = sc.linalg.eig(np.array(oper))
    eigensystem = {'vals':eigvals, 'vecs':eigvecs}

    return oper, eigensystem



def generate_rotation_operator(eigvecs, eigvals, angle, mode='eig',
                   gates=['Z']):
    '''
    generates the following gate: exp(-(i/2)*angle*operator)
    if mode == 'eig', then the input is eigvecs and eigvals
    if mode == 'gate', then the input is a gate.
    You can tell that this was added for the purpose (ad hoc)
    of the stabilizer rotations.
    '''
    if mode == 'gate':
        eigensys = oper_and_eigensystem(gates)[1]
        eigvecs = eigensys['vecs']
        eigvals = eigensys['vals']
        
    list_factors = []
    for i in range(len(eigvals)):
        phase = complex(np.cos(angle*eigvals[i]/2.), 
                       -np.sin(angle*eigvals[i]/2.))
        # The eigenvectors are the columns of the matrix.
        # But we need to turn them into numpy column vectors.
        eigvec = np.matrix(eigvecs[:,i]).T
        list_factors += [phase*(eigvec)*(eigvec.H)]

    return sum(list_factors) 
    


def rotate(eigvecs, eigvals, angle, operator):
    '''
    rotates an initial operator (density matrix or stabilizer) by an 
    angle 'angle' and about an axis defined by the eigenvectors and 
    eigenvalues.
    '''
    rotation = generate_rotation_operator(eigvecs, eigvals, angle)
    
    return (rotation*operator)*(rotation.H)



def reflect_XY_plane(X_gate, Z_gate, phi, dens):
    '''
    '''
    operX, eigsystX = oper_and_eigensystem(X_gate)
    dens1 = operX*dens*(operX.H)
    
    operZ, eigsystZ = oper_and_eigensystem(Z_gate)
    dens2 = rotate(eigsystZ['vecs'], eigsystZ['vals'], 2*phi, dens1)
    
    return dens2



def measure_logical_stabilizer(n_data_qs, data_dens, log_oper):
    '''
    measure the logical operator and collapses the data density
    matrix to the +1 eigenspace
    '''
    I = tensor_product([gate_matrix_dic['I'] for i in range(n_data_qs)])
    zero_part = np.kron(I, np.matrix([[1.,0.],[0.,0.]]))
    one_part = np.kron(log_oper, np.matrix([[0.,0.],[0.,1.]]))
    control_log_oper = zero_part + one_part
    
    X_plus = state_vector_dic['PrepareXPlus']
    initial_dens = np.kron(data_dens, X_plus*(X_plus.H))
    middle_dens = (control_log_oper*initial_dens)*(control_log_oper.H)

    meas_list = [gate_matrix_dic['I'] for i in range(n_data_qs + 1)]
    meas_list[n_data_qs] = gate_matrix_dic['MeasureX'][0]
    meas = tensor_product(meas_list)

    final_dens = (meas*middle_dens)*(meas.H)
    dens_no_ancilla = trace_out_ancillary_subsystem(final_dens, n_data_qs, 1)

    return dens_no_ancilla
    


def read_distances_from_json(absolute_filename, output='overlap',
                 strengths=None, kind='old'):
    '''
    If we want to calculate the threshold with a different channel for the
    physical and logical levels, then absolute_filename is actually a list
    with two strings: the first one for the physical and the second one
    for the logical.

    Returns 3 lists: strengths, physical distances, and logical distances
    kind = 'old':  like the original version ('l' or 'lc')
           'perfect': 'lcp'  
           'faulty':  'lcf'
    
    '''
    print 'strengths = ', strengths
    if type(absolute_filename) == type([]):
        phys_filename = absolute_filename[0]
        log_filename = absolute_filename[1]
        json_file_phys = open(phys_filename, 'r')
        distances_dict = json.load(json_file_phys)
        json_file_phys.close()
        json_file_log = open(log_filename, 'r')
        log_distances_dict = json.load(json_file_log)
        json_file_log.close()
    else:
        json_file = open(absolute_filename, 'r')
        distances_dict = json.load(json_file)
        log_distances_dict = distances_dict
        json_file.close()

    if strengths == None:
        strengths = distances_dict.keys()
    else: 
        #strengths = map(str, strengths)
        # str doesn't keep all the decimals; repr does.
        strengths = map(repr, strengths)
        
    strengths_dict = {}
    for strength in strengths:
        strengths_dict[float(strength)] = strength
    strengths_float = map(float, strengths)
    strengths_float.sort()
    phys_dists, log_dists = [], []
    if kind == 'old':
        if output == 'overlap' or output == 'fidelity':   log = 'l'
        elif output == 'distance':   log = 'lc'   
    elif kind == 'perfect':  log = 'lcp'
    elif kind == 'faulty':   log = 'lcf'
    for strength_float in strengths_float:
        strength = strengths_dict[strength_float]
        print strength_float
        print strength
        phys_dists += [distances_dict[strength]['p']]
        log_dists += [log_distances_dict[strength][log]]
    
    return strengths_float, phys_dists, log_dists


def fit_honesties_and_accuracies(fitting_func, x_list, y_list):
    '''
    fitting_func:  the function we want to do the fitting with.
               for example, for a func with quad and tert terms:
               def func(x, b, c):  return b*x**2 + c*x**3
    x_list:        x values
    y_list:        y values
    '''
    popt, pcov = sc.optimize.curve_fit(fitting_func,
                       np.array(x_list),
                       np.array(y_list))
    return popt, pcov


def get_one_error_dens_matrs(log_dens, n_qubits, code='Steane'):
    '''
    Returns a list with the original density matrix and all
    the 1-error density matrices.
    For example, for Steane, this list would have 64 density
    matrices, because we can have an X error on 8 sites (the 
    -1 site is interpreted as no error) and a Z error on 8 
    sites as well. 
    MGA 4/25/14
    '''
    dens_matrs = []
    X, Z = range(-1, n_qubits) , range(-1, n_qubits)
    
    #####  Just added this.  Make it nicer later on.  MGA 5/14/14   ####

    if code == 'TMR':
        for x in X:
            syndrome = ['I' for i in range(n_qubits)]
            if x > -1:
                syndrome[x] = 'X'
        
            oper = tensor_product([gate_matrix_dic[gate] for gate in syndrome])
            dens_matrs += [apply_operation(log_dens, oper)]
    
        return dens_matrs

    #####################################################################

    for x in X:
        for z in Z:
            syndrome = ['I' for i in range(n_qubits)]
            
            # If there's no X error
            if x == -1:
    
                # But there's a Z error
                if z > -1:
                    syndrome[z] = 'Z'
            
            # If there's an X error
            else:
                # But no Z error
                if z == -1:
                    syndrome[x] = 'X' 
        
                # If there's also a Z error
                else:
                    if x == z:
                        syndrome[x] = 'Y'
                    else:
                        syndrome[x] = 'X'
                        syndrome[z] = 'Z' 
            
            oper = tensor_product([gate_matrix_dic[gate] for gate in syndrome])
            dens_matrs += [apply_operation(log_dens, oper)]

    return dens_matrs



def normalize_states(states, norm_factor):
        '''
        states needs to be a dict
        '''
        new_states = {}
        for key in states:
                new_states[key] = norm_factor*states[key]
        return new_states



def translate_maj_vote_key(key, n_stabs=3):
    '''
    Used to translate from say '000001' to 
    out_list1 = ['0','0',None]
    out_list2 = [2]
    '''
    out_list1, out_list2 = [], []
    for i in range(n_stabs):
        if key[i] == key[i+n_stabs]:
            out_list1 += [key[i]]
        else:
            out_list1 += [None]
            out_list2 += [i]
    
    return out_list1, out_list2
    

#################################################################
##  Functions to be used by the new branching methods recently ##
##  added to the class Whole_Circuit.  MGA 04/14/15            ##
##                                                             ##


def add_one_to_branch(branch_id):
    '''
    '''
    total_len = len(branch_id)
    branch_plus_one = bin(int(branch_id, 2) + 1)[2:]
    # add leading zeroes
    leading_zeroes = total_len - len(branch_plus_one)
    return '0'*leading_zeroes + branch_plus_one
        
    

def eliminate_final_zeroes(branch_id):
    '''
    '''
    last_one_index = branch_id.rfind('1')
    # if there are no 1s, return the same string
    if (last_one_index < 0):  
        return branch_id[:]
    # else, return the string up to the last one
    else:
        return branch_id[:last_one_index + 1]



def compute_next_branch(branch_id, elim_branches, n):
    '''
    Nice recursive function to compute the next branch
    '''
    #print 'running next branch...'
    #print 'elim branches =', elim_branches
    #print '1', branch_id
    # This if means we have reached the final branch
    if branch_id.count('0') == 0:
        return '1' + '0'*(n-1)
    branch_id = add_one_to_branch(branch_id)
    #print '2', branch_id
    # if we haven't reached the final branch,
    # then we eliminate the final zeroes.
    branch_id = eliminate_final_zeroes(branch_id)
    #print '3', branch_id
    #t.sleep(10)
    # if the resulting branch is in the list of
    # eliminated branches, we calculate the next
    # branch again
    if branch_id in elim_branches:
        elim_branches.remove(branch_id)
        return compute_next_branch(branch_id,
                       elim_branches,
                       n)
    else:
        return branch_id

##                                                             ##
#################################################################

#################################################################
##  Functions to be used by States_and_Operations to implement ##
##  an error for which we lack a symbolic expression.          ##
##  The idea is to read the error from a json file previously  ##
##  obtained from a minimization.  MGA 12/23/2015.             ##



def read_operation_from_json(channel, error_rate, sym=False, cutoff=1.e-8):
    '''
    '''
    folder = '/home/mau/ApproximateErrors/Simulation/'
    json_filename = folder + channel + '.json'
    try:
        json_file = open(json_filename)
    except IOError:
        print 'Could not find the json file.'
        print json_filename
	sys.exit(0)

    data = json.load(json_file)
    json_file.close()

    try:
        results = data[repr(error_rate)][0]
    except KeyError:
        print 'Could not find that error rate.'
        sys.exit(0)


    sq_root_coefs = {'0': sqrt(1 - sum(results['coefs'].values()))}
    for ch_coef in results['coefs']:
        if results['coefs'][ch_coef] > cutoff:
            sq_root_coefs[ch_coef] = sqrt(results['coefs'][ch_coef]) 

    Kraus_ch = []
    for ch_coef in sq_root_coefs:
        ch = tot_group[int(ch_coef)]
        for oper in ch:
            Kraus_ch += [sq_root_coefs[ch_coef]*oper]
            #print ch_coef
            #print (sq_root_coefs[ch_coef])**2

    return Kraus_ch



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
        
    This function was copied from Script2.py.  Need to organize this.
    '''
    n_stab = len(stabilizers)
    circ = Circuit()
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



