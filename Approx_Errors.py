"""
Methods for the approximations
This file contains a collection of definitions and methods used
for the approximation of general error channels. 
Mauricio Gutierrez, Feb 2012. 

Notice: Depending on the sympy version you are using, the Matrix
object might be in sympy or in sympy.matrices.  This means you
might have to manually change all the instances where the string
'mat.Matrix' appears to 'sp.Matrix' or vice-versa.  In vim, this
is really easy to do.  
To replace 'mat.Matrix' for 'sp.Matrix':
:%s/mat.Matrix/sp.Matrix/g  
To replace 'sp.Matrix' for 'mat.Matrix':
:%s/sp.Matrix/mat.Matrix/g

"""

import numpy as np
import random as rd
import copy
from math import sqrt, pi, cos, sin
import sympy as sp
#import sympy.matrices as mat
import sympy.functions as fun
from sympy.solvers import solve
from scipy.optimize import fmin_slsqp



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
Ysym = sp.Matrix([[0.,-sp.I],[sp.I,0.]])
Zsym = sp.Matrix([[1.,0.],[0.,-1.]])
Hsym = sp.sqrt(1./2.)*sp.Matrix([[1.,1.], [1.,-1.]])
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



"""1-qubit Clifford group"""
Cliff = [[I,S,S*S,S*S*S],[H,H*S,H*S*S,H*S*S*S],[S*H,S*H*S,S*H*S*S,S*H*S*S*S],[H*S*S*H,H*S*S*H*S,H*S*S*H*S*S,H*S*S*H*S*S*S],[S*S*H,S*S*H*S,S*S*H*S*S,S*S*H*S*S*S],[Sinv*H,Sinv*H*S,Sinv*H*S*S,Sinv*H*S*S*S]]




"""Pauli measurements"""
meas_Zp = [np.matrix([[1.,0.],[0.,0.]]), np.matrix([[0.,1.],[0.,0.]])]  # translation to |0>
meas_Xp = [H*A*H for A in meas_Zp]					# translation to |+>
meas_Yp = [S*A*Sinv for A in meas_Xp]					# translation to |+i>
meas_Zn = [H*Z*A*Z*H for A in meas_Xp]					# translation to |1>
meas_Xn = [Z*A*Z for A in meas_Xp]					# translation to |->
meas_Yn = [Sinv*A*S for A in meas_Xp]					# translation to |-i>



def change_matrix_type(matrix, num_to_sym=1):
	"""
	Change a matrix or array from numpy to sympy and a sympy matrix
	to numpy matrix.
	"""
	if num_to_sym:
		n = len(matrix)
		new_matrix = sp.Matrix([[matrix[i,j] for j in range(n)] for i in range(n)])
	else:
		n = int(sqrt(len(matrix)))
		new_matrix = np.matrix([[complex(fun.re(matrix[i,j]), fun.im(matrix[i,j])) for j in range(n)] for i in range(n)])	
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
	This returns a list of the symbolic coefficients associated with each operator in
	the approximation channel.  Each coefficient will be an independent paramater in the
	distance minimization.  
	-group refers to the matrices that we want to include in our approximation channel.
	For example, if we want to use the Pauli group, then group = [0,2,12,14]
	because in the list of matrices, 0 = I, 2 = Z, 12 = X, 14 = Y.
	-cons refers to the constraints that we want to apply to the coefficients.  It should
	be a list of 2-entry lists, each one specifying which coefficients have the same value.
	For example, if want to use the Pauli group with the constraint that X and Y have the
	same vaule, then group = [0,2,12,14] and cons = [[12,14]]
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



def Kraus_group(group=range(30), cons=[]):
	"""Kraus_group is the Kraus-representation list of the operators 
	in our model, each one multiplied by its corresponding symbolic coefficient.
	Each operator is a sympy matrix.  This list is used to calculate the worst-case
	fidelity, which requires the channel to be expressed in the Kraus representation.
	We can choose any given subset of the total group.
	"""
	coef = coefficients(group, cons)
	Kraus = []
	for i,elem in enumerate(group):
		Kraus += [sp.sqrt(coef[i])*change_matrix_type(operator) for operator in tot_group[elem]]
	return Kraus 



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


def final_dens_matrix(channel, initial_dens_matrix="sym"):
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
		if (isinstance(initial_dens_matrix, np.matrix) or isinstance(initial_dens_matrix, np.ndarray)):
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
				dens_final += (Norm_Pauli_sym[i]*dens)*(channel[i,j]*(Norm_Pauli_sym[j].H)) 
	return dens_final
	


def fixed_point(channel):
	"""Calculates the fixed point of a 1-qubit channel"""
	rho = final_dens_matrix(channel)
	final_point = [fun.re((Xsym*rho).trace().expand()), fun.re((Ysym*rho).trace().expand()), fun.re((Zsym*rho).trace().expand())]
	final_point[0] -= x
	final_point[1] -= y
	final_point[2] -= z
	return solve(final_point)
	


def trace_distance(initial_density_matrix, target_process, approx, cons=[]):
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
	Dist = 0.5*sum([Eig[i][1]*sp.Abs(Eig[i][0]) for i in range(len(Eig))])
	return Dist




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



"""Symbolic expression for the fidelity when the initial state of the
system is a general density matrix.  The output of this function is used
for the later minimization to find the worst-case fidelity.
The symbols are defined at the beginning; see Basic Definitions.
"""
#def symb_fid(targ_unit, err_chan_ops):
#	dens_mat = 0.5*Isymb + 0.5*(x*Xsymb + y*Ysymb + z*Zsymb)
#	return fid(targ_unit, err_chan_ops, dens_mat, symbolic=1)


#def ave_fid(proc1,proc2):
#	return ((0.25)*((proc1*proc2).trace())).expand()



def ave_fidcons(target, approx=range(30), cons=[]):
	"""Constraint: The minimization is performed forcing this expression to be > 0."""
	Approx = proc_mat_group(approx, cons)
	if isinstance(target, list):
		target_process = process_matrix(target)
	elif isinstance(target, sp.Matrix):
		target_process = target

	cons = target_process[0] - Approx[0]  
	return cons				

# This is quicker. Notice that this constraint would be different if we were not 
# comparing the channels to the Identity.



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
	

