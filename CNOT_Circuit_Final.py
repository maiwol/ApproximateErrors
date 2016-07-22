import Where_1s as w1

def CNOT_Circuit(w,c,t):
## WIRES START AT 1:
## w is # of wires
## c is location of control
## t is location of target   		
	circ_mat0 = [[0 for a in range(2**(c))]for b in range(2**(c))]
	circ_mat02 = [[0 for a in range(2**w)] for b in range(2**w)]
	
	if c<t:
		circ_mat = [[0 for a in range(2**c)] for b in range(2**c)]
		circ_mat2 = [[0 for a in range(2**(t-1))]for b in range(2**(t-1))]
		circ_mat3 = [[0 for a in range(2**(t))]for b in range(2**(t))]
		circ_mat4 = [[0 for a in range(2**w)]for b in range(2**w)]
##Dealing with the case where control is a 1
##Step 1: Locate where the control is/falls in the matrix		
		for a in range(2**(c-1)):
			i = 1 + 2*a
			circ_mat[i][i] = 1
##Step 2: Add identities proportional to the amount of gates b/t c and t
		for a in range(2**(c-1)):			
			for d in range(2**(t-c-1)):
				j=2**(t-c-1)*(1+2*a)+d					
				circ_mat2[j][j] = 1
##Step 3: Add X Gates at the target
		for a in range(2**(c-1)):
			for b in range(2**(t-c-1)):
				i = 2*(2**(t-c-1)*(1+2*a)+b)		
				j = 2*(2**(t-c-1)*(1+2*a)+b)+1	
				circ_mat3[i][j] = 1
				circ_mat3[j][i] = 1
##Step 4: Tensoring with the rest of the Identities
		for a in range(2**(c-1)):	
			for b in range(2**(t-c-1)):			
				for d in range(2**(w-t)):
					i=2**(w-t)*(2**(t-c)*(1+2*a) + 2*b) + d 	
					j=2**(w-t)*(2**(t-c)*(1+2*a) + 2*b+1) + d
					circ_mat4[i][j] = 1
					circ_mat4[j][i] = 1	
		
#		print circ_mat4

## Handling the case where the control is a zero	
		for a in range(2**(c-1)):
			i = 2*a
			circ_mat0[i][i] = 1

		for a in range(2**(c-1)):
			for b in range(2**(w-c)):
				i = (2**(w-c))*(2*a) + b
				circ_mat02[i][i] = 1
#		print circ_mat02
		print w1.adding_arrays(w, circ_mat4, circ_mat02)

	else:
		circ_mat = [[0 for a in range(2**t)] for b in range(2**t)]
		circ_mat2 = [[0 for a in range(2**(c-1))] for b in range(2**(c-1))]
		circ_mat3 = [[0 for a in range(2**(c))] for b in range(2**(c))]
		circ_mat4 = [[0 for a in range(2**w)] for b in range(2**w)]
##Locate where X gate falls
		for a in range(2**(t-1)):
			i = (2*a)
			j = i+1
			circ_mat[i][j] = 1
			circ_mat[j][i] = 1
		print circ_mat
##Place the identities b/t target and control
		for a in range(2**(t-1)):
			for b in range(2**(c-t-1)):
				i = 2**(c-t-1)*2*a + b
				j = 2**(c-t-1)*(1+2*a) + b
				circ_mat2[i][j] = 1
				circ_mat2[j][i] = 1
		print circ_mat2
		print '\n'
##Place the One
		for a in range(2**(t-1)):
			for b in range(2**(c-t-1)):
				for d in range(c-t):
					i = 2*(2**(c-t-1)*2*a + b) + 1
					j = 2*(2**(c-t-1)*(1+2*a) + b) + 1 
					print a
					print b
#					print c
					print i
					print j
					print '\n'
					circ_mat3[i][j] = 1
					circ_mat3[j][i] = 1
		print '\n'	
		print circ_mat3
##Place the remaining Identities
		for a in range(2**(t-1)):
			for b in range(2**(c-t-1)):
				for e in range(2**(w-c)):
					i = 2**(w-c)*(2*(2**(c-t-1)*2*a + b) + 1) + e
					j = 2**(w-c)*(2*(2**(c-t-1)*(1+2*a) + b) + 1) + e 
					circ_mat4[i][j] = 1
					circ_mat4[j][i] = 1
		return circ_mat4
##Handling the case where control is a 0
		for a in range(2**(t)):
			i = 2*a
			print a
			print i
			circ_mat0[i][i] = 1
##		print circ_mat
		for a in range(2**(t)):
			for b in range(2**(w-c)):
				i = (2**(w-c))*(2*a) + b
				circ_mat02[i][i] = 1
		return circ_mat02
		print w1.adding_arrays(w, circ_mat4, circ_mat02)	

w=2
c=1
t=2

CNOT_Circuit(w,c,t)
		
		


##Place the case where the control is a zero

def control_zero(w,c,t):
	if c<t:	
		circ_mat0 = [[0 for a in range(2**(c))]for b in range(2**(c))]
		circ_mat02 = [[0 for a in range(2**w)] for b in range(2**w)]
		for a in range(2**(c-1)):
			i = 2*a
			print a
			print i
			circ_mat0[i][i] = 1
##		print circ_mat
		for a in range(2**(c-1)):
			for b in range(2**(w-c)):
				i = (2**(w-c))*(2*a) + b
				circ_mat02[i][i] = 1
		print circ_mat02	
	else:
		circ_mat0 = [[0 for a in range(2**(c))]for b in range(2**(c))]
		circ_mat02 = [[0 for a in range(2**w)] for b in range(2**w)]
		for a in range(2**(t)):
			i = 2*a
			print a
			print i
			circ_mat0[i][i] = 1
##		print circ_mat
		for a in range(2**(t)):
			for b in range(2**(w-c)):
				i = (2**(w-c))*(2*a) + b
				circ_mat02[i][i] = 1
		print circ_mat02	









#CNOT_Circuit(w,c,t)


def CNOT_t(w,c,t):
	circ_mat = [[0 for a in range(2**t)] for b in range(2**t)]
	circ_mat2 = [[0 for a in range(2**(c-1))] for b in range(2**(c-1))]
	circ_mat3 = [[0 for a in range(2**(c))] for b in range(2**(c))]
	circ_mat4 = [[0 for a in range(2**w)] for b in range(2**w)]
##Locate where X gate falls
	for a in range(2**(t-1)):
		i = (2*a)
		j = i+1
		circ_mat[i][i] = 1
		circ_mat[j][j] = -1
	print circ_mat
##Place the identities b/t target and control
	for a in range(2**(t-1)):
		for b in range(2**(c-t-1)):
			i = 2**(c-t-1)*2*a + b
			j = 2**(c-t-1)*(1+2*a) + b
			circ_mat2[i][j] = 1
			circ_mat2[j][i] = 1
	print circ_mat2
	print '\n'
##Place the One
	for a in range(2**(t-1)):
		for b in range(2**(c-t-1)):
			for d in range(c-t):
				i = 2*(2**(c-t-1)*2*a + b) + 1
				j = 2*(2**(c-t-1)*(1+2*a) + b) + 1 
				print a
				print b
#				print c
				print i
				print j
				print '\n'
				circ_mat3[i][j] = 1
				circ_mat3[j][i] = 1
	print '\n'	
	print circ_mat3
##Place the remaining Identities
	for a in range(2**(t-1)):
		for b in range(2**(c-t-1)):
			for e in range(2**(w-c)):
				i = 2**(w-c)*(2*(2**(c-t-1)*2*a + b) + 1) + e
				j = 2**(w-c)*(2*(2**(c-t-1)*(1+2*a) + b) + 1) + e 
				circ_mat4[i][j] = 1
				circ_mat4[j][i] = 1
	print '\n'
	print circ_mat4
w=2
c=2
t=1

##CNOT_t(w,c,t)









