import numpy as np
w = 2
x = 2





def X_Circuit(w, x):
	circ_mat = [[0 for y in range(2**x)] for z in range(2**x)]
	circ_mat2 = [[0 for y in range(2**w)] for z in range(2**w)]
	for b in range(2**(x-1)):
		i = 2*b
		j= 2*b +1
		circ_mat[i][j] = 1
		circ_mat[j][i] = 1	
	if w==x:
		print circ_mat
	else:
		for c in range(2**(x)):
			for d in range(2**(x)):	
				for a in range(2**(w-x)):
					if circ_mat[d][c] == 1:
						circ_mat2[2*d+a][2*c+a] = 1
		print circ_mat2
w=1
x=1
X_Error_Circuit(w,x);


##Defining identity
