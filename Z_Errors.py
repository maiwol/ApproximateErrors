import numpy as np
w = 2
x = 2





def Z_Circuit(w, z):
	circ_mat = [[0 for y in range(2**z)] for x in range(2**z)]
	circ_mat2 = [[0 for y in range(2**w)] for x in range(2**w)]
	for b in range(2**(z-1)):
		i = 2*b
		j= 2*b + 1
		circ_mat[i][i] = 1
		circ_mat[j][j] = -1	
	if w==z:
		print circ_mat
	else:
		for c in range(2**(z)):
			for d in range(2**(z)):	
				for a in range(2**(w-z)):
					if circ_mat[d][c] == 1 or \
					circ_mat[d][c] == -1:
						circ_mat2[2*d+a][2*c+a] = circ_mat[d][c]
		print circ_mat2
w=3
z=2

Z_Error_Circuit(w,z);


##Defining identity
