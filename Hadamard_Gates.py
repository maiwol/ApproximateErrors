import numpy as np
import Where_1s as w1
w = 2
h = 2





def Hada_Circuit(w, h):
	circ_mat = [[0 for y in range(2**h)] for x in range(2**h)]
	circ_mat2 = [[0 for y in range(2**w)] for x in range(2**w)]
	for b in range(2**(h-1)):
		i = 2*b
		j= 2*b +1
		circ_mat[i][j] = (1/np.sqrt(2))
		circ_mat[j][i] = (1/np.sqrt(2))
		circ_mat[i][i] = (1/np.sqrt(2))
		circ_mat[j][j] = -(1/np.sqrt(2))	
	if w==h:
		print circ_mat
	else:
		for c in range(2**(h)):
			for d in range(2**(h)):	
				for a in range(2**(w-h)):
					if circ_mat[d][c] == (1/np.sqrt(2)) or \
					circ_mat[d][c] == -(1/np.sqrt(2)):
						circ_mat2[2*d+a][2*c+a] = circ_mat[d][c]
		print circ_mat2
#w=2
#h=1
#Hada_Circuit(w,h);


##Defining identity
