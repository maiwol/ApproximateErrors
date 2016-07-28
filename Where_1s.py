import numpy as np
#c = 1

#for t in range(2,5):
#	for a in range(1,t):
#		print t;		
#		print a;
#		print 2.**t-2*(a)
#		print 'next'
#		print '\n'

circ1 = [[1,0],[0,1]]
circ2 = [[1,0],[0,0]]


##Performing addition of arrays
##w is number of wires

def adding_arrays(w, circ1, circ2):
	circ = [[0 for a in range(2**w)] for b in range(2**w)]
	for a in range(2**w):
		for b in range(2**w):
			circ[a][b] = circ1[a][b] + circ2[a][b]
	return circ


w=2

##print np.outer(circ1,circ2)

