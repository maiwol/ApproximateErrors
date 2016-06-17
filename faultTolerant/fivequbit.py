import sys
import os
from circuit import *


class Code:

	stabilizer = [
                       ['X','Z','Z','X','I'],
                       ['I','X','Z','Z','X'],
                       ['X','I','X','Z','Z'],
                       ['Z','X','I','X','Z']]

	logical = {'X': ['X','X','X','X','X'],
		   'Z': ['Z','Z','Z','Z','Z']}
