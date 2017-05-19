#!/usr/bin/env python3
import numpy as np
import math
from DSEBM import *
from parsing import *

import numpy as np
import tensorflow as tf

#  #################### #
#   Flags definition   #
# #################### #
flags = tf.app.flags
FLAGS = flags.FLAGS


def main():
	data, answer = load()
	# The number of parameter is 49
	print("Loaded !")
	machine=FC_DSEBM([49,5,7,10])
	machine.fit(data)


if __name__ == '__main__':
	main()