#!/usr/bin/env python3
import numpy as np
import math
from DSEBM import *

import numpy as np
import tensorflow as tf

#  #################### #
#   Flags definition   #
# #################### #
flags = tf.app.flags
FLAGS = flags.FLAGS


def main():
	machine=FC_DSEBM([2,5,7,10])
	machine.fit([(1,2),(2,2),(3,2)])
if __name__ == '__main__':
	main()