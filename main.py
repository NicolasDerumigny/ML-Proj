#!/usr/bin/env python3
import numpy as np
import math
from DSEBM import *
from parsing import *
from random import randint

import numpy as np
import tensorflow as tf


NUM_EPOCHS=100
NUM_EXAMPLES=10


def main():
	training, data, answer = load()
	# The number of parameter is 49
	print("Loaded !")
	machine=FC_DSEBM([49,5], num_epochs=NUM_EPOCHS)
	machine.fit(training)
	print("Trained !")

	newdata = []
	newresult = []
	outliers = 0
	for i in range(NUM_EXAMPLES):
		newindex = randint(0,len(data)-1)
		newdata += [data[newindex]]
		newresult += [answer[newindex]]
		if newresult[i] == 1:
			outliers += 1


	score = machine.score(newdata, newresult, 10e+4)
	print("Score: {} (ouliers ratio: {})".format(score, outliers/NUM_EXAMPLES))
	machine.delete()


if __name__ == '__main__':
	main()