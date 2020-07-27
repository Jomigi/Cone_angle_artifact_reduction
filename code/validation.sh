#!/bin/bash

# Copyright 2019 Centrum Wiskunde & Informatica, Amsterdam

#-----------------------------------------------------------------
# Author: Jordi Minnema
# Contact: jordi@cwi.nl
# Github: https://github.com/Jomigi/CBCT-artifact-reduction
# License: t.b.d. 

# This script is intended to find optimal training parameters 
# for cone-angle artifact reduction in walnut CBCT scans. 
#----------------------------------------------------------------


# File name of the script that is run:
script='main.py'

# Parameters that will be used for validation:
DEPTH=(10 30 50 70 80 90 100)
DILATIONS=(1,2,3,4,5,6,7,8,9,10 1,2,4,8,16)
POS=(1 2 3)
EPOCHS=200
EARLYSTOP=10

# Train network for all combinations of parameters
for pos in ${POS[*]}; do
	for dil in ${DILATIONS[*]}; do
		for dep in ${DEPTH[*]}; do
			echo Running python script with depth=$dep dilations=$dil position=$pos epochs=$EPOCHS early_stop=$EARLYSTOP
			python $script with depth=$dep dilations=$dil position=$pos epochs=$EPOCHS early_stop=$EARLYSTOP
		done
	done
done
