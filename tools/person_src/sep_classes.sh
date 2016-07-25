#!/bin/bash

#store class 0 elements
grep ' 0' assign/patch_test.txt assign/patch_train.txt | sed 's/.*://' > assign/background.txt
grep ' 1' assign/patch_test.txt assign/patch_train.txt | sed 's/.*://' > assign/person.txt
grep ' 2' assign/patch_test.txt assign/patch_train.txt | sed 's/.*://' > assign/car.txt
grep ' 3' assign/patch_test.txt assign/patch_train.txt | sed 's/.*://' > assign/random.txt
