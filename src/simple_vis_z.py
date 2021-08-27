#!/usr/bin/python

import sys


# arg: dir, max_range
dir = sys.argv[1]
max_range = sys.argv[2]
z_p_0 = []
z_p_1 = []
z_q_0 = []
z_q_1 = []

for i in range(1, int(max_range)):
    path = dir+"/"+str(i)+"0050/"
    f = open(path+"z_p/tensors.tsv", "r")
    z_p_0.append(float(f.readline()[:-2]))
    z_p_1.append(float(f.readline()[:-2]))
    f = open(path+"z_q/tensors.tsv", "r")
    z_q_0.append(float(f.readline()[:-2]))
    z_q_1.append(float(f.readline()[:-2]))

for idx in range(len(z_p_1)):
    print("agent (z_p, z_q): ({}, {}), ({}, {})".format(z_p_0, z_q_0, z_p_1, z_q_1))


