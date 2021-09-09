#!/usr/bin/python
import math
import sys
import numpy

# arg: dir, max_range

def softmax(vector):
    e = [math.exp(x) for x in vector]
    return [x / sum(e) for x in e]


def distance(a, b):
    ret = numpy.linalg.norm([a - b], ord=2)
    return ret

dir = sys.argv[1]
max_range = sys.argv[2]
num_agents = 2
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
    print("agent (z_p, z_q): ({}, {}), ({}, {})".format(z_p_0[idx], z_q_0[idx], z_p_1[idx], z_q_1[idx]))

    dist = []
    agent_0_gives = softmax([- distance(z_q_0[idx], [z_p_0[idx]]), -distance(z_q_0[idx], z_p_1[idx])])
    print("agent 0 gives {} to agent 0 and {} to agent 1".format(agent_0_gives[0], agent_0_gives[1]))
    agent_1_gives = softmax([- distance(z_q_1[idx], z_p_0[idx]), -distance(z_q_0[idx], z_p_1[idx])])
    print("agent 1 gives {} to agent 0 and {} to agent 1".format(agent_1_gives[0], agent_1_gives[1]))



