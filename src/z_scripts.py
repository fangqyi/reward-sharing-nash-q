#!/usr/bin/env python3.9

from random import choice

import numpy


def prop(vector):
    return [x / sum(vector) for x in vector]


def distance(a, b):
    ret = numpy.linalg.norm([a - b], ord=2)
    return ret


upper = 6
tasks = range(0, upper, 1)
div = 1
n_agents = 2
# z_q_cp = [1, 2]  # [2, 1]
# z_p_cp = [9, 6]  # [2, 1]
d = {}
d_tuples = {}

m = 20

def get_str(z_p_cp, z_q_cp):
    dist = []
    for giver in range(n_agents):
        dist.append(prop([upper+1e-4 - distance(z_q_cp[giver], z_p_cp[receiver]) for receiver in range(n_agents)]))

    str = "{}0%, {}0%, {}0%, {}0%".format(int(dist[0][0] * 10), int(dist[0][1] * 10), int(dist[1][0] * 10),
                                          int(dist[1][1] * 10))
    return str


def distribute(z_p_cp, z_q_cp):
    dist = []
    for giver in range(n_agents):
        dist.append(prop([upper+1e-4 - distance(z_q_cp[giver], z_p_cp[receiver]) for receiver in range(n_agents)]))

    for receiver in range(n_agents):
        for giver in range(n_agents):
            '''
            print("giver agent {} to receiver agent {} with distance {}: {}".format(giver, receiver,
                                                                                    - distance(z_q_cp[giver],
                                                                                               z_p_cp[receiver]),
                                                                                    dist[giver][receiver]))
                                                                                    '''
    str = get_str(z_p_cp, z_q_cp)
    if str in d:
        d[str] += 1
    else:
        d[str] = 1


all_s = []

def dis_print(z_p_cp, z_q_cp):

    dist = []
    for giver in range(n_agents):
        dist.append(prop([upper+1e-4- distance(z_q_cp[giver], z_p_cp[receiver]) for receiver in range(n_agents)]))

    for receiver in range(n_agents):
        for giver in range(n_agents):
            print("giver agent {} to receiver agent {} with distance {}: {}".format(giver, receiver, upper+1e-4 - distance(z_q_cp[giver], z_p_cp[receiver]), dist[giver][receiver]))


for i1 in tasks:
    for i2 in tasks:
        for i3 in [0]:
            for i4 in tasks:
                all_s.append(([i1 / div, i2 / div],[i3 / div, i4 / div]))
                # print("{} {} {} {}".format(i1/div, i2/div, i3/div, i4/div))
                distribute([i1 / div, i2 / div], [i3 / div, i4 / div])


# for x in sorted(d):
#     print("{}: {}".format(x, d[x]))
ret = []
def get_m_schemes(m,):
    dct = {}
    schemes = []
    while len(schemes) != m:
        x = choice(all_s)

        if get_str(x[0], x[1]) not in dct and x not in schemes:
            dct[get_str(x[0], x[1])] = 1
            schemes.append(x)
        elif len(dct) == len(d.keys()):
            schemes.append(x)

    ret = schemes
    return ret, ([x[0] for x in schemes], [x[1] for x in schemes])


ret, tasks = get_m_schemes(50)
print(tasks)
print(len(tasks[0]))
for x in ret:
    dis_print(x[0], x[1])

d = {}
for x in ret:
    distribute(x[0], x[1])
print(d)
