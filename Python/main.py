import random
from numpy import array, zeros, inf, argmin, argsort, ceil, append, mean
from numpy.random import rand, permutation, seed
from ypstruct import struct
from plotting import *
from cec2021 import *


# Select random seed
random.seed(10), seed(10)

# Number of population
NP = 40

# Number of dimensions select from [2, 10, 20, 30, 50, 100]
# for F8, F9 and F10 must select from [10, 20, 30, 50, 100]
# to plot search space (Except F8, F9 and F10) ND must be 2
ND = 10

# Number of iterations
NI = 100

# Selected CEC 2021 function (F1 to F10)		
F = F3(nd=ND)

# Function's optimal
opt = F.opt

# lower bound
LB = -100 * np.ones(ND)

# upper bound
UB = 100 * np.ones(ND)

# Create Initial Population
init = zeros((NP, ND))
for j in range(NP):
	init[j, :] = LB + rand(1, ND) * (UB - LB)

"""
define types of group as [PF, LF, a, b, m]
PF: Experimental’s Power Factor
Lf: Leadership’s Power Factor
 a: Experimental’s Power Incremental Coefficien
 b: Leadership’s Power Incremental Coefficien
 m: Maximum Power Modify Coefficien
"""
types = array([
	[2, 2, 0.9, 0.9, 0.1],
	[10, 2, 0.2, 0.9, 0.3],
	[2, 10, 0.9, 0.2, 0.3],
	[2, 12, 0.9, 0.9, 0.01]
])

# Relative population of each group
G = array([0.25, 0.25, 0.25, 0.25])

# Number of members of each type
NG = array(G * NP, dtype=int)
NG[3] = NP - sum(NG[0:3]) 

# Weights of Groups
W = [1000, 1000, 1000, 1000]

# Mutation Factor 
MF = 0.2

# Number of times best cost is repeated
o = 0

# Number of times leader motivated
t = 0

# History of solutions
HS = zeros((NI*NP, ND))

# History of best costs 
HC = zeros(NI)

# History of mean best cost
HM = zeros(NI)

# Define pop matrix
pop = []

# Populate the pop matrix with the structure for each member (Polar fox leash generation)
for i in range(NP):
	pop.append(struct())
	pop[i].s = init[i, :]				# solution
	pop[i].c = F.evaluate(init[i, :])	# cost

# Grouping polar fox
k = 0
for i in range(len(G)):
	for j in range(NG[i]):
		pop[k].PF = types[i, 0]
		pop[k].LF = types[i, 1]
		pop[k].a = types[i, 2]
		pop[k].b = types[i, 3]
		pop[k].m = types[i, 4]
		pop[k].t = i
		k += 1

# Index of best cost
index = argmin([pop[i].c for i in range(NP)])

# Best cost
bc = pop[index].c

# Define leader as best solution
L = pop[index].s

# Execute the algorithm by the number of iterations
for I in range(NI):

	# Experience-based phase
	for i in range(NP):
		f = inf
		NPF = pop[i].PF
		while f > pop[i].c:
			p = rand(1) * NPF
			d = 1 - 2 * rand(ND)
			x = pop[i].s + p * d
			for j in range(ND):
				if x[j] > UB[j]:
					x[j] = UB[j]
				elif x[j] < LB[j]:
					x[j] = LB[j]
			f = F.evaluate(x)
			NPF = NPF * pop[i].a
			if NPF < pop[i].m * pop[i].PF:
				f = inf
				break
		if f != inf:
			pop[i].s = x
			pop[i].c = f

	# Leader-based phase
	for i in range(NP):
		f = inf
		NLF = pop[i].LF
		while f > pop[i].c:
			x = pop[i].s + (1 - 2 * rand(ND)) * (pop[i].s - L) * NLF
			for j in range(ND):
				if x[j] > UB[j]:
					x[j] = UB[j]
				elif x[j] < LB[j]:
					x[j] = LB[j]
			f = F.evaluate(x)
			NLF = NLF * pop[i].b
			if NLF < pop[i].m * pop[i].LF:
				f = inf
				break
		if f != inf:
			pop[i].s = x
			pop[i].c = f

	# Index of sorted costs 
	index = np.flip(argsort([pop[i].c for i in range(NP)]))

	# Sort pop base on index
	pop = [pop[i] for i in index]

	# Check the best cost change
	if bc - pop[-1].c < 1e-8:
		o += 1
	else:
		o = 0

	"""
	IF number of times best cost is repeated was more than 50 and
	the leader's motivation number was less than 3 or 20 percent of
	the end of the iteration, change value of [PF, LF, a, b, m] for
	some of polar fox and number of mutation (NM) equal to the
	number of all members except the leader. otherwise number of
	mutation (NM) It is	equal to the product of the
	Mutation Factor (MF) multiplied by the Number of Population (NP)
	"""
	if ((o > 50 and t < 3) or I == 0.8 * NI):
		NM = NP - 1
		o = 0
		t += 1
		if t < 3:
			types[0, 2] = 0.99
			types[0, 3] = 0.99
			types[1, 3] = 0.99
			types[2, 2] = 0.99
			types[3, 4] = 0.001
	else:
		NM = int(MF * NP)

	# Mutation phase
	for i in range(NM):
		pop[i].s = LB + rand(ND) * (UB - LB)
		pop[i].c = F.evaluate(pop[i].s)

	# Leader’s motivation phase
	R = permutation(NP - 1)
	for i in range(NP - 1):
		n = R[i]

		# Guarantee the presence of at least one type in the population
		if i < 4:
			k = i

		# For other change types randomly base on weights of groups (W)
		else:
			k = random.choices([0, 1, 2, 3], weights=W, k=1)[0]
		pop[n].PF = types[k, 0]
		pop[n].LF = types[k, 1]
		pop[n].a = types[k, 2]
		pop[n].b = types[k, 3]
		pop[n].m = types[k, 4]
		pop[n].t = k

	# Type of group members matrix
	T = [pop[i].t for i in range(NP)]

	# Number of members of each type
	NG = [T.count(0), T.count(1), T.count(2), T.count(3)]

	# Fatigue simulation
	for i in range(len(G)):

		# If type's count less than 5, they should try harder (max energy)
		if NG[i] < 5:
			if i == 0:
				types[0, 2] = 0.99
				types[0, 3] = 0.99
			elif i == 1:
				types[1, 3] = 0.99
			elif i == 2:
				types[2, 2] = 0.99
			else:
				types[3, 4] = 0.001

		# Else, they lose energy at every stage
		else:
			if i == 0:
				types[0, 2] = max(types[0, 2] - 0.001, 0.9)
				types[0, 3] = max(types[0, 3] - 0.001, 0.9)
			elif i == 1:
				types[1, 3] = max(types[1, 3] - 0.001, 0.9)
			elif i == 2:
				types[2, 2] = max(types[2, 2] - 0.001, 0.9)
			else:
				types[3, 4] = min(types[3, 4] + 0.0001, 0.01)

	# Find min cost index
	index = argmin([pop[i].c for i in range(NP)])

	# Set best cost base on index
	best = pop[index].c

	"""
	If the best cost is less than the previous value then update
		- Weights of groups
		- Best cost
		- Best solution
		- Leader
	"""
	if best < bc:
		W[pop[index].t] = W[pop[index].t] + I ** 2 / NG[pop[index].t]
		bc = best
		L = pop[index].s

	# Add solutions and costs to history
	HS[(I)*NP:(I+1)*NP,:] = [pop[i].s for i in range(NP)]
	HC[I] = bc
	HM[I] = mean(HC[:I+1])

	# print result
	p = (I+1) * 100 / NI
	d = ((bc - opt) / opt) * 100
	print("# "+'{:3.1f}'.format(p)+"%", '| Best:', '{:.4f}'.format(bc), '| Delta:', '{:.2f}'.format(d)+"%")


plotProg(HC, HM, opt)
# Plot 2D and 3D search space
if ND == 2:
	best = append(L, bc)
	plotSearchSpace(func=F, hs=HS, sol=F.sol, best=best)