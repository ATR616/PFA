import os
import pandas as pd
import numpy as np
import functions as func


def shift_data(num, nd):
    data = pd.read_csv(f"data_2021/shift_data_{num}.txt", delimiter='\s+', index_col=False, header=None)
    return data.values.reshape((-1))[:nd]

def matrix_data(num, nd):
    data = pd.read_csv(f"data_2021/M_{num}_D{nd}.txt", delimiter='\s+', index_col=False, header=None)
    return data.values

def shuffle_data(num, nd):
    data = pd.read_csv(f"data_2021/shuffle_data_{num}_D{nd}.txt", delimiter='\s+', index_col=False, header=None)
    return (data.values.reshape((-1)) - 1).astype(int)

def checkND(nd, m=False):
    if m and not(nd in [10, 20, 30, 50, 100]):
        print('Error: The ND value must be selected from [10, 20, 30, 50, 100]!')
        quit()
    elif not(nd in [2, 10, 20, 30, 50, 100]):
        print('Error: The ND value must be selected from [2, 10, 20, 30, 50, 100]!')
        quit()


# Shifted and Rotated Bent Cigar Function (F1 CEC-2017)
class F1:
    def __init__(self, nd=2):
        checkND(nd)
        self.shift = shift_data(1, nd)
        self.matrix = matrix_data(1, nd)
        self.opt = 100.
        self.sol = self.shift

    def evaluate(self, x, *args):
        z = np.dot(self.matrix, x - self.shift)
        return func.bent_cigar_func(z) + self.opt


# Shifted and Rotated Schwefel’s Function (F11 CEC-2014)
class F2:
    def __init__(self, nd=2):
        checkND(nd)
        self.shift = shift_data(2, nd)
        self.matrix = matrix_data(2, nd)
        self.opt = 1100.
        self.sol = self.shift

    def evaluate(self, x, *args):
        z = np.dot(self.matrix, 1000.*(x - self.shift)/100)
        return func.modified_schwefel_func(z) + self.opt


# Shifted and Rotated Lunacek bi-Rastrigin Function (F7 CEC-2017)
class F3:
    def __init__(self, nd=2):
        checkND(nd)
        self.shift = shift_data(3, nd)
        self.matrix = matrix_data(3, nd)
        self.opt = 700.
        self.sol = self.shift

    def evaluate(self, x, *args):
        z = np.dot(self.matrix, 600.*(x - self.shift)/100)
        return func.lunacek_bi_rastrigin_func(z) + self.opt


# Expanded Rosenbrock’s plus Griewangk’s Function (F15 CEC-2014)
class F4:
    def __init__(self, nd=2):
        checkND(nd)
        self.shift = shift_data(4, nd)
        self.matrix = matrix_data(4, nd)
        self.opt = 1900.
        self.sol = self.shift

    def evaluate(self, x, *args):
        z = np.dot(self.matrix, 5. * (x - self.shift) / 100) + 1
        return func.expanded_griewank_rosenbrock_func(z) + self.opt


# Hybrid Function 1 (F17 CEC-2014)
class F5:
    def __init__(self, nd=2):
        checkND(nd)
        self.shift = shift_data(5, nd)
        self.matrix = matrix_data(5, nd)
        self.shuffle = shuffle_data(5, nd)
        self.opt = 1700.
        self.sol = self.shift
        self.n1 = int(np.ceil(0.3 * nd))
        self.n2 = int(np.ceil(0.3 * nd)) + self.n1
        self.idx1, self.idx2, self.idx3 = self.shuffle[:self.n1], self.shuffle[self.n1:self.n2], self.shuffle[self.n2:nd]
        self.g1 = func.modified_schwefel_func
        self.g2 = func.rastrigin_func
        self.g3 = func.elliptic_func

    def evaluate(self, x, *args):
        z = x - self.shift
        z1 = np.concatenate((z[self.idx1], z[self.idx2], z[self.idx3]))
        mz = np.dot(self.matrix, z1)
        return self.g1(mz[:self.n1]) + self.g2(mz[self.n1:self.n2]) + self.g3(mz[self.n2:]) + self.opt


# Hybrid Function 2 (F15 CEC-2017)
class F6:
    def __init__(self, nd=2):
        checkND(nd)
        self.shift = shift_data(6, nd)
        self.matrix = matrix_data(6, nd)
        self.shuffle = shuffle_data(6, nd)
        self.opt = 1600.
        self.sol = self.shift
        self.n1 = int(np.ceil(0.2 * nd))
        self.n2 = int(np.ceil(0.2 * nd)) + self.n1
        self.n3 = int(np.ceil(0.3 * nd)) + self.n2
        self.idx1, self.idx2 = self.shuffle[:self.n1], self.shuffle[self.n1:self.n2]
        self.idx3, self.idx4 = self.shuffle[self.n2:self.n3], self.shuffle[self.n3:nd]
        self.g1 = func.rotated_expanded_scaffer_func
        self.g2 = func.hgbat_func
        self.g3 = func.rosenbrock_func
        self.g4 = func.modified_schwefel_func

    def evaluate(self, x, *args):
        mz = np.dot(self.matrix, x - self.shift)
        return self.g1(mz[self.idx1]) + self.g2(mz[self.idx2]) + self.g3(mz[self.idx3]) + self.g4(mz[self.idx4]) + self.opt


# Hybrid Function 3 (F21 CEC-2014)
class F7:
    def __init__(self, nd=2):
        checkND(nd)
        self.shift = shift_data(7, nd)
        self.matrix = matrix_data(7, nd)
        self.shuffle = shuffle_data(7, nd)
        self.opt = 2100.
        self.sol = self.shift
        self.n1 = int(np.ceil(0.1 * nd))
        self.n2 = int(np.ceil(0.2 * nd)) + self.n1
        self.n3 = int(np.ceil(0.2 * nd)) + self.n2
        self.n4 = int(np.ceil(0.2 * nd)) + self.n3
        self.idx1, self.idx2, self.idx3 = self.shuffle[:self.n1], self.shuffle[self.n1:self.n2], self.shuffle[self.n2:self.n3]
        self.idx4, self.idx5 = self.shuffle[self.n3:self.n4], self.shuffle[self.n4:nd]
        self.g1 = func.rotated_expanded_scaffer_func
        self.g2 = func.hgbat_func
        self.g3 = func.rosenbrock_func
        self.g4 = func.modified_schwefel_func
        self.g5 = func.elliptic_func

    def evaluate(self, x, *args):
        z = x - self.shift
        z1 = np.concatenate((z[self.idx1], z[self.idx2], z[self.idx3], z[self.idx4], z[self.idx5]))
        mz = np.dot(self.matrix, z1)
        return self.g1(mz[:self.n1]) + self.g2(mz[self.n1:self.n2]) + self.g3(mz[self.n2:self.n3]) +\
               self.g4(mz[self.n3:self.n4]) + self.g5(mz[self.n4:]) + self.opt


# Composition Function 1 (F21 CEC-2017)
class F8:
    def __init__(self, nd=2):
        checkND(nd, True)
        self.nd = nd
        self.shift = shift_data(8, nd)
        self.matrix = matrix_data(8, nd)
        self.opt = 2200.
        self.sol = self.shift[0]
        self.xichmas = [10, 20, 30]
        self.lamdas = [1., 10., 1.]
        self.bias = [0, 100, 200]
        self.g0 = func.rastrigin_func
        self.g1 = func.griewank_func
        self.g2 = func.modified_schwefel_func

    def evaluate(self, x, *args):
        nd = self.nd
        # 1. Rastrigin’s Function F5
        z0 = np.dot(self.matrix[:nd, :], x - self.shift[0])
        g0 = self.lamdas[0] * self.g0(z0) + self.bias[0]
        w0 = func.calculate_weight(x - self.shift[0], self.xichmas[0])
        # 2. Griewank’s Function F15
        z1 = np.dot(self.matrix[nd:2*nd, :], x - self.shift[1])
        g1 = self.lamdas[1] * self.g1(z1) + self.bias[1]
        w1 = func.calculate_weight(x - self.shift[1], self.xichmas[1])
        z2 = 1000*(x - self.shift[2])/100
        g2 = self.lamdas[2] * self.g2(z2) + self.bias[2]
        w2 = func.calculate_weight(x - self.shift[2], self.xichmas[2])
        ws = np.array([w0, w1, w2])
        ws = ws / np.sum(ws)
        gs = np.array([g0, g1, g2])
        return np.dot(ws, gs) + self.opt


# Composition Function 2 (F23 CEC-2017)
class F9:
    def __init__(self, nd=2):
        checkND(nd, True)
        self.nd = nd
        self.shift = shift_data(9, nd)
        self.matrix = matrix_data(9, nd)
        self.opt = 2400.
        self.sol = self.shift[0]
        self.xichmas = [10, 20, 30, 40]
        self.lamdas = [10., 1e-6, 10, 1.]
        self.bias = [0, 100, 200, 300]
        self.g0 = func.ackley_func
        self.g1 = func.elliptic_func
        self.g2 = func.griewank_func
        self.g3 = func.rastrigin_func

    def evaluate(self, x, *args):
        nd = self.nd
        # 1. Ackley’s Function F13
        z0 = np.dot(self.matrix[:nd, :], x - self.shift[0])
        g0 = self.lamdas[0] * self.g0(z0) + self.bias[0]
        w0 = func.calculate_weight(x - self.shift[0], self.xichmas[0])
        # 2. High Conditioned Elliptic Function F11
        z1 = np.dot(self.matrix[nd:2*nd, :], x - self.shift[1])
        g1 = self.lamdas[1] * self.g1(z1) + self.bias[1]
        w1 = func.calculate_weight(x - self.shift[1], self.xichmas[1])
        # 3. Girewank Function F15
        z2 = np.dot(self.matrix[2*nd:3*nd, :], x - self.shift[2])
        g2 = self.lamdas[2] * self.g2(z2) + self.bias[2]
        w2 = func.calculate_weight(x - self.shift[2], self.xichmas[2])
        # 4. Rastrigin’s Function F5
        z3 = np.dot(self.matrix[3 * nd:4 * nd, :], x - self.shift[3])
        g3 = self.lamdas[3] * self.g3(z3) + self.bias[3]
        w3 = func.calculate_weight(x - self.shift[3], self.xichmas[3])
        ws = np.array([w0, w1, w2, w3])
        ws = ws / np.sum(ws)
        gs = np.array([g0, g1, g2, g3])
        return np.dot(ws, gs) + self.opt


# Composition Function 3 (F24 CEC-2017)
class F10:
    def __init__(self, nd=10):
        checkND(nd, True)
        self.nd = nd
        self.shift = shift_data(10, nd)
        self.matrix = matrix_data(10, nd)
        self.opt = 2500.
        self.sol = self.shift[0]
        self.xichmas = [10, 20, 30, 40, 50]
        self.lamdas = [10., 1., 10., 1e-6, 1.]
        self.bias = [0, 100, 200, 300, 400]
        self.g0 = func.rastrigin_func
        self.g1 = func.happy_cat_func
        self.g2 = func.ackley_func
        self.g3 = func.discus_func
        self.g4 = func.rosenbrock_func

    def evaluate(self, x, *args):
        nd = self.nd
        # 1. Rastrigin’s Function F5
        z0 = np.dot(self.matrix[:nd, :], x - self.shift[0])
        g0 = self.lamdas[0] * self.g0(z0) + self.bias[0]
        w0 = func.calculate_weight(x - self.shift[0], self.xichmas[0])
        # 2. Happycat Function F17
        z1 = np.dot(self.matrix[nd:2*nd, :], x - self.shift[0])
        g1 = self.lamdas[1] * self.g1(z1) + self.bias[1]
        w1 = func.calculate_weight(x - self.shift[1], self.xichmas[1])
        # 3. Ackley Function F13
        z2 = np.dot(self.matrix[2*nd:3*nd, :], x - self.shift[0])
        g2 = self.lamdas[2] * self.g2(z2) + self.bias[2]
        w2 = func.calculate_weight(x - self.shift[2], self.xichmas[2])
        # 4. Discus Function F12
        z3 = np.dot(self.matrix[3 * nd:4 * nd, :], x - self.shift[0])
        g3 = self.lamdas[3] * self.g3(z3) + self.bias[3]
        w3 = func.calculate_weight(x - self.shift[3], self.xichmas[3])
        # 5. Rosenbrock’s Function F4
        z4 = np.dot(self.matrix[4 * nd:5 * nd, :], 2.048*(x - self.shift[0])/100) + 1
        g4 = self.lamdas[4] * self.g4(z4) + self.bias[4]
        w4 = func.calculate_weight(x - self.shift[4], self.xichmas[4])
        ws = np.array([w0, w1, w2, w3, w4])
        ws = ws / np.sum(ws)
        gs = np.array([g0, g1, g2, g3, g4])
        return np.dot(ws, gs) + self.opt
