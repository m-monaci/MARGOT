import numpy as np
import time
import gurobipy as gp
from gurobipy import GRB
from math import floor
from sklearn.svm import SVC

class LocalSVM_H():
    def __init__(self, D, C, FS, B, alpha):

        self.D = D
        self.C = C                          #C is a list of #branch_nodes elements
        self.FS = FS                        #FS must be in [None, 'H', 'S']
        self.B = B                          #B is a list of #branch_nodes elements
        self.alpha = alpha

        self.T = self.find_T(D)  # total number of nodes
        floorTb = (floor(self.T / 2))
        self.Tb = [i for i in range(floorTb)]  # branch nodes
        self.Tl = [i for i in range(floorTb, self.T)]  # leaf nodes
        self.Tb_first = self.Tb[:self.find_T(D - 2)]  # branch nodes until the last level (escluso)
        self.Tb_last = self.Tb[self.find_T(D - 2):]  # branch nodes of the last level

    def find_T(self, D):
        return pow(2, (D + 1)) - 1

    def train(self, l, x, y, M_w, time_limit):

        self.l = l
        self.M_w = M_w

        self.n = np.shape(x)[1]
        time_t = time_limit/len(self.Tb)

        x_t, y_t = [[] for t in self.Tb], [[] for t in self.Tb]
        w, b = [[] for t in self.Tb], [[] for t in self.Tb]

        x_t[0] = x
        y_t[0] = y

        start = time.time()

        for t in self.Tb:
            if len(x_t[t]) > 0 and len(np.unique(y_t[t])) > 1:
                if t in self.Tb_first:

                    w[t], b[t] = self.local_fit(x_t[t], y_t[t], self.C[t], self.B[t], time_t)
                    x_t[2 * t + 1] = x_t[t][np.dot(x_t[t], w[t]) + b[t] <= -1e-12]
                    x_t[2 * t + 2] = x_t[t][np.dot(x_t[t], w[t]) + b[t] > -1e-12]

                    y_t[2 * t + 1] = y_t[t][np.dot(x_t[t], w[t]) + b[t] <= -1e-12]
                    y_t[2 * t + 2] = y_t[t][np.dot(x_t[t], w[t]) + b[t] > -1e-12]

                else:

                    w[t], b[t] = self.local_fit(x_t[t], y_t[t], self.C[t], self.B[t], time_t)
            else:
                
                w[t] = [0.0 for _ in range(self.n)]
                if len(np.unique(y_t[t])) == 1: b[t] = int(y_t[t][0])
                else: b[t] = [1 if t%2 == 0 else -1][0]

        end = time.time()
        print('Time warm start:', end - start)

        return w, b, end-start

    def local_fit(self, x, y, C, B, time_limit_t):

        P_t = len(x)

        if self.FS is None and self.l == 'l2':

            svc = SVC(C=C, kernel='linear')
            svc.fit(np.asarray(x), (np.asarray(y)).reshape(-1, ))
            w_t = list(svc.coef_[0])
            b_t = svc.intercept_[0]

        else:

            m = gp.Model()

            #vars and constraints
            if self.l == 'l2':
                w = m.addVars(self.n, vtype=GRB.CONTINUOUS, lb=-float('inf'), name='w')
            else:
                w_p = m.addVars(self.n, vtype=GRB.CONTINUOUS, lb=0, name='w_p')
                w_m = m.addVars(self.n, vtype=GRB.CONTINUOUS, lb=0, name='w_m')

            b = m.addVar(vtype=GRB.CONTINUOUS, lb=-float('inf'), name='b')
            xi = m.addVars(P_t, vtype=GRB.CONTINUOUS, name='xi')

            s = m.addVars(self.n, vtype=GRB.BINARY, name='s')  #1 if feature j is taken

            if self.l == 'l2':
                m.addConstrs(int(y[i]) * (sum(w[j] * x[i, j] for j in range(self.n)) + b) >= 1 - xi[i] for i in range(len(x)))
                m.addConstrs(w[j] <= self.M_w * s[j] for j in range(self.n))
                m.addConstrs(w[j] >= - self.M_w * s[j] for j in range(self.n))
            else:
                m.addConstrs(int(y[i]) * (sum((w_p[j]-w_m[j]) * x[i, j] for j in range(self.n)) + b) >= 1 - xi[i] for i in range(len(x)))
                m.addConstrs(w_p[j] <= self.M_w * s[j] for j in range(self.n))
                m.addConstrs(w_m[j] <= self.M_w * s[j] for j in range(self.n))

            if self.FS == 'H':
                m.addConstr(sum(s[j] for j in range(self.n)) <= B)
            else:
                u = m.addVar(vtype=GRB.CONTINUOUS, name='u')
                m.addConstr(u >= sum(s[j] for j in range(self.n)) - B)

            #objective function
            if self.FS == 'S':
                if self.l == 'l2':
                    m.setObjective(0.5 * sum(w[j] ** 2 for j in range(self.n)) + C * sum(xi[i] for i in range(len(x))) + self.alpha * u, sense=GRB.MINIMIZE)
                else:
                    m.setObjective(0.5 * sum(w_p[j]+w_m[j] for j in range(self.n)) + C * sum(xi[i] for i in range(len(x))) + self.alpha * u, sense=GRB.MINIMIZE)
            else:
                if self.l == 'l2':
                    m.setObjective(0.5 * sum(w[j] ** 2 for j in range(self.n)) + C * sum(xi[i] for i in range(len(x))), sense=GRB.MINIMIZE)
                else:
                    m.setObjective(0.5 * sum(w_p[j]+w_m[j] for j in range(self.n)) + C * sum(xi[i] for i in range(len(x))), sense=GRB.MINIMIZE)

            #solve
            m.Params.TimeLimit = time_limit_t
            m.Params.LogToConsole = False
            m.optimize()

            #retrieve solution
            if self.l == 'l2':
                w_t = [m.getVarByName('w[%d]' % j).x for j in range(self.n)]
            else:
                w_t = [m.getVarByName('w_p[%d]' % j).x - m.getVarByName('w_m[%d]' % j).x for j in range(self.n)]

            b_t = m.getVarByName('b').x

        return w_t, b_t
