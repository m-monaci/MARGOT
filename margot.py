import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from math import ceil, floor
import gurobipy as gp
from gurobipy import GRB, quicksum
import matplotlib.pyplot as plt
import pygraphviz as pgv
import os
import time
from warm_start import LocalSVM_H

if not 'C:\\Program Files\\Graphviz\\bin' in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin'

def get_sub(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    sub_s = "ₐ₈CDₑբGₕᵢⱼₖₗₘₙₒₚQᵣₛₜᵤᵥwₓᵧZₐ♭꜀ᑯₑբ₉ₕᵢⱼₖₗₘₙₒₚ૧ᵣₛₜᵤᵥwₓᵧ₂₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
    res = x.maketrans(''.join(normal), ''.join(sub_s))
    return x.translate(res)

class Margot():
    def __init__(self, D = 2, C = 1.0, FS = None, B = None, alpha = None, l = 'l2'):

        self.D = D
        self.C = C                                      #C is a list of #branch_nodes elements
        self.FS = FS                                    #FS must be in [None, 'H', 'S']
        self.B = B                                      #B is a list of #branch_nodes elements
        self.alpha = alpha
        self.l = l                                      #'l1' or 'l2'

        self.T = self.find_T(D)
        floorTb = (floor(self.T / 2))
        self.Tb = [i for i in range(floorTb)]
        self.Tl = [i for i in range(floorTb,self.T)]
        self.Tb_first = self.Tb[:self.find_T(D-2)]
        self.Tb_last = self.Tb[self.find_T(D-2):]
        self.parents = [-1]+[int(i) for i in np.arange(0,self.find_T(D-1), 0.5)]
        self.S_last, self.Sl_last, self.Sr_last = self.sub_nodes()

    def find_T(self, D):
        return pow(2, (D + 1)) - 1

    def sub_tree(self):

        Tb_level = [list(np.arange(2 ** level - 1, 2 ** (level + 1) - 1)) for level in range(self.D+1)]
        S, Sr, Sl = [[] for _ in self.Tb], [[] for _ in self.Tb], [[] for _ in self.Tb]
        S[0] = Tb_level
        Sl[0] = [elem[:ceil(len(elem)/2)] for elem in S[0]]
        Sr[0] = [elem[floor(len(elem)/2):] for elem in S[0]]
        for t in self.Tb[1:]:
            if t % 2 != 0:
                S[t] = Sl[self.parents[t]][1:]
                Sl[t] = [elem[:ceil(len(elem)/2)] for elem in S[t]]
                Sr[t] = [elem[floor(len(elem)/2):] for elem in S[t]]
            else:
                S[t] = Sr[self.parents[t]][1:]
                Sl[t] = [elem[:ceil(len(elem)/2)] for elem in S[t]]
                Sr[t] = [elem[floor(len(elem)/2):] for elem in S[t]]

        S = [[j for e in elem for j in e] for elem in S]

        return S

    def sub_nodes(self):

        S = self.sub_tree()
        Tb_last = set(self.Tb_last)
        S_last = [sorted(list(set(S[t]).intersection(Tb_last))) for t in self.Tb]
        Sl_last = [S_last[t][:len(S_last[t])//2] for t in self.Tb_first]
        Sr_last = [S_last[t][len(S_last[t])//2:] for t in self.Tb_first]

        return S_last, Sl_last, Sr_last

    def model(self, warm_start, time_limit_ws):

        self.M_xi, self.M_H, self.M_w = 50, 100, 50
        self.eps = 1e-03

        self.m = gp.Model()                 #initialize the model
        vars = self.variables()
        self.objective(vars)
        self.constraints(vars)

        self.time_ws = 0    #initialize the time

        if warm_start:
             svmtree_ws = LocalSVM_H(self.D, self.C, self.FS, self.B, self.alpha)
             w_ws, b_ws, self.time_ws = svmtree_ws.train(self.l, self.x, self.y, self.M_w, time_limit_ws)
             w, w_p, w_m, b, xi, z, s, u = vars

             for t in self.Tb:
                b[t].start = b_ws[t]
                if self.l == 'l2':
                    for j in range(self.n):
                        w[(t, j)].start = w_ws[t][j]
                else:
                    for j in range(self.n):
                        if w_ws[t][j] >= -1e-12:
                            w_p[(t, j)].start = w_ws[t][j]
                            w_m[(t, j)].start = 0.0
                        else:
                            w_p[(t, j)].start = 0.0
                            w_m[(t, j)].start = w_ws[t][j]

             self.m.update()

    def variables(self):

        w, w_m, w_p = None, None, None

        if self.l == 'l2':
            w = self.m.addVars(self.Tb, self.n, vtype = GRB.CONTINUOUS, lb=-float('inf'), name='w')
        else:
            w_m = self.m.addVars(self.Tb, self.n, vtype=GRB.CONTINUOUS, lb=0, name='w_m')
            w_p = self.m.addVars(self.Tb, self.n, vtype=GRB.CONTINUOUS, lb=0, name='w_p')

        b = self.m.addVars(self.Tb, vtype=GRB.CONTINUOUS, lb=-float('inf'), name='b')
        xi = self.m.addVars(self.Tb, self.P, vtype=GRB.CONTINUOUS, name='xi')
        z = self.m.addVars(self.P, self.Tb_last, vtype=GRB.BINARY, name='z')

        s, u = None, None

        if self.FS in ['H', 'S']:
            s = self.m.addVars(self.Tb, self.n, vtype=GRB.BINARY, name='s')
            if self.FS == 'S':
                u = self.m.addVars(self.Tb, vtype=GRB.CONTINUOUS, name='u')

        return w, w_p, w_m, b, xi, z, s, u

    def objective(self, vars):

        w, w_p, w_m, b, xi, z, s, u = vars

        if self.FS == 'S':
            if self.l == 'l2':
                self.m.setObjective(0.5 * quicksum(w[t, j] * w[t, j] for t in self.Tb for j in range(self.n))
                                    + quicksum(self.C[t] * xi[t, i] for i in range(self.P) for t in self.Tb)
                                    + self.alpha*quicksum(u[t] for t in self.Tb), sense=GRB.MINIMIZE)
            else:
                self.m.setObjective(
                    0.5 * quicksum(w_p[t, j] + w_m[t, j] for t in self.Tb for j in range(self.n)) +
                    quicksum(self.C[t]*xi[t, i] for i in range(self.P) for t in self.Tb)
                    + self.alpha * quicksum(u[t] for t in self.Tb), sense=GRB.MINIMIZE)
        else:
            if self.l == 'l2':
                self.m.setObjective(0.5 * quicksum(w[t, j] * w[t, j] for t in self.Tb for j in range(self.n))
                                + quicksum(self.C[t]*xi[t, i] for i in range(self.P) for t in self.Tb), sense=GRB.MINIMIZE)
            else:
                self.m.setObjective(
                    0.5 * quicksum(w_p[t, j] + w_m[t, j] for t in self.Tb for j in range(self.n)) +
                    quicksum(self.C[t]*xi[t, i] for i in range(self.P) for t in self.Tb), sense=GRB.MINIMIZE)

    def constraints(self, vars):

        w, w_p, w_m, b, xi, z, s, u = vars

        if self.l == 'l2':

            self.m.addConstrs(int(self.y[i]) * (quicksum(w[t, j] * self.x[i, j] for j in range(self.n)) + b[t]) >= 1 - xi[t, i] - self.M_xi * (1 - quicksum(z[i, l] for l in self.S_last[t])) for t in self.Tb for i in range(self.P))

            self.m.addConstrs(quicksum(w[t, j] * self.x[i,j] for j in range(self.n)) + b[t] >= - self.M_H * (1 - quicksum(z[i, l] for l in self.Sr_last[t]))
                      for i in range(self.P) for t in self.Tb_first)

            self.m.addConstrs(quicksum(w[t, j] * self.x[i,j] for j in range(self.n)) + b[t] + self.eps <= self.M_H * (1 - quicksum(z[i, l] for l in self.Sl_last[t]))
                      for i in range(self.P) for t in self.Tb_first)

        else:
            self.m.addConstrs(int(self.y[i]) * (quicksum((w_p[t, j]-w_m[t,j]) * self.x[i, j] for j in range(self.n)) + b[t]) >= 1 - xi[t, i] - self.M_xi * (1 - quicksum(z[i, l] for l in self.S_last[t])) for t in self.Tb for i in range(self.P))

            self.m.addConstrs(quicksum((w_p[t, j]-w_m[t,j]) * self.x[i,j] for j in range(self.n)) + b[t] >= - self.M_H * (1 - quicksum(z[i, l] for l in self.Sr_last[t]))
                      for i in range(self.P) for t in self.Tb_first)

            self.m.addConstrs(quicksum((w_p[t, j]-w_m[t,j]) * self.x[i,j] for j in range(self.n)) + b[t] + self.eps <= self.M_H * (1 - quicksum(z[i, l] for l in self.Sl_last[t]))
                      for i in range(self.P) for t in self.Tb_first)

        self.m.addConstrs(quicksum(z[i, t] for t in self.Tb_last) == 1 for i in range(self.P))

        if self.FS in ['H','S']:

            if self.l == 'l2':

                self.m.addConstrs(w[t, j] <= self.M_w * s[t, j] for t in self.Tb for j in range(self.n))
                self.m.addConstrs(w[t, j] >= -self.M_w * s[t, j] for t in self.Tb for j in range(self.n))

            else:

                self.m.addConstrs(w_p[t, j] <= self.M_w * s[t, j] for t in self.Tb for j in range(self.n))
                self.m.addConstrs(w_m[t, j] <= self.M_w * s[t, j] for t in self.Tb for j in range(self.n))

            if self.FS == 'H':
                self.m.addConstrs(quicksum(s[t, j] for j in range(self.n)) <= self.B[t] for t in self.Tb)

            else:
                self.m.addConstrs(u[t] >= quicksum(s[t, j] for j in range(self.n)) - self.B[t] for t in self.Tb)

    def train(self, x = None, y = None, dataset = None, warm_start = True, time_limit_ws = 30, log_flag = True, time_limit = 10*60):

        self.x, self.y = x, y
        self.dataset = dataset
        self.P, self.n = len(x), len(x[0])

        self.model(warm_start, time_limit_ws)

        self.m.Params.LogToConsole = log_flag
        if time_limit is not None:
            self.m.Params.TimeLimit = time_limit

        start = time.time()
        self.m.optimize()
        end = time.time()

        print('Time %s:', end - start)
        print('Optimization was stopped with status' + str(self.m.Status))

        try:
            obj_value = self.m.ObjVal
            w_p, w_m = None, None

            if self.l == 'l2':
                self.w = [[self.m.getVarByName('w[%d,%d]' % (t, j)).x for j in range(self.n)] for t in self.Tb]
            else:
                w_p = [[self.m.getVarByName('w_p[%d,%d]' % (t, j)).x for j in range(self.n)] for t in self.Tb]
                w_m = [[self.m.getVarByName('w_m[%d,%d]' % (t, j)).x for j in range(self.n)] for t in self.Tb]
                self.w = [[w_p[t][j]-w_m[t][j] for j in range(self.n)] for t in self.Tb]

            self.b = [self.m.getVarByName('b[%d]' % t).x for t in self.Tb]
            xi = [[self.m.getVarByName('xi[%d,%d]' % (t, i)).x for i in range(self.P)] for t in self.Tb]
            z = [[self.m.getVarByName('z[%d,%d]' % (i, t)).x for t in self.Tb_last] for i in range(self.P)]

            s, u = None, None

            if self.FS in ['S','H']:
                s = [[self.m.getVarByName('s[%d,%d]' % (t, j)).x for j in range(self.n)] for t in self.Tb]
                if self.FS == 'S':
                    u = [self.m.getVarByName('u[%d]' % t).x for t in self.Tb]

            vars_dict = {'w': self.w, 'w_p': w_p, 'w_m': w_m, 'b': self.b, 'xi': xi, 'z': z, 's': s, 'u': u}

            F = list(np.unique([j for w_t in self.w for j in range(len(w_t)) if np.abs(w_t[j]) >= 5e-04]))
            cardF = len(F)
            self.F_t = [[j for j in range(len(self.w[t])) if np.abs(self.w[t][j]) >= 5e-04] for t in self.Tb]
            cardF_t = [len(elem) for elem in self.F_t]

            return obj_value, self.time_ws, end - start, self.m.MIPGap, F, cardF, self.F_t, cardF_t, vars_dict
        except:
            print('Error: impossible to retrieve a solution')
            return None, None, None, None, None, None, None, None, None

    def predict(self, x):

        paths = np.zeros((len(x), self.D + 1), dtype=object)  # path (sequences of nodes) for each sample i
        for i in range(len(x)):
            for level in range(self.D):
                t = paths[i][level]
                if (np.dot(self.w[t], x[i]) + self.b[t]) <= -1e-12:
                    paths[i][level + 1] = 2 * t + 1
                else:
                    paths[i][level + 1] = 2 * t + 2

        return np.array([1 if paths[i][-1] % 2 == 0 else -1 for i in range(len(x))]).reshape(-1,)

    def score(self, x, y):

        pred = self.predict(x)
        acc = accuracy_score(y, pred)
        cm = confusion_matrix(y, pred)

        tn, tp = sum([1 for i in range(len(pred)) if pred[i] == -1 and pred[i] == y[i]]), sum([1 for i in range(len(pred)) if pred[i] == 1 and pred[i] == y[i]]),
        P_neg, P_pos = len(y[y == -1]), len(y[y == 1])
        tnr, tpr = tn/P_neg, tp/P_pos
        bacc = (tnr+tpr)/2

        return acc, cm, bacc

    def tree_plot(self, x, y, phase = 'train', path = '', date = None, input = None):

        if input is None:
            w, b = self.w, self.b
        else:
            w,b = input

        paths = np.zeros((len(x), self.D + 1), dtype=object)  # path (sequences of nodes) for each sample i

        for i in range(len(x)):
            for d in range(self.D):
                t = paths[i][d]
                if (np.dot(w[t], x[i]) + b[t]) <= -1e-12:
                    paths[i][d + 1] = 2 * t + 1
                else:
                    paths[i][d + 1] = 2 * t + 2

        N_m1 = [0 for node in self.Tb+self.Tl]
        N_1 = [0 for node in self.Tb+self.Tl]
        for i in range(len(x)):
            for node in self.Tb+self.Tl:
                if node in paths[i]:
                    if y[i] == -1:
                        N_m1[node] += 1
                    else:
                        N_1[node] += 1

        g = pgv.AGraph(directed=True)  # initialize the graph
        nodes = np.arange(self.T)  # nodes = np.append(self.Tb, self.Tl)

        l_feat_used = [[j for j in range(len(w[t])) if w[t][j] <= -(5e-04) or w[t][j] >= 5e-04] for t in self.Tb]

        for n in nodes:  # the graph has a node for each node of the tree
            g.add_node(n, shape='circle', size=24)
            if n != 0:
                parent = self.parents[n]
                g.add_edge(parent, n)

        for t in self.Tb:
            g.get_node(t).attr['label'] = '[' + str(N_m1[t]) + ',' + str(N_1[t]) + ']' + '\\n' 'F{}:'.format(get_sub('%s' % t)) + str(
                self.F_t[t])

        for le in self.Tl:
            if le % 2 == 0:
                g.get_node(le).attr['label'] = '[' + str(N_m1[le]) + ',' + str(N_1[le]) + ']' + '\\n' + '+1'
                g.get_node(le).attr['color'] = 'green'
            else:
                g.get_node(le).attr['label'] = '[' + str(N_m1[le]) + ',' + str(N_1[le]) + ']' + '\\n' + '-1'
                g.get_node(le).attr['color'] = 'red'

        g.layout(prog='dot')
        g.draw(path + 'graph_margot_FS%s_%s_%s_D%d_%s_C%s_alpha%s_gap%s_%s.png' % (
        self.FS, self.dataset, phase, self.D, self.l, str(self.C), str(self.alpha), str(round(self.m.MIPGap*100,2)), date))
        img = plt.imread(path + 'graph_margot_FS%s_%s_%s_D%d_%s_C%s_alpha%s_gap%s_%s.png' % (
        self.FS, self.dataset, phase,  self.D, self.l, str(self.C), str(self.alpha), str(round(self.m.MIPGap*100,2)), date))
        plt.imshow(img)
        plt.show()
        return g
