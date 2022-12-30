from margot import Margot
from preprocessing import *
import os.path
import pandas as pd

path = ''
folder = "results_margot/"
if not os.path.isdir(folder):
    os.mkdir(folder)
path = ''+folder

folder_plots = "plots/"
if not os.path.isdir(path + folder_plots):
    os.mkdir(path + folder_plots)
path_plots = path + folder_plots

date = '' #insert the date
datasets = ['breast_cancer_diagnostic','breast_cancer_wisconsin'] #,'climate_model','cleveland','ionosphere','parkinsons','sonar','spectf','tic_tac_toe','wholesale']

D = 2
l = 'l2'    #l must be in ['l1','l2']

#each C is a list of #branch_nodes elements
Cs = [[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,10**4,10**4,10**4],[10**(-1),10**(-1),10**(-1)],[10**(-3),10**(-1),10**(-1)],[10**(-1),10**(-1),10**(-1)],[1,1,1],[10**3,10**3,10**3]]
C_H = [[10**3,10**3,10**3],[10**2,10**3,10**3],[10,10,10],[10**4,10**4,10**4],[10**4,10**5,10**5],[10,10**3,10**3],[10**(-4),10,10],[10**(-4),10**(-2),10**(-2)],[10**(-2),1,1],[10**(-2),10**2,10**2]]
C_S = [[10**2,10**2,10**2],[1,1,1],[10**2,10**2,10**2],[1,1,1],[10**2,10**4,10**4],[1,1,1],[1,1,1],[10**(-4),10**2,10**2],[10**(-1),10**(-1),10**(-1)],[10**2,10**2,10**2]]

FSs = [None,'H','S']                                                                     #FS must be in [None, 'H', 'S']
alphas = [2**10, 2**4, 2**10, 2**8, 2**10, 2**2, 2**2, 2**8, 2**0, 2**8]
Bs = [[2,2,2],[2,2,2],[2,4,4],[2,2,2],[2,4,4],[1,2,2],[1,2,2],[2,3,3],[2,3,3],[1,2,2]]   #each B is a list of #branch_nodes elements

warm_start = True
time_limit = 2 #10*60

columns = ['Date','Dataset','(P,n)','(P-1,P1)','D','l','C','FS','B','alpha','Warm start','Obj value','Train ACC','Test ACC','Train CM','Test CM','Train BACC','Test BACC','Time ws','Time','Total time','Gap','F','|F|','F_t','|F_t|','Vars']

try:
    df_tot = pd.read_excel(path + 'Stat_' + 'Margot_' + str(date) + '.xlsx', index_col = None, header = 0)
except:
    df_tot = pd.DataFrame(columns = columns)

for i in range(len(datasets)):
    dataset = datasets[i]
    for FS in FSs:
        if FS == 'H': C = C_H[i]
        elif FS == 'S': C = C_S[i]
        else: C = Cs[i]

        alpha = alphas[i]
        B = Bs[i]

        x, x_test, y, y_test = preprocessing_data(dataset, 0.20)

        print('\nDATASET: %s \n' %(dataset))
        P,n = len(x)+len(x_test), len(x[0])
        y_tot = np.hstack((y,y_test))
        Pm1, P1 = len(y_tot[y_tot==-1]), len(y_tot[y_tot==1])
        print(dataset, P, n, len(y_tot[y_tot==-1]),len(y_tot[y_tot==1]))

        margot = Margot(D = D, C = C, FS = FS, B = B, alpha = alpha, l = l)
        obj_value, time_ws, time_opt, mip_gap, F, cardF, F_t, cardF_t, vars_dict = margot.train(x = x, y = y, dataset = dataset, warm_start = warm_start, time_limit = time_limit)

        if obj_value is None: continue

        acc_train, cm_train, bacc_train = margot.score(x, y)
        acc_test, cm_test, bacc_test = margot.score(x_test, y_test)

        margot.tree_plot(x, y, phase = 'train', path = path_plots, date = date)
        margot.tree_plot(x_test, y_test, phase = 'test', path = path_plots, date = date)

        row = np.array([date, dataset, (P, n), (Pm1, P1), D, l, C, FS, B, alpha, warm_start, obj_value, acc_train, acc_test,
         cm_train, cm_test, bacc_train, bacc_test, time_ws, time_opt, time_ws+time_opt, mip_gap, F, cardF, F_t, cardF_t,
         vars_dict], dtype=object).reshape(1, -1)

        df_new = pd.DataFrame(row, columns = columns)
        df_tot = pd.concat([df_tot,df_new])
        df_tot.to_excel(path + 'stats_margot_.xlsx', index = False, header = True)

        print('TOTAL TIME OPT: ' + str(time_opt + time_ws))
        print('ACC TRAIN', acc_train)
        print('ACC TEST', acc_test)

