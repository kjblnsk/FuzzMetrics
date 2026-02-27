#####################################################
#                                                   #
# Piece of code for fuzzy numbers                   #
# including portfolio optimization                  #
#                                                   #
# 12.02.2025                                        #
#                                                   #
# Kaja Bilińska, kaja.bilinska@pwr.edu.pl           # 
# Jan Schneider, jan.schneider@pwr.edu.pl           #
#                                                   #
#   	arXiv:2602.20183                            #
#                                                   #
#####################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import quad
import time
import seaborn as sns
import pandas as pd
from sympy import symbols, solve
from datetime import datetime

class FuzzyNumber:

    def __init__(self):
        pass

    def make_fuzzy(self, L=2, M=6, R=10, number_of_nodes_sym = 5, manual = False, X = [], Y = [], plot=False):
        """
        For given (L, M, R) provide random fuzzy number
        Nodes on left (i.e., LM) and right side (i.e., MR) is symmetric (so far) and given by number_of_nodes_sym (excluding L, M, R)

        Params:
            L (int, float): left node of fuzzy number
            M (int, float): mode of fuzzy number
            R (int, float): right node of fuzzy number
        
        Returns:
            self.X, self.Y (list of floats): X and Y coordinates of nodes for fuzy number
        """

        if not manual:

            self.L = L
            self.M = M
            self.R = R

            self.number_of_left_y_nodes = number_of_nodes_sym
            left_y_list = np.sort(np.random.rand(self.number_of_left_y_nodes))
            left_x_list = np.sort(np.random.uniform(L, M, self.number_of_left_y_nodes))

            self.number_of_right_y_nodes = self.number_of_left_y_nodes
            right_y_list = np.sort(np.random.rand(self.number_of_right_y_nodes))[::-1]
            right_x_list = np.sort(np.random.uniform(M, R, self.number_of_right_y_nodes))

            Y = np.append([0], left_y_list)
            Y = np.append(Y, [1])
            Y = np.append(Y, right_y_list)
            Y = np.append(Y, [0])

            X = np.append(L, left_x_list)
            X = np.append(X, M)
            X = np.append(X, right_x_list)
            X = np.append(X, R)

            self.X = X
            self.Y = Y

        if manual and X != [] and Y != []:

            self.X = X
            self.Y = Y
        
        if plot:
            plt.figure(figsize=(8,8))
            plt.plot(X, Y, linewidth=2)
            plt.xlabel("x", fontsize = 20)
            plt.ylabel("μ(x)", fontsize=20)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.show()

        return self.X, self.Y

    def manual_fuzzy(self, X, Y, interpolate=True, plot=False):
        """
        Allows you to enter X and Y coordinates manually

        If interpolate == True, the interpolated form of fuzzy number (with piece linar functions) is returned

        Params:
            X (tab of int / float): X coordinates of fuzzy number (ordered)
            Y (tab of int / float): Y coordinates of fuzzy number (ordered)
            interpolate (bool): provides or not the interpolated form of fuzzy number

        Returns:
            None or interpolated form of fuzzy number
        """

        self.F1_X = X
        self.F1_Y = Y

        number_of_side_nodes = int(( len(X) - 3 ) / 2)

        F1_X_left = self.F1_X[:number_of_side_nodes+2]
        F1_X_right = self.F1_X[number_of_side_nodes+1:]
        F1_Y_left = self.F1_Y[:number_of_side_nodes+2]
        F1_Y_right = self.F1_Y[number_of_side_nodes+1:]

        X_Ksi_u = F1_Y_right
        Y_Ksi_u = F1_X_right
        X_Ksi_d = F1_Y_left
        Y_Ksi_d = F1_X_left

        Ksi_u = interp1d(X_Ksi_u, Y_Ksi_u, kind='linear')
        Ksi_d = interp1d(X_Ksi_d, Y_Ksi_d, kind='linear')

        if plot:

            x_mesh = np.linspace(F1_Y_left[0], F1_Y_right[0], 1000)

            plt.figure(figsize=(8,8))

            plt.plot(F1_X_left, F1_Y_left, color='#c2553d', linestyle='-', linewidth=3, label = r"$ξ_l$")
            plt.plot(x_mesh, Ksi_d(x_mesh), color='#c2553d', linestyle='--', label = r"$ξ_d$", linewidth=3)
            plt.plot(F1_X_right, F1_Y_right, color='#3f7f93', linestyle='-', linewidth=3, label = r"$ξ_r$")
            plt.plot(x_mesh, Ksi_u(x_mesh), label = r"$ξ_u$", color='#3f7f93', linestyle='--', linewidth=3)

            x_vals = np.linspace(plt.xlim()[0], plt.xlim()[1], 1000)
            plt.plot(x_vals, x_vals, color="k", linestyle="-", linewidth=.5)

            plt.legend(fontsize=15)
            plt.xlabel("x", fontsize = 20)
            plt.ylabel("μ(x)", fontsize=20)
            plt.xticks([1,2,3,4,5,6,7,8])
            plt.yticks([1,2,3,4,5,6,7,8])
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.xlim([0,8.1])
            plt.ylim([0,8.1])

            plt.axhline(y=1, linestyle=':', color='black')
            plt.axvline(x=1, linestyle=':', color='black')

            plt.show()

        if interpolate:
            return Ksi_u, Ksi_d
        elif interpolate == False:
            return None

    def multiple_sum(self, fuzzies, weights=None, plot=False):
        """
        Provides multiple sum of given fuzzies (split into Ksi up and down) with respective weights
        Required for portfolio optimization

        sum_Ksi_u = w1 * Ksi_1_u + w2 * Ksi_2_u + ... + wn * Ksi_n_u
        sum_Ksi_d = w1 * Ksi_1_d + w2 * Ksi_2_d + ... + wn * Ksi_n_d

        Params:
            fuzzies (tab of floats): required form [[Ksi_1_u, Ksi_1_d], [Ksi_2_u, Ksi_2_d], ..., [Ksi_n_u, Ksi_n_d]]
            weights (tab of floats): [w1, w2, ..., wn], whereas w1 + w2 + ... + wn = 1 for portfolio optimization, default - all equal weights to 1

        Returns:
            sum_Ksi_u, sum_Ksi_d (interpolated functions): sum of Ksi up and down separately given as functions
        """

        if weights is None:
            weights = np.ones(len(fuzzies))

        x = symbols("x")

        def sum_Ksi_u(x):
            return sum(weight * fuzzy[0](x) for weight, fuzzy in zip(weights, fuzzies))
        
        def sum_Ksi_d(x):
            return sum(weight * fuzzy[1](x) for weight, fuzzy in zip(weights, fuzzies))

        self.l = sum_Ksi_d(0)
        self.m = sum_Ksi_u(1)
        self.r = sum_Ksi_u(0)

        if plot:
            x_mesh = np.linspace(0, 1, 1000)
            plt.plot(x_mesh, sum_Ksi_u(x_mesh), label = "Ksi up")
            plt.plot(x_mesh, sum_Ksi_d(x_mesh), label = "Ksi down")
            plt.legend()
            plt.show()

        return sum_Ksi_u, sum_Ksi_d, self.l, self.m, self.r

    def calculate_skewness(self, fuzzy, version=3, input_form = "function", mean_skewness_var=False, ALPHA=.25):
        """

        Params:
            fuzzy (): fuzzy number to be investigated in terms of its skewness
                for "function" a required form is tab: [Ksi up, Ksi down]
                for "coordinates" a required form is tab: [X, Y] --> use manual fuzzy and get Ksi up and down

            version (int): 1, 2, 3 (default 3)
            input_form (str): "function" for interpolated form of fuzzy or "coordinates" for X and Y nodes (in progress...)

        Returns:


        """

        def integrate_partial_sum(func, a, b, num_intervals=10):
            """
            Some of functions here are more complex and due to IntegrationWarnings
            a partial sum must be implemented

            Calculate int of func on [a, b] with num_intervals

            Params:
                func: function to be integrated
                a, b (float, int): start and finish of the int range
                num_intervals (int): numbers of splits

            Returns:
                sum of partial ints
            """
            sub_intervals = np.linspace(a, b, num_intervals + 1)
            integral_sum = 0
            
            for i in range(num_intervals):
                S, _ = quad(func, sub_intervals[i], sub_intervals[i + 1])
                integral_sum += S
            
            return integral_sum

        def skewness_v1():
            """
            Li et al.
            """

            def x_Ksi_d_x_Ksi_u(x):
                return x * (self.Ksi_d(x) + self.Ksi_u(x))

            def _Ksi_d(x):
                return self.Ksi_d(x)
            
            def _Ksi_u(x):
                return self.Ksi_u(x)

            self.mean_v1 = integrate_partial_sum(x_Ksi_d_x_Ksi_u, 0, 1)

            def x_sum(x):
                return x * ( (_Ksi_d(x) - self.mean_v1)**3 + (_Ksi_u(x) - self.mean_v1)**3 )

            self.skewness_v1 = integrate_partial_sum(x_sum, 0, 1, num_intervals=15)

            def var_v1(x):
                return x * ( (_Ksi_d(x) - self.mean_v1)**2 + (_Ksi_u(x) - self.mean_v1)**2 )
        
            self.var_v1 = integrate_partial_sum(var_v1, 0, 1, num_intervals=15)

        def skewness_v2():
            """
            Bermudez & Vercher
            """

            def sum_Ksi_u_d(x):
                return self.Ksi_d(x) + self.Ksi_u(x)

            self.mean_v2 = integrate_partial_sum(sum_Ksi_u_d, 0, 1) / 2

            def dif_Ksi_u_d(x):
                return x * ( self.Ksi_u(x) - self.Ksi_d(x) )
            
            self.var_v2 = 2 * integrate_partial_sum(dif_Ksi_u_d, 0, 1)

            def Ksi_d_mean(x):
                return (self.Ksi_d(x) - self.mean_v2)**3
            
            def Ksi_u_mean(x):
                return (self.Ksi_u(x) - self.mean_v2)**3

            i1 = integrate_partial_sum(Ksi_d_mean, 0, 1)
            i2 = integrate_partial_sum(Ksi_u_mean, 0, 1)
            self.third_possibilistic_moment = .5 * i1 + .5 * i2

            self.skewness_v2 = self.third_possibilistic_moment / self.var_v2**3
    
        def skewness_v3(alpha = .1): # 0.25
            """
            Our definition
            """

            try:
                alpha = self.alpha
            except:
                pass
            
            l = self.Ksi_d(0)
            m = self.Ksi_d(1)
            r = self.Ksi_u(0)

            m_up = m
            m_down = m
            
            def Ksi_d(x):
                return self.Ksi_d(x)
            
            def Ksi_u(x):
                return self.Ksi_u(x)

            E_18 = (Ksi_d(1-alpha) + Ksi_d(alpha) - 2*Ksi_d(.5)) / (Ksi_d(1-alpha) - Ksi_d(alpha))
            E_19 = (Ksi_u(alpha) + Ksi_u(1-alpha) - 2*Ksi_u(.5)) / (Ksi_u(alpha) - Ksi_u(1-alpha))
            E_20 = (Ksi_d(alpha) + Ksi_u(alpha) - 2 * m) / (Ksi_u(alpha) - Ksi_d(alpha))

            self.skewness_v3 = 1/4 * E_18 + 1/4 * E_19 + 1/2 * E_20

            def dif_Ksi_u_d(x):
                return self.Ksi_u(x) - self.Ksi_d(x)
        
            self.var_v3 = integrate_partial_sum(dif_Ksi_u_d, 0, 1)

            self.mean_v3 = m
  
        def skewness_v4():
            """
            Our definition
            """
            
            l = self.Ksi_d(0)
            m = self.Ksi_d(1)
            r = self.Ksi_u(0)

            m_up = m
            m_down = m
            
            def Ksi_d(x):
                return self.Ksi_d(x)
            
            def Ksi_u(x):
                return self.Ksi_u(x)
            
            def e1(x):
                return Ksi_d(1 - x) + Ksi_d(x) - 2 * Ksi_d(0.5)
            
            def e2(x):
                return Ksi_d(1 - x) - Ksi_d(x)

            Sgm_Ksi_d = integrate_partial_sum(e1, 0, .5) / integrate_partial_sum(e2, 0, .5)

            def e3(x):
                return Ksi_u(x) + Ksi_u(1 - x) - 2 * Ksi_u(0.5)

            def e4(x):
                return Ksi_u(x) - Ksi_u(1 - x)
            
            Sgm_Ksi_u = integrate_partial_sum(e3, 0, .5) / integrate_partial_sum(e4, 0, .5)

            def e5(x):
                return Ksi_d(x) + Ksi_u(x) - 2 * m

            def e6(x):
                return Ksi_u(x) - Ksi_d(x)

            S_outer = integrate_partial_sum(e5, 0, .5) / integrate_partial_sum(e6, 0, .5)

            self.skewness_v4 = 1/4 * Sgm_Ksi_d + 1/4 * Sgm_Ksi_u + 1/2 * S_outer

            def dif_Ksi_u_d(x):
                return self.Ksi_u(x) - self.Ksi_d(x)
        
            self.var_v3 = integrate_partial_sum(dif_Ksi_u_d, 0, 1)

            self.mean_v3 = m

        if input_form == "function":
            self.Ksi_u = fuzzy[0]
            self.Ksi_d = fuzzy[1]

            if version == 1:
                skewness_v1()
            elif version == 2:
                skewness_v2()
            elif version == 3:
                skewness_v3()
            elif version == 4:
                skewness_v4()

        elif input_form == "coordinates":
            print("Not implemented yet...\nFin")
            exit()
        else:
            print("Wrong input of a fuzzy number given.\nFin")
            exit()

        if mean_skewness_var:
            if version == 1:
                return self.mean_v1, self.skewness_v1, self.var_v1
            elif version == 2:
                return self.mean_v2, self.skewness_v2, self.var_v2
            elif version == 3:
                return self.mean_v3, self.skewness_v3, self.var_v3
            elif version == 4:
                return self.mean_v3, self.skewness_v4, self.var_v3
        else:
            if version == 1:
                return self.skewness_v1
            elif version == 2:
                return self.skewness_v2
            elif version == 3:
                return self.skewness_v3
            elif version == 4:
                return self.skewness_v4

    def portfolio_optimization(self, n, version="S", weights="auto", skewness_version=3, N=11, alpha =.25):
        """
        Portfolio optimization issue
        Three versions are possible:
            "S" --> E >= alpha, var <= beta, max(S)
            "E" --> S >= gamma, var <= beta, max(E)
            "var" --> E>= alpha, S >=gamma, min(var)

        Params:
            n (int) > 0: number of assets (Ksi and corresponding weight) considered
            version (int): 1, 2, 3
            weights (str / tab of floats): method of weights implementation, default = "auto",
                but also tab of floats can be given, e.g., [[w1, w2, w3], [w1', w2', w3'], ..., [w1^n', w2^n', w3^n']]
            N (int) > 0: number of mesh points for each weight
        """

        self.alpha = alpha

        start = datetime.now()

        if weights == "auto":

            def generate_weights_recursive(n, num_points=N):
                step = 1 / (num_points - 1)
                weights = []

                def helper(current, remaining_sum, depth):
                    if depth == n - 1:
                        if 0 <= remaining_sum <= 1:
                            weights.append(current + [remaining_sum])
                        return
                    for i in range(num_points):
                        value = i * step
                        if value > remaining_sum:
                            break

                        helper(current + [value], remaining_sum - value, depth + 1)

                helper([], 1.0, 0)

                return np.array(weights)
            
            weights = generate_weights_recursive(n)
        
        else:

            if type(weights) != list:
                print("Wrong format of weights given.\nFin")
                exit()

        nodes = 1

        y_5 = [0, 0.25, 1, 0.75, 0]
        x_5_ver1 = [2, 2.25, 4, 4.25, 10]
        x_5_ver2 = [102, 102.25, 104, 104.25, 110]
        x_5_ver3 = [200, 225, 400, 425, 1000]

        # generate random fuzzy numbers with 3 nodes on each side (excluding L, M, R), e.g.:
        version_opt = 5
        F1_X, F1_Y = FuzzyNumber().make_fuzzy(number_of_nodes_sym=nodes, L = 2, M = 4, R = 10, manual=True, X = x_5_ver1, Y = y_5)
        F2_X, F2_Y = FuzzyNumber().make_fuzzy(number_of_nodes_sym=nodes, L = 102, M = 105, R = 110, manual=True, X = x_5_ver2, Y = y_5)
        F3_X, F3_Y = FuzzyNumber().make_fuzzy(number_of_nodes_sym=nodes, L = 200, M = 600, R = 1000, manual=True, X = x_5_ver3, Y = y_5)

        # get the interpolated Ksi up and down forms of above fuzzies
        Ksi_1_u, Ksi_1_d = FuzzyNumber().manual_fuzzy(F1_X, F1_Y)
        Ksi_2_u, Ksi_2_d = FuzzyNumber().manual_fuzzy(F2_X, F2_Y)
        Ksi_3_u, Ksi_3_d = FuzzyNumber().manual_fuzzy(F3_X, F3_Y)

        list_of_fuzzies = [[Ksi_1_u, Ksi_1_d], [Ksi_2_u, Ksi_2_d], [Ksi_3_u, Ksi_3_d]]

        # gather all the values of expected value (mean), skewness and variance
        E_values = []
        S_values = []
        var_values = []

        for weight_combination in weights:
            
            Ksi_u, Ksi_d, l, m, r = self.multiple_sum(fuzzies=list_of_fuzzies, weights=weight_combination, plot=False)
            
            mean, skewness, var = self.calculate_skewness(fuzzy = [Ksi_u, Ksi_d], version=skewness_version, mean_skewness_var=True)

            E_values.append(mean)
            S_values.append(skewness)
            var_values.append(var)        

        if version == "S":
            alpha = np.mean(E_values) * .7 # E >= alpha
            beta = np.mean(var_values) * 1.3 # var <= beta

            new_S = []
            new_w = []

            for i in range(len(S_values)):
                if E_values[i] >= alpha and var_values[i] <= beta:
                    new_S.append(S_values[i])
                    new_w.append(weights[i])
        
            new_max = max(new_S)
            new_max_index = new_S.index(new_max)
            new_max_w = new_w[new_max_index]

        elif version =="E":
            beta = np.mean(var_values) * 1.3 # var <= beta
            gamma = np.mean(S_values) * 1.1 # S >= gamma

            new_E = []
            new_w = []

            for i in range(len(E_values)):
                if S_values[i] >= gamma and var_values[i] <= beta:
                    new_E.append(E_values[i])
                    new_w.append(weights[i])

            new_max = max(new_E)
            new_max_index = new_E.index(new_max)
            new_max_w = new_w[new_max_index]
        
        elif version == "var":
            alpha = np.mean(E_values) * .7 # E >= alpha
            gamma = np.mean(S_values) * 1.1 # S >= gamma

            new_var = []
            new_w = []

            for i in range(len(var_values)):
                if S_values[i] >= gamma and E_values[i] >= alpha:
                    new_var.append(var_values[i])
                    new_w.append(weights[i])

            new_max = max(new_var)
            new_max_index = new_var.index(new_max)
            new_max_w = new_w[new_max_index]


        end = datetime.now()

        time = (end - start).total_seconds()

        with open(f"portfolio_opt_ver{version_opt}.txt", 'a') as f:
            f.write(f"{skewness_version} (alpha = {self.alpha}) --> {np.round(new_max,3)}, {new_max_w}, {np.round(time,2)} s\n")
        print(skewness_version, " --> ", new_max, new_max_w, time)
    
    def time_analysis(self, L=2, M=6, R=10):

        df_time_analysis_S1 = pd.DataFrame()
        df_time_analysis_S2 = pd.DataFrame()
        df_time_analysis_S3 = pd.DataFrame()

        N_tab = range(100, 1100, 100)

        for ii in range(1):

            S1_time = []
            S2_time = []
            S3_time =[]

            for N in N_tab:

                start = time.perf_counter()

                for _ in range(N):

                    M = np.random.randint(3,10)
                    X, Y = self.make_fuzzy(L, M, R)
                    Ksi_u, Ksi_d = self.manual_fuzzy(X, Y)
                    self.calculate_skewness(fuzzy=[Ksi_u, Ksi_d], version=1)
                
                end = time.perf_counter()

                S1_time.append(np.round(end - start))

                start = time.perf_counter()

                for _ in range(N):

                    M = np.random.randint(3,10)
                    X, Y = self.make_fuzzy(L, M, R)
                    Ksi_u, Ksi_d = self.manual_fuzzy(X, Y)
                    self.calculate_skewness(fuzzy=[Ksi_u, Ksi_d], version=2)
                
                end = time.perf_counter()

                S2_time.append(np.round(end - start))

                start = time.perf_counter()

                for _ in range(N):

                    M = np.random.randint(3,10)
                    X, Y = self.make_fuzzy(L, M, R)
                    Ksi_u, Ksi_d = self.manual_fuzzy(X, Y)
                    self.calculate_skewness(fuzzy=[Ksi_u, Ksi_d], version=3)
                
                end = time.perf_counter()

                S3_time.append(np.round(end - start))

            df_time_analysis_S1[ii] = S1_time
            df_time_analysis_S2[ii] = S2_time
            df_time_analysis_S3[ii] = S3_time

        df_time_analysis_S1.to_csv("fuzzy_time_analysis_S1.csv", index=False)
        df_time_analysis_S1.to_csv("fuzzy_time_analysis_S2.csv", index=False)
        df_time_analysis_S1.to_csv("fuzzy_time_analysis_S3.csv", index=False)

# for m in range(3,10):
# for m in range(103,110):
for m in range(300,1000,100):
    fn = FuzzyNumber()
    x, y = fn.make_fuzzy(L=2, M=m, R=10, number_of_nodes_sym=1)
    ksi_u, ksi_d = fn.manual_fuzzy(X=x, Y=y)
    fn.calculate_skewness([ksi_u,ksi_d], version=3)
    print(np.round(fn.skewness_v3,2))

N = 101

FuzzyNumber().portfolio_optimization(n=3, skewness_version=1, N=N)
FuzzyNumber().portfolio_optimization(n=3, skewness_version=2, N=N)
FuzzyNumber().portfolio_optimization(n=3, skewness_version=3, N=N, alpha = 0.1)
FuzzyNumber().portfolio_optimization(n=3, skewness_version=3, N=N, alpha = 0.25)
FuzzyNumber().portfolio_optimization(n=3, skewness_version=4, N=N)
