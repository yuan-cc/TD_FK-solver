import numpy as np
from scipy.linalg import solve

class FK_TriDiagonalMatrix:
    def __init__(self):
        
        self.beta = 0.5
        self.c0 = 3e10 # c
        self.mp = 1.6e-24 # gram
        self.N_decade = 25
        self.p_min = self.mp * self.c0 * 5
        self.p_max = 1e11 * self.p_min
        self.N_p = int(np.log10(self.p_max/self.p_min)) * self.N_decade
        
        self.p_grid = 10**np.arange(np.log10(self.p_min), np.log10(self.p_max), (np.log10(self.p_max) - np.log10(self.p_min))/(self.N_p - 1.0))
        self.xi_p=np.log(self.p_grid)

        self.InjectionArray = np.zeros_like(self.p_grid)
        self.CoolingArray = np.zeros_like(self.p_grid)
        self.DiffusionArray = np.zeros_like(self.p_grid)
        self.EscapeArray = np.zeros_like(self.p_grid)
        self.InitialSpec = np.zeros_like(self.p_grid)
        self.FinalSpec = np.zeros_like(self.p_grid)
        self.FinalSpecS = np.zeros_like(self.p_grid)

        
        self.R_array = np.zeros_like(self.p_grid)
        self.B_array = np.zeros_like(self.p_grid)
        self.A_array = np.zeros_like(self.p_grid)
        self.C_array = np.zeros_like(self.p_grid)
        self.RS_array = np.zeros_like(self.p_grid)
        self.BS_array = np.zeros_like(self.p_grid)
        self.AS_array = np.zeros_like(self.p_grid)
        self.CS_array = np.zeros_like(self.p_grid)

        
    def EvaluateCoefficient_T(self, delta_t):
        N_vec = len(self.p_grid) 
        a = 0
        b = 0
        c = 0
        d = 0

        for i in range(N_vec):
            xi_half_p = 0
            Di_half_p = 0
            dotpi_half_p = 0
            xi_half_m = 0
            Di_half_m = 0
            dotpi_half_m = 0
            delta_xi = 0
            w=0.0
            if i < N_vec - 1:
                xi_half_p =  (self.xi_p[i] + self.xi_p[i+1]) / 2.0 # p_i+1/2
                Di_half_p =  (self.DiffusionArray[i] + self.DiffusionArray[i+1]) / 2.0 # D_i+1/2
                dotpi_half_p = (self.CoolingArray[i] + self.CoolingArray[i+1]) / 2.0 # \dot q_i+1/2
                if i == 0:
                    delta_xi = self.xi_p[1] - self.xi_p[0]
                else:
                    delta_xi = (self.xi_p[i+1] - self.xi_p[i]) 
                    
                
                a = np.exp(xi_half_p) * Di_half_p / (self.p_grid[i]**3 * delta_xi**2 )

                b = np.exp(2 * xi_half_p) * dotpi_half_p / (self.p_grid[i]**3 * delta_xi)
                
            if i > 0:
                xi_half_m =  (self.xi_p[i] + self.xi_p[i-1]) / 2.0 # p_i-1/2
                Di_half_m =  (self.DiffusionArray[i] + self.DiffusionArray[i-1]) / 2.0 # D_i-1/2
                dotpi_half_m = (self.CoolingArray[i] + self.CoolingArray[i-1]) / 2.0 # \dot q_i-1/2
                if i == N_vec - 1:
                    delta_xi = self.xi_p[i] - self.xi_p[i-1]
                else:
                    delta_xi = (self.xi_p[i+1] - self.xi_p[i]) 
                
                c = np.exp(xi_half_m) * Di_half_m / (self.p_grid[i]**3 * delta_xi**2)
                d = np.exp(2*xi_half_m) * dotpi_half_m / (self.p_grid[i]**3 * delta_xi)
                
                
            ### Chang-Cooper weighting
            w = delta_xi * self.CoolingArray[i]/ self.DiffusionArray[i]
            if w < 1e-4:
                theta = 0.5
            if w > 10000:
                theta = 0
            else:
                theta = 1/w - 1/(np.exp(w)-1)

            delta = theta ## Chang-Cooper weighting factor, previously I wrote it as 'theta' to distinguish it from delta_t :-)
            
            self.A_array[i] = -delta_t / 2.0 * (c - self.beta * d * 2 * delta)
            self.B_array[i] = 1.0 + delta_t / 2.0 * (a - self.beta * b * delta * 2 + c + self.beta * d * 2 * (1-delta) + 1.0 / self.EscapeArray[i])
            self.C_array[i] = -delta_t / 2.0 * (a + self.beta * b * 2 * (1-delta)) 
            if i == 0: 
                self.R_array[i] = (1 - delta_t / 2.0 /self.EscapeArray[i]) * self.InitialSpec[i] + delta_t / 2.0 * (a * (self.InitialSpec[i+1] - self.InitialSpec[i]) + (1-self.beta) * b * 2 * (self.InitialSpec[i+1] * (1-delta) + self.InitialSpec[i] * delta)) + self.InjectionArray[i] * delta_t
            if i == N_vec - 1:
                self.R_array[i] = (1 - delta_t / 2.0 /self.EscapeArray[i]) * self.InitialSpec[i] + delta_t / 2.0 * ( - c * (self.InitialSpec[i] - self.InitialSpec[i-1]) - (1-self.beta) * d * 2 * (self.InitialSpec[i] * (1-delta) + self.InitialSpec[i-1] * delta)) + self.InjectionArray[i] * delta_t
            else:
                self.R_array[i] = (1 - delta_t / 2.0 /self.EscapeArray[i]) * self.InitialSpec[i] + delta_t / 2.0 * (a * (self.InitialSpec[i+1] - self.InitialSpec[i]) + (1-self.beta) * b * 2 * (self.InitialSpec[i+1] * (1-delta) + self.InitialSpec[i] * delta) - c * (self.InitialSpec[i] - self.InitialSpec[i-1]) - (1-self.beta) * d * 2 * (self.InitialSpec[i] * (1-delta) + self.InitialSpec[i-1]* delta)) + self.InjectionArray[i] * delta_t
        

     
    def FP_T_evolve(self, delta_t):
        ##### Fully time dependent treatment
        self.EvaluateCoefficient_T(delta_t)
        BB = self.B_array + 1e-50 # main diagonal
        AA = self.A_array[1:] + 1e-50
        CC = self.C_array[:-1] + 1e-50
        MatrixM = np.diag(BB) + np.diag(AA, k = -1) + np.diag(CC, k = 1)
        II = np.eye(len(BB))

        MatrixM_inv = solve(MatrixM, II)
        self.FinalSpec = np.dot(MatrixM_inv, self.R_array)     
        self.FinalSpec[self.FinalSpec<0]=0    
    
 
 
    def EvaluateCoefficient_S(self):
        N_vec = len(self.p_grid) 
        a = 0
        b = 0
        c = 0
        d = 0
        
        for i in range(N_vec):
            xi_half_p = 0
            Di_half_p = 0
            dotpi_half_p = 0
            xi_half_m = 0
            Di_half_m = 0
            dotpi_half_m = 0
            delta_xi = 0
            a0 = 0
            w=0.0
            if i < N_vec - 1:
                xi_half_p =  (self.xi_p[i] + self.xi_p[i+1]) / 2.0 # p_i+1/2
                Di_half_p =  (self.DiffusionArray[i] + self.DiffusionArray[i+1]) / 2.0 # D_i+1/2
                dotpi_half_p = (self.CoolingArray[i] + self.CoolingArray[i+1]) / 2.0 # \dot q_i+1/2
                if i == 0:
                    delta_xi = self.xi_p[1] - self.xi_p[0]
                else:
                    delta_xi = (self.xi_p[i+1] - self.xi_p[i]) 
                    
                    
                a = np.exp(xi_half_p) * Di_half_p / (self.p_grid[i]**3 * delta_xi**2 )
                if i ==0:
                    a0 = a
    
                b = np.exp(2 * xi_half_p) * dotpi_half_p / (self.p_grid[i]**3 * delta_xi)
                
            if i > 0:
                xi_half_m =  (self.xi_p[i] + self.xi_p[i-1]) / 2.0 # p_i-1/2
                Di_half_m =  (self.DiffusionArray[i] + self.DiffusionArray[i-1]) / 2.0 # D_i-1/2
                dotpi_half_m = (self.CoolingArray[i] + self.CoolingArray[i-1]) / 2.0 # \dot q_i-1/2
                if i == N_vec - 1:
                    delta_xi = self.xi_p[i] - self.xi_p[i-1]
                else:
                    delta_xi = (self.xi_p[i+1] - self.xi_p[i]) 
                
                c = np.exp(xi_half_m) * Di_half_m / (self.p_grid[i]**3 * delta_xi**2)
                d = np.exp(2*xi_half_m) * dotpi_half_m / (self.p_grid[i]**3 * delta_xi)
            
            w = delta_xi * self.CoolingArray[i]/ self.DiffusionArray[i]
            if w < 1e-4:
                theta = 0.5
            if w > 10000:
                theta = 0
            else:
                theta = 1/w - 1/(np.exp(w)-1)
            
            delta = theta ## Chang-Cooper weighting factor. Chang-Cooper weighting factor, previously I wrote it as 'theta' :-)
            
            self.AS_array[i] = c - delta * d
            self.BS_array[i] = -a + delta * b - c - (1-delta) * d - 1.0 / self.EscapeArray[i]
            self.CS_array[i] = a + (1 - delta) * b
            self.RS_array[i] = - self.InjectionArray[i]
        
        ### boundary conditions     
        ## No flux for p_min: D df/dp + dot p f = 0 for p_min
        ## Absorption for p_max: f = 0
        self.AS_array[0] = 0
        self.BS_array[0] = -a0 - self.CoolingArray[0] * (self.p_grid[1] - self.p_grid[0])
        self.CS_array[0] = a0 
        
        self.AS_array[N_vec - 1] = 0
        self.BS_array[N_vec - 1] = 1
        self.CS_array[N_vec - 1] = 0
        self.RS_array[N_vec - 1] = 0.0


    def FP_S_evolve(self):
        ##### steady state solution
        self.EvaluateCoefficient_S()
        BB = self.BS_array + 1e-50 # main diagonal
        AA = self.AS_array[1:] + 1e-50
        CC = self.CS_array[:-1] + 1e-50
        
        #print(len(BB),len(AA), len(CC))
        MatrixM = np.diag(BB) + np.diag(AA, k = -1) + np.diag(CC, k = 1)
        II = np.eye(len(BB))

        MatrixM_inv = solve(MatrixM, II)
        self.FinalSpecS = np.dot(MatrixM_inv, self.RS_array)


    
    def normalize_factor(self, LumiDensity, SpecInj):
        integral1 = 0.0
        for i in range(len(self.p_grid) - 1):
            integral1 += self.p_grid[i]**2 * self.DiffusionArray[i] * (SpecInj[i+1] - SpecInj[i]) 
        
        return 1.0 / (- 4* 3.14 * integral1 * self.c0) * LumiDensity
    
        
        
