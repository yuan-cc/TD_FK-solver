import numpy as np
import matplotlib.pyplot as plt
from Fokker_Planck import FK_TriDiagonalMatrix
import matplotlib.cm as cm
import matplotlib.colors as mcolors

FKT = FK_TriDiagonalMatrix()

proton_momentum = FKT.p_grid # momentum array, proton_momentum * light speed (3e10) = proton energy in ergs!
print(len(proton_momentum))          

p_pk_c = 1e7 * FKT.mp * FKT.c0
tacc = 1e6 * (proton_momentum / p_pk_c)**1
tcool = 1e4 * (proton_momentum / p_pk_c)**-1
t_esc = 2e5 + np.zeros_like(proton_momentum)

injection = np.zeros_like(proton_momentum)

E_pk = 100 # 1.0e3 m_p*c^2,  p_pk = 1.0e2 x m_p*c
erg2GeV = 624.0
c0 = 3.0e10

### set the injection function q_p, arbitrary units
p_pk = E_pk / erg2GeV / c0
injection = np.zeros_like(proton_momentum) 

for i in range(len(injection)):
    if proton_momentum[i] <= p_pk * (1.1) and proton_momentum[i] >= p_pk /1.1:
        injection[i]=1


#delta_t = min(0.01 * t_esc[0], 0.1 * tcool[-1])
delta_t = 0.01 * t_esc[0] 
runtime = 0.0




#t_list = np.array([0.1, 0.2, 0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1, 2, 3, 4, 5, 6, 7,8, 9, 10])* t_esc[0] 
t_list = 10**np.arange(np.log10(0.1*t_esc[0]), np.log10(10*t_esc[0])+0.05, 0.05) ## for making the figure

spec =[proton_momentum, injection]

while runtime < 10.2 * t_esc[0]:
    # update the input arrays
    FKT.InjectionArray = injection # dot q
    FKT.DiffusionArray = proton_momentum**2 / tacc # D_pp = p^2 / t_acc
    FKT.EscapeArray = t_esc # t_esc in s
    FKT.CoolingArray =   proton_momentum / tcool # dot p
    
    ## evolut the time dependent FP equation by delta_t
    FKT.FP_T_evolve(delta_t)
    
    for j in t_list:
        if runtime < j and runtime + delta_t >= j:
            spec.append(FKT.FinalSpec) # for the figure
            
    FKT.InitialSpec = FKT.FinalSpec # assign the evolved proton spectra to the initial distribution for next evolution
    runtime += delta_t
    

FKT.FP_S_evolve() # find the steady-state solution using the existing FKT.InjectionArray, FKT.DiffusionArray, FKT.EscapeArray, FKT.CoolingArray
                  # accelerated spectra: FKT.FinalSpecS


LuminosityDensity =  1500 # an arbrary value to make the spectra look beautiful, erg/s/cm^3

# normalization
Norm1 = FKT.normalize_factor(LuminosityDensity, FKT.FinalSpec)
Norm2 = FKT.normalize_factor(LuminosityDensity, FKT.FinalSpecS)


### Make a fancy figure

data = np.transpose(spec)
X_values = t_list / t_esc[0]

cmap = cm.inferno_r
log_norm = mcolors.LogNorm(vmin=X_values.min(), vmax=X_values.max())
data2 = FKT.FinalSpecS


fig, ax = plt.subplots(figsize=(8, 5)) 
ax.loglog()
for i in range(len(X_values)):
    ax.plot(data[:,0] * 3e10 * 624, data[:,i+1] * data[:,0]**3 * 4 * 3.14 * Norm1, lw = 1, color=cmap(log_norm(X_values[i])))
    
sm = cm.ScalarMappable(cmap=cmap, norm=log_norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label(r'$t/t_{\rm esc}$', fontsize=14)

ax.plot(data[:,0] * 3e10 * 624, data2 * data[:,0]**3* 4 * 3.14 * Norm2, "--", lw = 2, color = "red",  label = "Steady-state (ODE)") 
ax.plot(data[:,0] * 3e10 * 624, injection * data[:,0]**3 * 4 * 3.14 * Norm1 * t_esc[0], "-.", lw = 2, color = "blue", label = r"Injection: $q\times t_{\rm esc}$")

ax.grid(color="gray", alpha = 0.1, which="minor")
ax.grid(color="gray", alpha = 0.2, which="major")
ax.set_xlabel(r"$E_p$ [GeV]", fontsize=14)
ax.set_ylabel(r"$EdN/dE/dV~[\rm a.u.]$ ", fontsize=14)
ax.set_ylim(1e-1,4e9)
ax.legend(fontsize=12)
ax.set_xlim(5,1e8)
ax.set_title("Test case: time-dependent v.s. steady-state", fontsize = 12)
ax.tick_params(axis='both', which='major', labelsize=14)  


inset_pos = [0.21, 0.67, 0.3, 0.25] # position of the inset
ax_inset = fig.add_axes(inset_pos)
ax_inset.plot(data[:,0] * 3e10 * 624, 1.0/tcool, lw = 2,color = "blue",  label = r"$t_{\rm cool}^{-1}$")
ax_inset.plot(data[:,0] * 3e10 * 624, 1.0/tacc, lw = 2, color = "green", label = r"$t_{\rm acc}^{-1}$")
ax_inset.plot(data[:,0] * 3e10 * 624, 1.0/t_esc, "--", lw=2, color = "black", label = r"$t_{\rm esc}^{-1}$")

ax_inset.set_yscale('log')
ax_inset.set_xscale('log')

ax_inset.text(30, 9e-6, "Escape", color = "black", fontsize=9)
ax_inset.text(1e6, 5e-5, "Cooling", color = "blue", fontsize=9, rotation = 38)
ax_inset.text(5e2, 2e-5, "Acceleration", color = "green", fontsize=9, rotation = -36)


ax_inset.tick_params(labelsize=9)
ax_inset.set_xlabel(r"$E_p$ [GeV]", fontsize=10)
ax_inset.set_ylabel(r"$t^{-1}~[s^{-1}]$", fontsize=10)
ax_inset.set_xlim(1e1, 1e8)   
ax_inset.set_ylim(1e-7,1e-2)    

fig.tight_layout()

plt.savefig("Test_p_spec_inj.pdf")


"""

plt.figure(figsize = (8,6))
plt.loglog()

plt.plot(data[:,0] * 3e10 * 624, 1.0/tcool, label = r"$t_{\rm cool}^{-1}$")
plt.plot(data[:,0] * 3e10 * 624, 1.0/tacc, label = r"$t_{\rm acc}^{-1}$")
plt.plot(data[:,0] * 3e10 * 624, 1.0/t_esc, label = r"$t_{\rm esc}^{-1}$")
plt.grid(color="gray", alpha = 0.1, which="minor")
plt.grid(color="gray", alpha = 0.2, which="major")
plt.xlabel(r"$E_p$ [GeV]", fontsize=12)
plt.ylabel(r"$t^{-1}~[s^{-1}]$", fontsize=12)
plt.ylim(1e-7,1e2)
plt.legend()
plt.xlim(10,1e9)
plt.tight_layout()
plt.savefig("Rate.pdf")
"""

plt.show()

