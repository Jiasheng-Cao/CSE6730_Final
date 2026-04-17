import numpy as np
import matplotlib.pyplot as plt

params = {
    'mu_AOB_max': 0.38,
    'mu_NOB_max': 0.405,
    'K_NH4': 1.0,
    'K_O2_AOB': 0.5,
    'K_NO2': 1.3,
    'K_O2_NOB': 0.68,
    'theta': 1.07,
    'SO2': 2.0,
    'Y_AOB': 0.33,
    'Y_NOB': 0.083,
    'b_AOB': 0.11,
    'b_NOB': 0.11,
    'V_Q': 10.0,
    'kLa': 300.0,
    'S_O2_star': 8.0,
}

def system_dynamics(y, t, p):
    S_NH4, S_NO2, S_NO3, S_O2, X_AOB, X_NOB = y

    S_O2_eff = max(S_O2, 1e-6)

    rho_AOB = p['mu_AOB_max'] * (S_NH4 / (p['K_NH4'] + S_NH4)) * (S_O2_eff / (p['K_O2_AOB'] + S_O2_eff)) * X_AOB
    rho_NOB = p['mu_NOB_max'] * (S_NO2 / (p['K_NO2'] + S_NO2)) * (S_O2_eff / (p['K_O2_NOB'] + S_O2_eff)) * X_NOB

    S_NH4_in = 16.8

    dNH4 = (1 / p['V_Q']) * (S_NH4_in - S_NH4) - (1 / p['Y_AOB']) * rho_AOB
    dNO2 = (1 / p['V_Q']) * (0 - S_NO2) + (1 / p['Y_AOB']) * rho_AOB - (1 / p['Y_NOB']) * rho_NOB
    dNO3 = (1 / p['V_Q']) * (0 - S_NO3) + (1 / p['Y_NOB']) * rho_NOB

    dX_AOB = rho_AOB - p['b_AOB'] * X_AOB - (1 / p['V_Q']) * X_AOB
    dX_NOB = rho_NOB - p['b_NOB'] * X_NOB - (1 / p['V_Q']) * X_NOB
    
    r_AOB = rho_AOB / p['Y_AOB']
    r_NOB = rho_NOB / p['Y_NOB']
    dO2 = p['kLa'] * (p['S_O2_star'] - S_O2) - 3.43 * r_AOB - 1.14 * r_NOB

    return np.array([dNH4, dNO2, dNO3, dO2, dX_AOB, dX_NOB])

# 100 day
t = np.linspace(0, 100, 50000)
# y0 = [NH4, NO2, NO3, O2, X_AOB, X_NOB]
y0 = [16.8, 0.0, 0.0, 2.0, 20.0, 10.0]

def rk4_step_sys(y, t, dt, p):
    k1 = system_dynamics(y, t, p)
    k2 = system_dynamics(y + dt/2 * k1, t + dt/2, p)
    k3 = system_dynamics(y + dt/2 * k2, t + dt/2, p)
    k4 = system_dynamics(y + dt * k3, t + dt, p)

    return y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)


dt = t[1] - t[0]

y = np.array(y0)
sol = []

for ti in t:
    sol.append(y.copy())
    y = rk4_step_sys(y, ti, dt, params)

sol = np.array(sol)

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(t, sol[:, 0], label='NH4-N', color='blue')
ax1.plot(t, sol[:, 1], label='NO2-N', color='orange')
ax1.plot(t, sol[:, 2], label='NO3-N', color='green')
ax1.set_xlabel('Time (days)')
ax1.set_ylabel('Nitrogen (mg N/L)')
ax1.legend(loc='upper left')
ax1.grid(True)

ax1.set_xscale('log')

ax2 = ax1.twinx()
ax2.plot(t, sol[:, 3], '--', label='DO', color='red')
ax2.set_ylabel('DO (mg O2/L)')
ax2.legend(loc='upper right')

plt.title('Nitrogen Transformation + DO Dynamics (Log Time)')
plt.show()


fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(t, sol[:, 0], label='NH4-N', color='blue')
ax1.plot(t, sol[:, 1], label='NO2-N', color='orange')
ax1.plot(t, sol[:, 2], label='NO3-N', color='green')
ax1.set_xlabel('Time (days)')
ax1.set_ylabel('Nitrogen (mg N/L)')
ax1.set_xscale('log')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.plot(t, sol[:, 3], '--', label='DO', color='red')
ax2.set_ylabel('DO (mg O2/L)')

ax3 = ax1.twinx()
ax3.spines["right"].set_position(("outward", 60))
ax3.plot(t, sol[:, 4], ':', label='X_AOB', color='purple')
ax3.plot(t, sol[:, 5], ':', label='X_NOB', color='brown')
ax3.set_ylabel('Biomass (mg/L)')

lines = ax1.get_lines() + ax2.get_lines() + ax3.get_lines()
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='best')

plt.title('Nitrogen + DO + Biomass Dynamics (Log Time)')
plt.show()