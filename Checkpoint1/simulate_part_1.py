import numpy as np
import matplotlib.pyplot as plt

mu_AOB_max = 0.76  # /day
mu_NOB_max = 0.81  # /day
K_NH4 = 1.0  # mg N/L
K_NO2 = 1.3  # mg N/L
K_O2_AOB = 0.5  # mg O2/L
K_O2_NOB = 0.68  # mg O2/L
b_AOB = 0.11  # /day
b_NOB = 0.11  # /day
V_Q = 10.0  # HRT (days)
S_O2 = 2.0  # mg/L


def biomass_dynamics(y, t):
    S_NH4, S_NO2, X_AOB, X_NOB = y

    growth_AOB = mu_AOB_max * (S_NH4 / (K_NH4 + S_NH4)) * (S_O2 / (K_O2_AOB + S_O2)) * X_AOB
    growth_NOB = mu_NOB_max * (S_NO2 / (K_NO2 + S_NO2)) * (S_O2 / (K_O2_NOB + S_O2)) * X_NOB

    dX_AOB = growth_AOB - b_AOB * X_AOB - (1 / V_Q) * X_AOB
    dX_NOB = growth_NOB - b_NOB * X_NOB - (1 / V_Q) * X_NOB

    dS_NH4 = (1 / V_Q) * (40 - S_NH4) - (1 / 0.33) * growth_AOB
    dS_NO2 = (1 / V_Q) * (0 - S_NO2) + (1 / 0.33) * growth_AOB - (1 / 0.083) * growth_NOB

    return [dS_NH4, dS_NO2, dX_AOB, dX_NOB]


t = np.linspace(0, 100, 5000)  # 100 days
y0 = [40, 0, 20, 10]  # [NH4, NO2, X_AOB, X_NOB]

# RK4
def rk4_step(y, t, dt):
    k1 = np.array(biomass_dynamics(y, t))
    k2 = np.array(biomass_dynamics(y + dt/2 * k1, t + dt/2))
    k3 = np.array(biomass_dynamics(y + dt/2 * k2, t + dt/2))
    k4 = np.array(biomass_dynamics(y + dt * k3, t + dt))

    return y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

dt = t[1] - t[0]

y = np.array(y0)
solution = []

for ti in t:
    solution.append(y.copy())
    y = rk4_step(y, ti, dt)

solution = np.array(solution)

plt.figure(figsize=(10, 5))
plt.plot(t, solution[:, 2], 'b-', label='X_AOB (Biomass)')
plt.plot(t, solution[:, 3], 'g-', label='X_NOB (Biomass)')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Biomass Simulation (AOB & NOB)')
plt.xlabel('Days')
plt.ylabel('Concentration (mg/L)')
plt.legend()
plt.grid(True)
plt.show()


params = {
    'mu_AOB_max': 0.76, # /day
    'mu_NOB_max': 0.81, # /day
    'K_NH4': 1.0,       # mg N/L
    'K_O2_AOB': 0.5,    # mg O2/L
    'K_NO2': 1.3,       # mg N/L
    'K_O2_NOB': 0.68,   # mg O2/L
    'theta': 1.07,
    'SO2': 2.0,         # mg O2/L
    'Y_AOB': 0.33,      # mg VSS/mg N
    'Y_NOB': 0.083,     # mg VSS/mg N
    'b_AOB': 0.11,      # /day
    'b_NOB': 0.11,      # /day
    'V_Q': 10.0         # (V/Q) days
}


def system_dynamics(y, t, p):
    S_NH4, S_NO2, S_NO3, X_AOB, X_NOB = y

    rho_AOB = p['mu_AOB_max'] * (S_NH4 / (p['K_NH4'] + S_NH4)) * (p['SO2'] / (p['K_O2_AOB'] + p['SO2'])) * X_AOB
    rho_NOB = p['mu_NOB_max'] * (S_NO2 / (p['K_NO2'] + S_NO2)) * (p['SO2'] / (p['K_O2_NOB'] + p['SO2'])) * X_NOB

    S_NH4_in = 40.0  # mg/L

    dNH4 = (1 / p['V_Q']) * (S_NH4_in - S_NH4) - (1 / p['Y_AOB']) * rho_AOB
    dNO2 = (1 / p['V_Q']) * (0 - S_NO2) + (1 / p['Y_AOB']) * rho_AOB - (1 / p['Y_NOB']) * rho_NOB
    dNO3 = (1 / p['V_Q']) * (0 - S_NO3) + (1 / p['Y_NOB']) * rho_NOB

    dX_AOB = rho_AOB - p['b_AOB'] * X_AOB - (1 / p['V_Q']) * X_AOB
    dX_NOB = rho_NOB - p['b_NOB'] * X_NOB - (1 / p['V_Q']) * X_NOB

    return np.array([dNH4, dNO2, dNO3, dX_AOB, dX_NOB])


t = np.linspace(0, 5, 50000)  # 50 days
y0 = [40.0, 0.0, 0.0, 200.0, 100.0]  # [NH4, NO2, NO3, X_AOB, X_NOB]

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

plt.figure(figsize=(10, 6))
plt.plot(t, sol[:, 0], label='NH4-N (Ammonia)')
plt.plot(t, sol[:, 1], label='NO2-N (Nitrite)')
plt.plot(t, sol[:, 2], label='NO3-N (Nitrate)')
plt.xlabel('Time (days)')
plt.ylabel('Concentration (mg N/L)')
plt.title('Nitrogen Transformation Simulation')
plt.legend()
plt.grid(True)
plt.show()
