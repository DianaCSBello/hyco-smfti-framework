# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import eigh
import joblib

# ============================================================
# 1. Funciones auxiliares (para calcular el TB base)
# ============================================================
def compute_metrics_guides(c):
    I = np.abs(c)**2
    I_norm = I / np.sum(I)
    IPR = np.sum(I_norm**2)
    indices = np.arange(len(c))
    COM = np.sum(indices * I_norm)
    phases = np.angle(c)
    MCD = np.sum((indices - COM) * phases * I_norm)
    Edge = I_norm[0]
    return IPR, COM, MCD, Edge

def simulate_tb_param(N, params, Z_vals, omega1_val):
    t0, t1, t2, t3, u0, u1, u2, u3, e0, e1, e2, e3, a1, a2, b1, b2, *rest = params
    omega1 = omega1_val
    omega2 = 2.0 * omega1_val
    dZ = Z_vals[1] - Z_vals[0]
    phi = np.zeros((len(Z_vals), N), dtype=complex)
    phi[0, 0] = 1.0
    for k in range(len(Z_vals)-1):
        Z = Z_vals[k]
        t = t0 + t1*np.cos(omega1*Z) + t2*np.cos(omega2*Z) + t3*Z
        u = u0 + u1*np.cos(omega1*Z) + u2*np.cos(omega2*Z) + u3*Z
        e = e0 + e1*np.cos(omega1*Z) + e2*np.cos(omega2*Z) + e3*Z
        phase = a1*np.cos(omega1*Z) + a2*np.cos(omega2*Z) + b1*Z + b2*Z**2
        H = np.zeros((N, N), dtype=complex)
        for n in range(N-1):
            H[n, n+1] = -t * np.exp(1j*phase)
            H[n+1, n] = -t * np.exp(-1j*phase)
        for n in range(N-2):
            H[n, n+2] = -u
            H[n+2, n] = -u
        for n in range(N):
            H[n, n] = e
        I = np.eye(N)
        A = I + 1j * H * dZ / 2
        B = I - 1j * H * dZ / 2
        phi[k+1] = np.linalg.solve(A, B @ phi[k])
    return phi

# ============================================================
# 2. Cargar el modelo y el escalador guardados
# ============================================================
modelo = joblib.load('modelo_hibrido_entrenado.pkl')
scaler = joblib.load('scaler_entrenado.pkl')

# Parámetros óptimos del TB (los mismos que usaste para entrenar)
params_opt = np.array([
    3.110123, -0.437199, 0.215664, -0.478449,
    -0.226969, 0.481867, -0.374918, 0.451723,
    -0.405352, 0.400132, 0.099560, -0.380682,
    3.924851, 1.797626, -0.535561, 0.067966,
    0, 0, 0, 0, 0, 0, 0
])

# ============================================================
# 3. Definir los parámetros de la NUEVA simulación
# ============================================================
A1_pred = 0.08
A2_pred = 0.12
omega1_pred = 9.5
N_guides = 20

# Generar el mismo rango de Z que en FreeFem++ (dZ = 0.05)
Zmax = 8.0 * np.pi / omega1_pred
Z_vals = np.arange(0, Zmax + 0.05, 0.05)
print(f"Prediciendo simulación para A1={A1_pred}, A2={A2_pred}, om1={omega1_pred}...")

# ============================================================
# 4. Calcular el TB base para esta nueva combinación
# ============================================================
c_TB = simulate_tb_param(N_guides, params_opt, Z_vals, omega1_pred)
metrics_TB = np.array([compute_metrics_guides(c) for c in c_TB])

# ============================================================
# 5. Construir las características para el MLP
# ============================================================
omega2_pred = 2.0 * omega1_pred
X_new = np.column_stack((
    Z_vals,
    np.full_like(Z_vals, A1_pred),
    np.full_like(Z_vals, A2_pred),
    np.full_like(Z_vals, omega1_pred),
    metrics_TB[:, 0],  # IPR_TB
    metrics_TB[:, 1],  # COM_TB
    metrics_TB[:, 2],  # MCD_TB
    metrics_TB[:, 3],  # Edge_TB
    np.sin(omega1_pred * Z_vals),
    np.cos(omega1_pred * Z_vals),
    np.sin(omega2_pred * Z_vals),
    np.cos(omega2_pred * Z_vals)
))

# ============================================================
# 6. Predecir el residuo y obtener la métrica final híbrida
# ============================================================
X_new_scaled = scaler.transform(X_new)
res_predicho = modelo.predict(X_new_scaled)
metrics_hibridas = metrics_TB + res_predicho

# ============================================================
# 7. Mostrar resultados
# ============================================================
metric_names = ['IPR', 'COM', 'MCD', 'Edge']
print("\nResultado promedio de la predicción híbrida:")
for i, name in enumerate(metric_names):
    print(f"{name} promedio: {np.mean(metrics_hibridas[:, i]):.5f}")

print("\n¡Predicción completada en milisegundos!")
np.savetxt(f"prediccion_A1_{A1_pred}_A2_{A2_pred}_om1_{omega1_pred}.dat", 
           np.column_stack((Z_vals, metrics_hibridas)), 
           header="Z IPR COM MCD Edge", comments='')
print(f"Datos guardados en prediccion_A1_{A1_pred}_A2_{A2_pred}_om1_{omega1_pred}.dat")
