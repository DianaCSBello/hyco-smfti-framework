import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# ============================================================
# FUNCIONES AUXILIARES (lectura, Löwdin, métricas)
# ============================================================
def read_floats(f, n):
    values = []
    while len(values) < n:
        line = f.readline()
        if not line: break
        parts = line.strip().split()
        if not parts: continue
        values.extend([float(x) for x in parts])
    return values

def load_all_steps(filename):
    steps = []
    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if not line: break
            parts = line.strip().split()
            if len(parts) != 3: continue
            Z = float(parts[0])
            normPsi2 = float(parts[1])
            N = int(parts[2])
            S_flat = read_floats(f, N*N)
            if len(S_flat) < N*N: break
            S = np.array(S_flat).reshape(N, N)
            uR = np.array(read_floats(f, N))
            uI = np.array(read_floats(f, N))
            u = uR + 1j * uI
            steps.append({'Z': Z, 'S': S, 'u': u})
    return steps

def lowdin_orthogonalize(S, u):
    vals, vecs = eigh(S)
    vals_safe = np.maximum(vals, 1e-10)
    S_inv_sqrt = vecs @ np.diag(1.0/np.sqrt(vals_safe)) @ vecs.T
    return S_inv_sqrt @ u

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

# ============================================================
# MODELO TB DE 23 PARÁMETROS (óptimo)
# ============================================================
def simulate_tb_param(N, params, Z_vals):
    t0, t1, t2, t3, u0, u1, u2, u3, e0, e1, e2, e3, a1, a2, b1, b2, *rest = params
    omega1, omega2 = 10.0, 20.0
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
# CARGA DE DATOS Y CONCATENACIÓN
# ============================================================
file1 = "overlap_data_pml_laude_final.dat"
file2 = "overlap_data_pml_laude_second.dat"

steps1 = load_all_steps(file1)
steps2 = load_all_steps(file2)
if not steps1 or not steps2:
    raise FileNotFoundError("No se encontraron los archivos de datos.")

steps = steps1 + steps2
steps.sort(key=lambda s: s['Z'])
Z_vals = np.array([s['Z'] for s in steps])
Z_vals, idx = np.unique(Z_vals, return_index=True)
steps = [steps[i] for i in idx]
Z_vals = np.array([s['Z'] for s in steps])
N_guides = steps[0]['S'].shape[0]
c_FEM = np.array([lowdin_orthogonalize(s['S'], s['u']) for s in steps])

# ============================================================
# PARÁMETROS ÓPTIMOS (Differential Evolution)
# ============================================================
params_opt = np.array([
    3.110123, -0.437199, 0.215664, -0.478449,
    -0.226969, 0.481867, -0.374918, 0.451723,
    -0.405352, 0.400132, 0.099560, -0.380682,
    3.924851, 1.797626, -0.535561, 0.067966,
    0, 0, 0, 0, 0, 0, 0
])

# ============================================================
# SIMULAR TB ÓPTIMO Y OBTENER MÉTRICAS
# ============================================================
c_TB = simulate_tb_param(N_guides, params_opt, Z_vals)
metrics_FEM = np.array([compute_metrics_guides(c) for c in c_FEM])
metrics_TB = np.array([compute_metrics_guides(c) for c in c_TB])

# ============================================================
# PREPARAR CARACTERÍSTICAS PARA LOS MODELOS RESIDUALES
# ============================================================
omega1, omega2 = 10.0, 20.0
features = np.column_stack((
    Z_vals,
    metrics_TB[:, 0],
    metrics_TB[:, 1],
    metrics_TB[:, 2],
    metrics_TB[:, 3],
    np.sin(omega1 * Z_vals),
    np.cos(omega1 * Z_vals),
    np.sin(omega2 * Z_vals),
    np.cos(omega2 * Z_vals)
))

# Residuos (targets)
residuals = metrics_FEM - metrics_TB

# Normalizar características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# ============================================================
# ENTRENAR MLP PARA CADA MÉTRICA (MODELOS SEPARADOS)
# ============================================================
metric_names = ['IPR', 'COM', 'MCD', 'Edge']
mlp_models = []
predictions_mlp = []

print("Entrenando MLP residual para cada métrica...")
for i in range(4):
    mlp = MLPRegressor(
        hidden_layer_sizes=(100, 100, 50),
        activation='relu',
        solver='adam',
        max_iter=2000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    mlp.fit(X_scaled, residuals[:, i])
    mlp_models.append(mlp)
    pred_mlp = mlp.predict(X_scaled)
    predictions_mlp.append(pred_mlp)

print("Entrenamiento completado.")

# ============================================================
# CALCULAR RMSE DEL MODELO HÍBRIDO (TB + MLP residual)
# ============================================================
final_pred = metrics_TB + np.column_stack(predictions_mlp)
rmse_hybrid = [
    np.sqrt(mean_squared_error(metrics_FEM[:, i], final_pred[:, i]))
    for i in range(4)
]

print("\n" + "="*60)
print("   RMSE CON MODELO HÍBRIDO AVANZADO (MLP residual, características enriquecidas)")
print("="*60)
print(f"IPR: {rmse_hybrid[0]:.6f}")
print(f"COM: {rmse_hybrid[1]:.6f}")
print(f"MCD: {rmse_hybrid[2]:.6f}")
print(f"Edge: {rmse_hybrid[3]:.6f}")
print(f"Suma RMSE: {sum(rmse_hybrid):.6f}")
print("="*60)

# ============================================================
# GRÁFICAS DEL MODELO HÍBRIDO
# ============================================================
plt.figure(figsize=(12, 10))

for i, name in enumerate(metric_names):
    plt.subplot(4, 1, i+1)
    plt.plot(Z_vals, metrics_FEM[:, i], 'b-', label='FEM (real)')
    plt.plot(Z_vals, final_pred[:, i], 'r--', label='Hybrid (TB + MLP)')
    plt.ylabel(name)
    plt.legend()
    plt.grid(True)
    if i == 3:
        plt.xlabel('Z')

plt.tight_layout()
plt.savefig('hybrid_mlp_advanced_results.pdf', dpi=300)
plt.show()

print("¡Proceso completado! Gráficas guardadas en hybrid_mlp_advanced_results.pdf")
