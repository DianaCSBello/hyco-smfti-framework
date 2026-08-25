import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh

# ============================================================
# Funciones auxiliares (carga de datos, Löwdin, métricas)
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
# Modelo TB paramétrico (23 parámetros con DE)
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
# Carga de datos y preparación
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
metrics_FEM = np.array([compute_metrics_guides(c) for c in c_FEM])

# Parámetros óptimos del modelo DE
params_opt = np.array([
    3.110123, -0.437199, 0.215664, -0.478449,
    -0.226969, 0.481867, -0.374918, 0.451723,
    -0.405352, 0.400132, 0.099560, -0.380682,
    3.924851, 1.797626, -0.535561, 0.067966,
    0, 0, 0, 0, 0, 0, 0
])

# ============================================================
# Simular TB con DE y calcular el residuo
# ============================================================
c_TB_DE = simulate_tb_param(N_guides, params_opt, Z_vals)
metrics_TB_DE = np.array([compute_metrics_guides(c) for c in c_TB_DE])

residual = metrics_FEM - metrics_TB_DE

# ============================================================
# Entrenar MLP para predecir el residuo (baseline)
# ============================================================
X = Z_vals.reshape(-1, 1)
y_res = residual

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

mlp_res = MLPRegressor(
    hidden_layer_sizes=(100, 100, 50),
    max_iter=2000,
    random_state=42,
    early_stopping=True
)

print("Entrenando MLP para predecir el residuo...")
mlp_res.fit(X_scaled, y_res)

# ============================================================
# Predicción híbrida (TB + MLP_residuo)
# ============================================================
res_pred = mlp_res.predict(X_scaled)
metrics_hybrid = metrics_TB_DE + res_pred

# ============================================================
# Cálculo del RMSE híbrido
# ============================================================
rmse_IPR = np.sqrt(np.mean((metrics_FEM[:,0] - metrics_hybrid[:,0])**2))
rmse_COM = np.sqrt(np.mean((metrics_FEM[:,1] - metrics_hybrid[:,1])**2))
rmse_MCD = np.sqrt(np.mean((metrics_FEM[:,2] - metrics_hybrid[:,2])**2))
rmse_Edge = np.sqrt(np.mean((metrics_FEM[:,3] - metrics_hybrid[:,3])**2))

print("\n" + "="*60)
print("   RMSE CON MODELO HÍBRIDO (TB + MLP residual)")
print("="*60)
print(f"IPR: {rmse_IPR:.6f}")
print(f"COM: {rmse_COM:.6f}")
print(f"MCD: {rmse_MCD:.6f}")
print(f"Edge: {rmse_Edge:.6f}")
print(f"Suma RMSE: {rmse_IPR + rmse_COM + rmse_MCD + rmse_Edge:.6f}")
print("="*60)

# ============================================================
# Gráficas comparativas
# ============================================================
plt.figure(figsize=(12, 10))

plt.subplot(4,1,1)
plt.plot(Z_vals, metrics_FEM[:,0], 'b-', label='FEM')
plt.plot(Z_vals, metrics_hybrid[:,0], 'r--', label='Híbrido (TB+MLP)')
plt.ylabel('IPR'); plt.legend(); plt.grid(True)

plt.subplot(4,1,2)
plt.plot(Z_vals, metrics_FEM[:,1]/2.0, 'b-', label='FEM')
plt.plot(Z_vals, metrics_hybrid[:,1]/2.0, 'r--', label='Híbrido')
plt.ylabel('COM'); plt.legend(); plt.grid(True)

plt.subplot(4,1,3)
plt.plot(Z_vals, metrics_FEM[:,2], 'b-', label='FEM')
plt.plot(Z_vals, metrics_hybrid[:,2], 'r--', label='Híbrido')
plt.ylabel('MCD'); plt.legend(); plt.grid(True)

plt.subplot(4,1,4)
plt.plot(Z_vals, metrics_FEM[:,3], 'b-', label='FEM')
plt.plot(Z_vals, metrics_hybrid[:,3], 'r--', label='Híbrido')
plt.ylabel('Edge Intensity'); plt.xlabel('Z'); plt.legend(); plt.grid(True)

plt.tight_layout()
plt.savefig('hybrid_model_results.pdf', dpi=300)
plt.show()
