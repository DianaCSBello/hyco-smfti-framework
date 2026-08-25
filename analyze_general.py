import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.linalg import eigh
import glob
import os
import joblib

# ============================================================
# 1. Funciones auxiliares (carga, Löwdin, métricas)
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
# 2. Modelo TB paramétrico (dinámico con omega1)
# ============================================================
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
# 3. Parámetros óptimos del DE
# ============================================================
params_opt = np.array([
    3.110123, -0.437199, 0.215664, -0.478449,
    -0.226969, 0.481867, -0.374918, 0.451723,
    -0.405352, 0.400132, 0.099560, -0.380682,
    3.924851, 1.797626, -0.535561, 0.067966,
    0, 0, 0, 0, 0, 0, 0
])

# ============================================================
# 4. Carga de datos y extracción de características
# ============================================================
archivos = sorted(glob.glob("overlap_data_A1_*.dat"))
if not archivos:
    raise FileNotFoundError("No se encontraron archivos 'overlap_data_A1_*.dat' en la carpeta.")

print(f"Se encontraron {len(archivos)} archivos. Procesando...")

all_Z, all_A1, all_A2, all_omega1 = [], [], [], []
all_metrics_FEM, all_metrics_TB = [], []

for filepath in archivos:
    filename = os.path.basename(filepath)
    parts = filename.replace('overlap_data_', '').replace('.dat', '').split('_')
    A1 = float(parts[1]); A2 = float(parts[3]); om1 = float(parts[5])
    
    print(f"Procesando {filename}...")
    steps = load_all_steps(filepath)
    Z_vals = np.array([s['Z'] for s in steps])
    N_guides = steps[0]['S'].shape[0]
    
    # FEM
    c_FEM = np.array([lowdin_orthogonalize(s['S'], s['u']) for s in steps])
    metrics_FEM = np.array([compute_metrics_guides(c) for c in c_FEM])
    
    # TB
    c_TB = simulate_tb_param(N_guides, params_opt, Z_vals, om1)
    metrics_TB = np.array([compute_metrics_guides(c) for c in c_TB])
    
    all_Z.append(Z_vals)
    all_A1.append(np.full_like(Z_vals, A1))
    all_A2.append(np.full_like(Z_vals, A2))
    all_omega1.append(np.full_like(Z_vals, om1))
    all_metrics_FEM.append(metrics_FEM)
    all_metrics_TB.append(metrics_TB)

# Concatenar todo
all_Z = np.concatenate(all_Z)
all_A1 = np.concatenate(all_A1)
all_A2 = np.concatenate(all_A2)
all_omega1 = np.concatenate(all_omega1)
all_metrics_FEM = np.vstack(all_metrics_FEM)
all_metrics_TB = np.vstack(all_metrics_TB)
print(f"\nTotal de puntos combinados: {len(all_Z)}")

# ============================================================
# 5. Características enriquecidas
# ============================================================
residual = all_metrics_FEM - all_metrics_TB
omega2_vals = 2.0 * all_omega1

X = np.column_stack((
    all_Z, all_A1, all_A2, all_omega1,
    all_metrics_TB[:, 0], all_metrics_TB[:, 1], all_metrics_TB[:, 2], all_metrics_TB[:, 3],
    np.sin(all_omega1 * all_Z), np.cos(all_omega1 * all_Z),
    np.sin(omega2_vals * all_Z), np.cos(omega2_vals * all_Z)
))
y_res = residual

# ============================================================
# 6. División en Train (80%) y Test (20%)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_res, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Entrenamiento: {X_train_scaled.shape[0]} puntos")
print(f"Test (ocultos): {X_test_scaled.shape[0]} puntos")

# ============================================================
# 7. Entrenar el MLP (Arquitectura grande)
# ============================================================
mlp_res = MLPRegressor(
    hidden_layer_sizes=(128, 128, 64),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    max_iter=3000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.15
)

print("\nEntrenando MLP...")
mlp_res.fit(X_train_scaled, y_train)

# ============================================================
# 8. Evaluación en Test
# ============================================================
y_pred_train = mlp_res.predict(X_train_scaled)
y_pred_test = mlp_res.predict(X_test_scaled)

rmse_train = np.sqrt(np.mean((y_train - y_pred_train)**2, axis=0))
rmse_test = np.sqrt(np.mean((y_test - y_pred_test)**2, axis=0))

metric_names = ['IPR', 'COM', 'MCD', 'Edge']
print("\n" + "="*70)
print("   RMSE SOBRE TRAIN (80%) Y TEST (20% - DATOS NUEVOS)")
print("="*70)
for i, name in enumerate(metric_names):
    print(f"{name} -> Train: {rmse_train[i]:.6f} | Test: {rmse_test[i]:.6f}")
print("="*70)
print(f"Suma RMSE (Test): {np.sum(rmse_test):.6f}")
print(f"Suma RMSE (Train): {np.sum(rmse_train):.6f}")
print("=> Si el Test es cercano al Train, ¡el modelo GENERALIZA! (No ha memorizado)")

# ============================================================
# 9. Gráficas de validación
# ============================================================
plt.figure()
plt.plot(mlp_res.loss_curve_, label='Pérdida de entrenamiento (MSE)')
if hasattr(mlp_res, 'validation_scores_') and mlp_res.validation_scores_ is not None:
    plt.plot(mlp_res.validation_scores_, label='Score de validación (R²)')
plt.xlabel('Épocas')
plt.ylabel('Valor')
plt.title('Curva de aprendizaje')
plt.legend()
plt.grid(True)
plt.savefig('curva_aprendizaje_final.png')
plt.show()

# ============================================================
# 10. Predicción híbrida y residuos
# ============================================================
X_scaled_full = scaler.transform(X)
res_pred_full = mlp_res.predict(X_scaled_full)
metrics_hybrid_full = all_metrics_TB + res_pred_full

error_final = all_metrics_FEM - metrics_hybrid_full

plt.figure(figsize=(8, 4))
plt.scatter(all_Z, error_final[:, 2], s=1, alpha=0.3, label='Error en MCD')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Z')
plt.ylabel('Error FEM - Híbrido')
plt.title('Análisis de residuos (¿Ruido aleatorio?)')
plt.legend()
plt.grid(True)
plt.savefig('residuos_final.png')
plt.show()

# ============================================================
# 11. Gráfica comparativa FEM vs Híbrido
# ============================================================
plt.figure(figsize=(12, 10))
for i, name in enumerate(metric_names):
    plt.subplot(4, 1, i+1)
    plt.plot(all_Z, all_metrics_FEM[:, i], 'b.', markersize=2, label=f'FEM real')
    plt.plot(all_Z, metrics_hybrid_full[:, i], 'r.', markersize=2, label=f'Híbrido (TB + MLP)')
    plt.ylabel(name)
    plt.legend()
    plt.grid(True)
    if i == 3:
        plt.xlabel('Z')

plt.tight_layout()
plt.savefig('hibrido_comparacion_final.pdf', dpi=300)
plt.show()

# ============================================================
# 12. Guardar modelo y escalador
# ============================================================
joblib.dump(mlp_res, 'modelo_hibrido_entrenado.pkl')
joblib.dump(scaler, 'scaler_entrenado.pkl')
print("\nModelo y escalador guardados en 'modelo_hibrido_entrenado.pkl' y 'scaler_entrenado.pkl'")
print("¡Proceso finalizado! Puedes usar el modelo para predecir nuevos parámetros en milisegundos.")
