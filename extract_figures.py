# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.signal import savgol_filter
import glob
import os
from matplotlib.animation import FuncAnimation, PillowWriter

# ============================================================
# CONFIGURACIÓN DE ARCHIVOS
# ============================================================
file1_data = "overlap_data_pml_laude_final.dat"
file1_field_pattern = "field_*.txt"

file2_data = "overlap_data_pml_laude_second.dat"
file2_field_pattern = "field_second_*.txt"

output_plot = "evolucion_completa_metrics.pdf"
output_gif = "evolucion_completa_campo.gif"
output_dir_frames = "frames_completo_png"

# ============================================================
# FUNCIONES AUXILIARES
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
# INFERENCIA DE ACOPLAMIENTOS (parámetros óptimos)
# ============================================================
def infer_links_smooth(Z_vals, c_FEM):
    N = c_FEM.shape[1]
    n_steps = len(Z_vals)
    dZ = Z_vals[1] - Z_vals[0]
    t_links = np.zeros((n_steps, N-1))
    for k in range(1, n_steps-1):
        dc_dZ = (c_FEM[k+1] - c_FEM[k-1]) / (2 * dZ)
        c_k = c_FEM[k]
        t_k = np.zeros(N-1, dtype=complex)
        for i in range(N-1):
            if np.abs(c_k[i+1]) > 1e-12:
                t_k[i] = 1j * dc_dZ[i] / c_k[i+1]
        t_links[k, :] = np.real(t_k)
    for i in range(N-1):
        t_links[:, i] = savgol_filter(t_links[:, i], window_length=9, polyorder=2, mode='interp')
    t_links[0, :] = t_links[1, :]
    t_links[-1, :] = t_links[-2, :]
    return t_links

def simulate_tb_links(N, t_links, Z_vals):
    dZ = Z_vals[1] - Z_vals[0]
    phi = np.zeros((len(Z_vals), N), dtype=complex)
    phi[0, 0] = 1.0
    for k in range(len(Z_vals)-1):
        H = np.zeros((N, N), dtype=complex)
        for n in range(N-1):
            H[n, n+1] = -t_links[k, n]
            H[n+1, n] = -t_links[k, n]
        I = np.eye(N)
        A = I + 1j * H * dZ / 2
        B = I - 1j * H * dZ / 2
        phi[k+1] = np.linalg.solve(A, B @ phi[k])
    return phi

# ============================================================
# PROCESAR UN LOTE Y DEVOLVER MÉTRICAS Y CAMPOS
# ============================================================
def process_batch(data_file, field_pattern):
    steps = load_all_steps(data_file)
    if not steps:
        return None, None, None, None, None, None
    Z_vals = np.array([s['Z'] for s in steps])
    N = steps[0]['S'].shape[0]
    c_FEM = np.array([lowdin_orthogonalize(s['S'], s['u']) for s in steps])
    metrics = np.array([compute_metrics_guides(c) for c in c_FEM])
    t_links = infer_links_smooth(Z_vals, c_FEM)
    c_TB = simulate_tb_links(N, t_links, Z_vals)
    metrics_TB = np.array([compute_metrics_guides(c) for c in c_TB])
    field_files = sorted(glob.glob(field_pattern))
    field_intensities = []
    field_Z = []
    coords = None
    for fname in field_files:
        if "_second" in fname:
            continue
        base = os.path.basename(fname)
        Z_str = base.replace("field_", "").replace(".txt", "")
        try:
            Z = float(Z_str)
        except ValueError:
            continue
        field_Z.append(Z)
        data = np.loadtxt(fname)
        if coords is None:
            coords = (data[:, 0], data[:, 1])
        psiR = data[:, 2]
        psiI = data[:, 3]
        intensity = np.sqrt(psiR**2 + psiI**2)
        field_intensities.append(intensity)
    if field_Z:
        order = np.argsort(field_Z)
        field_Z = np.array(field_Z)[order]
        field_intensities = [field_intensities[i] for i in order]
    return Z_vals, metrics, metrics_TB, field_Z, field_intensities, coords

# ============================================================
# PROCESAR AMBOS LOTES Y CONCATENAR
# ============================================================
Z1, met1, metTB1, fZ1, fInt1, coords1 = process_batch(file1_data, file1_field_pattern)
Z2, met2, metTB2, fZ2, fInt2, coords2 = process_batch(file2_data, file2_field_pattern)

if Z1 is None or Z2 is None:
    raise FileNotFoundError("No se pudieron leer uno o ambos archivos de datos.")

Z_full = np.concatenate((Z1, Z2))
metrics_FEM = np.concatenate((met1, met2), axis=0)
metrics_TB = np.concatenate((metTB1, metTB2), axis=0)

IPR_FEM, COM_FEM, MCD_FEM, Edge_FEM = metrics_FEM[:,0], metrics_FEM[:,1], metrics_FEM[:,2], metrics_FEM[:,3]
IPR_TB, COM_TB, MCD_TB, Edge_TB = metrics_TB[:,0], metrics_TB[:,1], metrics_TB[:,2], metrics_TB[:,3]

# ============================================================
# GRÁFICAS DE EVOLUCIÓN (se mantiene en PDF)
# ============================================================
plt.figure(figsize=(12, 10))
plt.subplot(4,1,1)
plt.plot(Z_full, IPR_FEM, 'b-', label='FEM')
plt.plot(Z_full, IPR_TB, 'r--', label='TB (opt)')
plt.ylabel('IPR'); plt.legend(); plt.grid(True)
plt.subplot(4,1,2)
plt.plot(Z_full, COM_FEM/2.0, 'b-', label='FEM')
plt.plot(Z_full, COM_TB/2.0, 'r--', label='TB')
plt.ylabel('COM (celda SSH)'); plt.legend(); plt.grid(True)
plt.subplot(4,1,3)
plt.plot(Z_full, MCD_FEM, 'b-', label='FEM')
plt.plot(Z_full, MCD_TB, 'r--', label='TB')
plt.ylabel('MCD'); plt.legend(); plt.grid(True)
plt.subplot(4,1,4)
plt.plot(Z_full, Edge_FEM, 'b-', label='FEM')
plt.plot(Z_full, Edge_TB, 'r--', label='TB')
plt.ylabel('Edge Intensity'); plt.xlabel('Z'); plt.legend(); plt.grid(True)
plt.tight_layout()
plt.savefig(output_plot, dpi=300)
plt.close()
print(f"Gráfica guardada: {output_plot}")

# ============================================================
# RMSE GLOBAL
# ============================================================
rmse_IPR = np.sqrt(np.mean((IPR_FEM - IPR_TB)**2))
rmse_COM = np.sqrt(np.mean((COM_FEM - COM_TB)**2))
rmse_MCD = np.sqrt(np.mean((MCD_FEM - MCD_TB)**2))
rmse_Edge = np.sqrt(np.mean((Edge_FEM - Edge_TB)**2))
print("\n" + "="*60)
print("   RMSE GLOBAL (EVOLUCIÓN COMPLETA)")
print("="*60)
print(f"IPR: {rmse_IPR:.6f}")
print(f"COM: {rmse_COM:.6f}")
print(f"MCD: {rmse_MCD:.6f}")
print(f"Edge: {rmse_Edge:.6f}")
print("="*60)

# ============================================================
# ANIMACIÓN Y FRAMES PNG (con tripcolor para mejor visualización)
# ============================================================
all_field_Z = np.concatenate((fZ1, fZ2))
all_intensities = fInt1 + fInt2

if len(all_intensities) != len(Z_full):
    field_dict = {z: I for z, I in zip(all_field_Z, all_intensities)}
    aligned = []
    for z in Z_full:
        if z in field_dict:
            aligned.append(field_dict[z])
        else:
            aligned.append(aligned[-1] if aligned else np.zeros_like(all_intensities[0]))
    all_intensities = aligned
    all_field_Z = Z_full

x, y = None, None
if fInt1:
    fname = sorted(glob.glob(file1_field_pattern))[0]
    while "_second" in fname:
        fname = sorted(glob.glob(file1_field_pattern))[1]
    data = np.loadtxt(fname)
    x, y = data[:, 0], data[:, 1]
elif fInt2:
    fname = sorted(glob.glob(file2_field_pattern))[0]
    data = np.loadtxt(fname)
    x, y = data[:, 0], data[:, 1]

if x is not None and len(all_intensities) > 0:
    os.makedirs(output_dir_frames, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.tripcolor(x, y, all_intensities[0], shading='gouraud', cmap='inferno')
    cbar = plt.colorbar(im, ax=ax, label='Intensidad')
    ax.set_title(f'Z = {Z_full[0]:.4f}')
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.axis('equal')

    def update(frame):
        ax.clear()
        im = ax.tripcolor(x, y, all_intensities[frame], shading='gouraud', cmap='inferno')
        ax.set_title(f'Z = {Z_full[frame]:.4f}')
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.axis('equal')
        return im,

    ani = FuncAnimation(fig, update, frames=len(all_intensities), interval=100, blit=False)
    ani.save(output_gif, writer=PillowWriter(fps=10))
    print(f"Animación GIF guardada: {output_gif}")

    for i, intensity in enumerate(all_intensities):
        plt.figure(figsize=(10, 4.5), dpi=150)
        plt.tripcolor(x, y, intensity, shading='gouraud', cmap='inferno')
        plt.colorbar(label='Intensidad')
        plt.title(f'Z = {Z_full[i]:.4f}')
        plt.xlabel('x'); plt.ylabel('y'); plt.axis('equal')
        plt.tight_layout()
        plt.savefig(f"{output_dir_frames}/frame_{i:04d}.png")
        plt.close()
    print(f"Frames PNG guardados en: {output_dir_frames}")

print("¡Proceso completado!")
