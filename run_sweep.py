import subprocess
import os
import itertools
from multiprocessing import Pool

# ============================================================
# CONFIGURACIÓN
# ============================================================
template_file = "hyco_pml_laude_param.edp"

A1_vals = [0.05, 0.1, 0.15]
A2_vals = [0.05, 0.1, 0.2]
omega1_vals = [8.0, 10.0, 12.0]

# ============================================================
# FILTRO DE COMBINACIONES
# ============================================================
def is_valid_combination(A1, A2, om1):
    if A1 + A2 >= 1.4:
        return False
    if A1 == 0.1 and A2 == 0.2 and om1 == 10.0:
        return False
    return True

# ============================================================
# FUNCIÓN DE SIMULACIÓN
# ============================================================
def run_simulation(params):
    A1, A2, om1 = params
    tag = f"A1_{A1}_A2_{A2}_om1_{om1}"
    output_file = f"overlap_data_{tag}.dat"

    if os.path.exists(output_file):
        print(f"[{tag}] Archivo ya existe. Saltando.", flush=True)
        return tag

    edp_file = f"sim_{tag}.edp"
    
    with open(template_file, 'r') as f:
        code = f.read()
    
    code = code.replace("real A1 = 0.1;", f"real A1 = {A1};")
    code = code.replace("real A2 = 0.2;", f"real A2 = {A2};")
    code = code.replace("real omega1 = 10.0;", f"real omega1 = {om1};")
    code = code.replace('string outputTag = "A1_0.1_A2_0.2_om1_10";', f'string outputTag = "{tag}";')
    
    with open(edp_file, 'w') as f:
        f.write(code)
    
    print(f"[{tag}] Iniciando simulación... (Tardará ~10h)", flush=True)
    subprocess.run(["FreeFem++", edp_file])
    print(f"[{tag}] Completada. Archivo: {output_file}", flush=True)
    return tag

# ============================================================
# BLOQUE PRINCIPAL
# ============================================================
if __name__ == '__main__':
    combinations = list(itertools.product(A1_vals, A2_vals, omega1_vals))
    valid_combinations = [c for c in combinations if is_valid_combination(c[0], c[1], c[2])]

    print(f"Combinaciones totales: {len(combinations)}", flush=True)
    print(f"Combinaciones válidas: {len(valid_combinations)}", flush=True)

    with Pool(processes=6) as pool:
        results = pool.map(run_simulation, valid_combinations)

    print("¡Todas las simulaciones pendientes han finalizado!", flush=True)
