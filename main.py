import os
from model import simulate
from analysis import (load_and_prepare_dataset, compute_quartiles,
                      linear_regression_a, linear_regression_b, predict_a, predict_b)
from utils import dimensionalize, save_results, plot_trajectory

DATA_PATH = "dataset.csv"
OUTPUT_DIR = "final_project_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = load_and_prepare_dataset(DATA_PATH)
q1, q2, q3 = compute_quartiles(df)
a_k, a_b = linear_regression_a(df)
b_k, b_b = linear_regression_b(df)

v0 = q3
a_val = predict_a(v0, a_k, a_b)
b_val = predict_b(v0, b_k, b_b)

sol_nofric = simulate(30, 0, 0)
sol_fric = simulate(30, a_val, b_val)

t_f, x_f, y_f = dimensionalize(sol_fric, v0)
_, x_nf, y_nf = dimensionalize(sol_nofric, v0)

d = x_f[-1]
h = max(y_f)
t_total = t_f[-1]
t_max = t_f[list(y_f).index(h)]
vx_end = sol_fric.y[2, -1]
vy_end = sol_fric.y[3, -1]
v_end = v0 * (vx_end**2 + vy_end**2)**0.5

save_results(os.path.join(OUTPUT_DIR, "results.txt"),
             q1, q2, q3, a_k, a_b, b_k, b_b, d, h, t_total, t_max, v_end)

plot_trajectory(x_nf, y_nf, x_f, y_f, os.path.join(OUTPUT_DIR, "trajectory_comparison.png"))