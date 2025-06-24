import numpy as np
import matplotlib.pyplot as plt
import os

g = 9.8

def dimensionalize(sol, v0):
    x = sol.y[0] * v0**2 / g
    y = sol.y[1] * v0**2 / g
    t = sol.t * v0 / g
    return t, x, y

def save_results(path, q1, q2, q3, a_k, a_b, b_k, b_b, d, h, t_f, t_h, v_end):
    with open(path, "w") as f:
        f.write(f"Q1: {q1:.2f}\nQ2: {q2:.2f}\nQ3: {q3:.2f}\n")
        f.write(f"a(v) = {a_k:.6f} * v + {a_b:.6f}\n")
        f.write(f"b(v^2) = {b_k:.6f} * v^2 + {b_b:.6f}\n")
        f.write(f"Flight Distance: {d:.2f} m\n")
        f.write(f"Max Height: {h:.2f} m\n")
        f.write(f"Flight Time: {t_f:.2f} s\n")
        f.write(f"Time to Max Height: {t_h:.2f} s\n")
        f.write(f"Final Speed: {v_end:.2f} m/s\n")

def plot_trajectory(x1, y1, x2, y2, path):
    plt.figure()
    plt.plot(x1, y1, label="Без сопротивления")
    plt.plot(x2, y2, label="С сопротивлением")
    plt.xlabel("Горизонтальное расстояние (м)")
    plt.ylabel("Высота (м)")
    plt.title("Траектория тела")
    plt.legend()
    plt.grid()
    plt.savefig(path)