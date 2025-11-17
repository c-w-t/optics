
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# ===================== Homebrew UI =====================
homebrew_css = """
<style>
body { background-color: #f4f1e9; color: #3a2e2e; font-family: 'Courier New', monospace; }
.sidebar .sidebar-content { background-color: #fff7d6; border-right: 3px solid #8b0000; }
.sidebar .sidebar-content h1, h2, h3 { color: #6b0000 !important; }
h1, h2, h3 { color: #6b0000 !important; text-shadow: 1px 2px 1px #e8dccc; }
.stButton>button { background-color: #8b0000; color: white; border: 1px solid #330000; border-radius: 6px; padding: 0.5em 1em; font-weight: bold; box-shadow: 2px 2px 4px #c9b8a8; }
.stButton>button:hover { background-color: #a00000; border-color: #220000; }
.stDownloadButton>button { background-color: #006400; color: white; border-radius: 6px; padding: 0.5em 1em; font-weight: bold; border: 1px solid #003300; box-shadow: 2px 2px 4px #c9b8a8; }
.stDownloadButton>button:hover { background-color: #008f00; }
.dataframe { border: 2px solid #8b0000 !important; background-color: #fffaf0 !important; }
.stProgress>div>div>div>div { background-color: #8b0000; }
.element-container div[role="img"] { background-color: #f4f1e9 !important; padding: 10px; border-radius: 6px; border: 1px solid #d4c7b5; }
</style>
"""
st.markdown(homebrew_css, unsafe_allow_html=True)
st.title("Optics Simulation Platform")

# ===================== RK4 Solver =====================
def rk4_step(f, x, y, h):
    k1 = f(x, y)
    k2 = f(x + h/2, y + h*k1/2)
    k3 = f(x + h/2, y + h*k2/2)
    k4 = f(x + h, y + h*k3)
    return y + h*(k1 + 2*k2 + 2*k3 + k4)/6

def rk4_solve(f, x0, y0, h, steps):
    xs = [x0]
    ys = [y0]
    x, y = x0, y0
    for _ in range(steps):
        y = rk4_step(f, x, y, h)
        x += h
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# ===================== Tridiagonal Solver =====================
@njit
def thomas(a, b, c, d):
    n = len(d)
    cp = np.zeros(n)
    dp = np.zeros(n)
    x = np.zeros(n)
    cp[0] = c[0]/(b[0]+1e-12)
    dp[0] = d[0]/(b[0]+1e-12)
    for i in range(1, n):
        denom = b[i] - a[i]*cp[i-1]
        denom = denom if denom != 0 else 1e-12
        cp[i] = c[i]/denom if i < n-1 else 0
        dp[i] = (d[i] - a[i]*dp[i-1])/denom
    x[-1] = dp[-1]
    for i in range(n-2, -1, -1):
        x[i] = dp[i] - cp[i]*x[i+1]
    return x

# ===================== Demo Simulation =====================
st.sidebar.header("Simulation Settings")
x0 = st.sidebar.number_input("Initial x (x0)", value=0.0)
y0 = st.sidebar.number_input("Initial y (y0)", value=1.0)
h = st.sidebar.number_input("Step size h", value=0.1)
steps = st.sidebar.number_input("Number of steps", value=50)

def demo_func(x, y):
    return -0.5 * y  # simple exponential decay

if st.sidebar.button("Run RK4 Simulation"):
    xs, ys = rk4_solve(demo_func, x0, y0, h, steps)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(xs, ys, label="RK4 Solution")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("RK4 Simulation Result")
    ax.legend()
    st.pyplot(fig)
