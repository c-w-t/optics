import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# ===================== Homebrew Yellow UI =====================
homebrew_css = """
<style>
body { background-color: #fdf8e2; color: #3a2e2e; font-family: 'Courier New', monospace; }
.sidebar .sidebar-content { background-color: #fff8b0; border-right: 3px solid #b8860b; }
.sidebar .sidebar-content h1, h2, h3 { color: #b8860b !important; }
h1, h2, h3 { color: #b8860b !important; text-shadow: 1px 2px 1px #fef7d6; }
.stButton>button { background-color: #b8860b; color: white; border: 1px solid #8b6508; border-radius: 6px; padding: 0.5em 1em; font-weight: bold; box-shadow: 2px 2px 4px #e8d8a0; }
.stButton>button:hover { background-color: #d4a017; border-color: #a07505; }
.stDownloadButton>button { background-color: #006400; color: white; border-radius: 6px; padding: 0.5em 1em; font-weight: bold; border: 1px solid #003300; box-shadow: 2px 2px 4px #c9b8a8; }
.stDownloadButton>button:hover { background-color: #008f00; }
.dataframe { border: 2px solid #b8860b !important; background-color: #fffaf0 !important; }
.stProgress>div>div>div>div { background-color: #b8860b; }
.element-container div[role="img"] { background-color: #fdf8e2 !important; padding: 10px; border-radius: 6px; border: 1px solid #e8d8a0; }
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

# ===================== Sidebar Settings =====================
st.sidebar.header("Simulation Settings")

# --- RK4 Settings ---
x0 = st.sidebar.number_input("Initial x (x0)", value=0.0)
y0 = st.sidebar.number_input("Initial y (y0)", value=1.0)
h = st.sidebar.number_input("Step size h", value=0.1)
steps = st.sidebar.number_input("Number of steps", value=50)

# --- Tridiagonal Settings ---
st.sidebar.subheader("Tridiagonal System")
n = st.sidebar.number_input("System size (n)", min_value=2, max_value=20, value=5, step=1)
b_diag = st.sidebar.text_input("Main diagonal b (comma-separated)", value="2,2,2,2,2")
a_diag = st.sidebar.text_input("Sub diagonal a (comma-separated)", value="1,1,1,1")
c_diag = st.sidebar.text_input("Super diagonal c (comma-separated)", value="1,1,1,1")
d_rhs = st.sidebar.text_input("RHS d (comma-separated)", value="5,5,5,5,5")

# ===================== RK4 Simulation =====================
def demo_func(x, y):
    return -0.5 * y  # simple exponential decay

if st.sidebar.button("RK4"):
    xs, ys = rk4_solve(demo_func, x0, y0, h, steps)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(xs, ys, label="RK4 Solution")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("RK4 Simulation Result")
    ax.legend()
    st.pyplot(fig)

# ===================== Tridiagonal Simulation =====================
if st.sidebar.button("Tridiagonal"):
    try:
        b = np.array([float(x) for x in b_diag.split(",")])
        a = np.array([float(x) for x in a_diag.split(",")])
        c = np.array([float(x) for x in c_diag.split(",")])
        d = np.array([float(x) for x in d_rhs.split(",")])
        if len(b) != n or len(a) != n-1 or len(c) != n-1 or len(d) != n:
            st.error("Array sizes do not match system size n.")
        else:
            x_sol = thomas(np.concatenate(([0], a)), b, np.concatenate((c, [0])), d)
            st.write("Solution x:")
            st.write(x_sol)
            fig, ax = plt.subplots(figsize=(6,4))
            ax.bar(range(n), x_sol)
            ax.set_xlabel("Index")
            ax.set_ylabel("Value")
            ax.set_title("Tridiagonal Solver Result")
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Error parsing input: {e}")
