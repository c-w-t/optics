import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from scipy.linalg import solve_banded
from sympy import symbols, cos, pi, sqrt, N, latex
import pandas as pd

# ===================== Homebrew Yellow UI =====================
st.markdown("""
<style>
body { background-color: #fdf8e2; color: #3a2e2e; font-family: 'Courier New', monospace; }
.stNumberInput, .stTextInput { margin-bottom: 0.2em; }
.stButton>button { background-color: #b8860b; color: white; border: 1px solid #8b6508; border-radius: 6px; padding: 0.4em 0.8em; font-weight: bold; box-shadow: 1px 1px 3px #e8d8a0; margin-top:0.2em;}
.stButton>button:hover { background-color: #d4a017; border-color: #a07505; }
.dataframe { border: 2px solid #b8860b !important; background-color: #fffaf0 !important; }
.stProgress>div>div>div>div { background-color: #b8860b; }
.custom-title {
    font-size: 0.8rem !important;
    font-weight: bold;
    color: #b8860b !important;
    margin-bottom:0.5em;
}
</style>
""", unsafe_allow_html=True)

# ===================== Page Title =====================
st.markdown('<p class="custom-title">Tridiagonal Matrix Solver</p>', unsafe_allow_html=True)

# ===================== Thomas Solver =====================
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

# ===================== SciPy Solver =====================
def solve_tridiagonal_scipy(a, b, c, d):
    n = len(b)
    ab = np.zeros((3, n))
    ab[0,1:] = c   # super-diagonal
    ab[1,:] = b    # main diagonal
    ab[2,:-1] = a  # sub-diagonal
    x = solve_banded((1,1), ab, d)
    return x

# ===================== Inputs =====================
n = st.number_input("System size (n)", min_value=2, max_value=20, value=5, step=1)
b_val = st.number_input("Main diagonal b", value=2.0)
c_val = st.number_input("Off-diagonal c (symmetric)", value=1.0)
d_rhs = st.text_input("RHS d (comma-separated)", value="5,5,5,5,5")

# ===================== Solve Button =====================
if st.button("Solve"):
    try:
        d = np.array([float(x) for x in d_rhs.split(",")])
        if len(d) != n:
            st.error("RHS length must match n.")
        else:
            a = np.full(n-1, c_val)
            b = np.full(n, b_val)

            # Thomas and SciPy solutions
            x_thomas = thomas(np.concatenate(([0], a)), b, np.concatenate((a, [0])), d)
            x_scipy = solve_tridiagonal_scipy(a, b, a, d)

            # Symbolic eigenvalues
            k = symbols('k', integer=True)
            eigen_cos = b_val + 2*c_val*cos(pi*k/(n+1))
            eigen_sqrt = b_val + sqrt((2*c_val*cos(pi*k/(n+1)))**2)
            cos_vals = [eigen_cos.subs(k,i) for i in range(1, n+1)]
            sqrt_vals = [eigen_sqrt.subs(k,i) for i in range(1, n+1)]
            cos_numeric = [N(val) for val in cos_vals]
            sqrt_numeric = [N(val) for val in sqrt_vals]

            # Build DataFrame
            df = pd.DataFrame({
                "Index": range(n),
                "Thomas": x_thomas,
                "SciPy": x_scipy,
                "Eigen (cos)": cos_vals,
                "Eigen (sqrt)": sqrt_vals,
                "Eigen (cos numeric)": cos_numeric,
                "Eigen (sqrt numeric)": sqrt_numeric
            })
            st.markdown("**Solutions & Eigenvalues Table**")
            st.dataframe(df.style.format("{:.6f}"))

            # Plot numeric eigenvalues and solver solutions
            fig, ax = plt.subplots(figsize=(6,4))
            ax.plot(range(n), cos_numeric, 'o-', label="Eigen (cos numeric)")
            ax.plot(range(n), sqrt_numeric, 's--', label="Eigen (sqrt numeric)")
            ax.plot(range(n), x_thomas, 'd-.', label="Thomas")
            ax.plot(range(n), x_scipy, 'x:', label="SciPy")
            ax.set_xlabel("Index")
            ax.set_ylabel("Value")
            ax.set_title("Solver & Eigenvalue Comparison")
            ax.legend()
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")
