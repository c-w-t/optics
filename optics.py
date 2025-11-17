import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from scipy.linalg import solve_banded
import sympy as sp
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
    font-size: 1.6rem !important;
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
n = st.number_input("Matrix size", min_value=2, max_value=20, value=5, step=1)
b_vals = st.text_input("Main diagonal", value="-2,-2,-2,-2,-2")
a_vals = st.text_input("Subdiagonal", value="1,1,1,1")
c_vals = st.text_input("Superdiagonal", value="1,1,1,1")
d_rhs = st.text_input("RHS", value="5,5,5,5,5")

# ===================== Solve Button =====================
if st.button("Solve"):
    try:
        # Convert inputs to SymPy Rational for exact eigenvalues
        b = [sp.Rational(x) for x in b_vals.split(",")]
        a = [sp.Rational(x) for x in a_vals.split(",")]
        c = [sp.Rational(x) for x in c_vals.split(",")]
        # RHS remains numeric for solvers
        d = np.array([float(x) for x in d_rhs.split(",")])

        if len(b) != n or len(a) != n-1 or len(c) != n-1 or len(d) != n:
            st.error("Lengths must match: b=n, d=n, a=n-1, c=n-1.")
        else:
            # ===================== Solve Systems =====================
            x_thomas = thomas(np.concatenate(([0], [float(ai) for ai in a])),
                              np.array([float(bi) for bi in b]),
                              np.concatenate(([float(ci) for ci in c], [0])),
                              d)
            x_scipy = solve_tridiagonal_scipy([float(ai) for ai in a],
                                              np.array([float(bi) for bi in b]),
                                              [float(ci) for ci in c],
                                              d)

            # ===================== Symbolic Eigenvalues =====================
            A = sp.Matrix(n, n, lambda i,j: 0)
            for i in range(n):
                A[i,i] = b[i]
            for i in range(n-1):
                A[i+1,i] = a[i]
                A[i,i+1] = c[i]

            eigenvals_dict = A.eigenvals()
            eigenvals_list = list(eigenvals_dict.keys())
            eigen_numeric = [float(val.evalf()) for val in eigenvals_list]

            st.markdown("**Symbolic Eigenvalues (exact)**")
            for idx, val in enumerate(eigenvals_list):
                st.latex(f"\\lambda_{{{idx+1}}} = {sp.latex(val)}")

            # ===================== Solutions & Eigenvalues Table =====================
            df = pd.DataFrame({
                "Index": range(n),
                "Thomas": x_thomas,
                "SciPy": x_scipy,
                "Eigen (numeric)": eigen_numeric
            })
            st.markdown("**Solutions & Eigenvalues Table**")
            st.dataframe(df.style.format("{:.6f}"))

            # ===================== Plot Thomas & SciPy =====================
            fig1, ax1 = plt.subplots(figsize=(6,4))
            ax1.plot(range(n), x_thomas, 'd-.', label="Thomas")
            ax1.plot(range(n), x_scipy, 'x:', label="SciPy")
            ax1.set_xlabel("Index")
            ax1.set_ylabel("Value")
            ax1.set_title("Thomas & SciPy Solver Comparison")
            ax1.legend()
            st.pyplot(fig1)

            # ===================== Plot Numeric Eigenvalues =====================
            fig2, ax2 = plt.subplots(figsize=(6,4))
            ax2.plot(range(n), eigen_numeric, 'o-', label="Eigenvalues (numeric)")
            ax2.set_xlabel("Index")
            ax2.set_ylabel("Value")
            ax2.set_title("Numeric Eigenvalues from Symbolic")
            ax2.legend()
            st.pyplot(fig2)

    except Exception as e:
        st.error(f"Error: {e}")
