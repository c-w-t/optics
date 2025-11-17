import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from scipy.linalg import solve_banded, eigh_tridiagonal

# ===================== Compact Homebrew Yellow UI =====================
homebrew_css = """
<style>
body { background-color: #fdf8e2; color: #3a2e2e; font-family: 'Courier New', monospace; }
h2 { color: #b8860b !important; text-shadow: 1px 1px 1px #fef7d6; margin:0 0 0.1em 0;}
.stNumberInput, .stTextInput { margin-bottom: 0.2em; }
.stButton>button { background-color: #b8860b; color: white; border: 1px solid #8b6508; border-radius: 6px; padding: 0.4em 0.8em; font-weight: bold; box-shadow: 1px 1px 3px #e8d8a0; margin-top:0.2em;}
.stButton>button:hover { background-color: #d4a017; border-color: #a07505; }
.dataframe { border: 2px solid #b8860b !important; background-color: #fffaf0 !important; }
.stProgress>div>div>div>div { background-color: #b8860b; }
</style>
"""
st.markdown(homebrew_css, unsafe_allow_html=True)

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

# ===================== Main Page Inputs =====================
st.markdown("<h2>Tridiagonal System Settings</h2>", unsafe_allow_html=True)

n = st.number_input("System size (n)", min_value=2, max_value=20, value=5, step=1)
b_diag = st.text_input("Main diagonal b (comma-separated)", value="2,2,2,2,2")
a_diag = st.text_input("Sub diagonal a (comma-separated)", value="1,1,1,1")
c_diag = st.text_input("Super diagonal c (comma-separated)", value="1,1,1,1")
d_rhs = st.text_input("RHS d (comma-separated)", value="5,5,5,5,5")

solver = st.radio("Choose solver:", ["Thomas", "SciPy"])
show_eigen = st.checkbox("Compute eigenvalues", value=True)

# ===================== Solve Button =====================
if st.button("Solve"):
    try:
        b = np.array([float(x) for x in b_diag.split(",")])
        a = np.array([float(x) for x in a_diag.split(",")])
        c = np.array([float(x) for x in c_diag.split(",")])
        d = np.array([float(x) for x in d_rhs.split(",")])
        if len(b) != n or len(a) != n-1 or len(c) != n-1 or len(d) != n:
            st.error("Array sizes do not match system size n.")
        else:
            # Solve system
            if solver == "Thomas":
                x_sol = thomas(np.concatenate(([0], a)), b, np.concatenate((c, [0])), d)
            else:
                x_sol = solve_tridiagonal_scipy(a, b, c, d)

            st.write("Solution x:")
            st.write(x_sol)

            fig, ax = plt.subplots(figsize=(6,4))
            ax.bar(range(n), x_sol)
            ax.set_xlabel("Index")
            ax.set_ylabel("Value")
            ax.set_title(f"{solver} Solver Result")
            st.pyplot(fig)

            # Compute eigenvalues if checkbox selected
            if show_eigen:
                eigenvals = eigh_tridiagonal(b, a)  # b=main diag, a=sub diag
                st.write("Eigenvalues of the tridiagonal matrix:")
                st.write(np.round(eigenvals[0], 6))  # eigenvals[0] contains eigenvalues

    except Exception as e:
        st.error(f"Error parsing input: {e}")
