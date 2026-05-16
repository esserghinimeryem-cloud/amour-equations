# app.py
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.integrate import odeint
from scipy.optimize import fsolve

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="L'amour en équations",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ L'amour en équations")
st.markdown("""
### Portrait de phase interactif du modèle de Rey (2010)

Ce modèle décrit l'évolution d'un couple à travers deux variables :
- $x(t)$ : le niveau de sentiment commun
- $\\lambda(t)$ : la variable adjointe (liée à l'effort optimal)

Le système dynamique optimal est :
$$\\frac{dx}{dt} = -rx + \\Phi(\\lambda), \\quad 
\\frac{d\\lambda}{dt} = (r+\\rho)\\lambda - U'(x)$$

avec $U(x) = 5\\log(1+x)$ et $D(c) = \\frac{1}{2}|c-\\bar{c}|^2$.

> *Paramètres de référence tirés de : Rey J-M (2010), PLoS ONE 5(3): e9881 
et Chazel F. (2025), Bureau d'études 3MIC-MA, INSA Toulouse.*
""")

st.markdown("---")

# ============================================================
# SLIDERS DANS LA SIDEBAR
# ============================================================
st.sidebar.title("⚙️ Paramètres")
st.sidebar.markdown("Faites varier les paramètres pour observer "
                    "l'influence sur le portrait de phase.")

r = st.sidebar.slider(
    "Taux d'érosion $r$",
    min_value=0.5, max_value=5.0,
    value=2.0, step=0.1,
    help="Plus r est grand, plus les sentiments s'érodent vite"
)

rho = st.sidebar.slider(
    "Facteur d'impatience $\\rho$",
    min_value=0.1, max_value=5.0,
    value=1.0, step=0.1,
    help="Préférence pour le présent vs le futur"
)

cb = st.sidebar.slider(
    "Seuil naturel d'effort $\\bar{c}$",
    min_value=0.0, max_value=1.0,
    value=0.2, step=0.05,
    help="Niveau d'effort qui minimise le sacrifice"
)

xmin = st.sidebar.slider(
    "Seuil de rupture $x_{min}$",
    min_value=0.01, max_value=0.5,
    value=0.1, step=0.01,
    help="En dessous de ce seuil, la relation est insatisfaisante"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Trajectoires initiales**")
x0_traj = st.sidebar.slider(
    "Position initiale $x_0$",
    min_value=0.5, max_value=2.0,
    value=1.8, step=0.1
)

# ============================================================
# FONCTIONS
# ============================================================
def Uprime(x):
    return 5 / (1 + x)

def Useconde(x):
    return -5 / (1 + x)**2

def Dprime(c, cb):
    return c - cb

def Phi(lam, cb):
    return np.maximum(lam + cb, 0)

def Phiprime(lam, cb):
    return np.where(lam + cb >= 0, 1.0, 0.0)

def system(Y, t, r, rho, cb):
    x, lam = Y
    dxdt   = -r * x + Phi(lam, cb)
    dlamdt = (r + rho) * lam - Uprime(x)
    return [dxdt, dlamdt]

def trouver_equilibre(r, rho, cb):
    def eq(x):
        return Dprime(r * x, cb) - Uprime(x) / (r + rho)
    try:
        x_eq = fsolve(eq, 0.5)[0]
        lam_eq = Uprime(x_eq) / (r + rho)
        return x_eq, lam_eq
    except:
        return None, None

# ============================================================
# CALCUL EQUILIBRE ET STABILITE
# ============================================================
x_eq, lam_eq = trouver_equilibre(r, rho, cb)
c_eq = Phi(lam_eq, cb) if x_eq is not None else None

# Matrice jacobienne
if x_eq is not None:
    A = np.array([
        [-r,               Phiprime(lam_eq, cb)],
        [-Useconde(x_eq),  r + rho             ]
    ])
    valeurs_propres = np.linalg.eigvals(A)
    det_A = np.linalg.det(A)
    effort_gap = c_eq - cb

# ============================================================
# AFFICHAGE DES INFOS NUMERIQUES
# ============================================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("$x^*$ (sentiment éq.)",
              f"{x_eq:.3f}" if x_eq else "N/A")
with col2:
    st.metric("$\\lambda^*$ (variable adj.)",
              f"{lam_eq:.3f}" if lam_eq else "N/A")
with col3:
    st.metric("$c^*$ (effort éq.)",
              f"{c_eq:.3f}" if c_eq else "N/A")
with col4:
    st.metric("Effort gap $c^* - \\bar{c}$",
              f"{effort_gap:.3f}" if x_eq else "N/A",
              delta="positif ✓" if (x_eq and effort_gap > 0)
              else "négatif ✗")

st.markdown("---")

# ============================================================
# TRACE DU PORTRAIT DE PHASE
# ============================================================
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor('#0e1117')
ax.set_facecolor('#0e1117')

# Champ de vecteurs
x_vals   = np.linspace(0.01, 2.0, 18)
lam_vals = np.linspace(-0.5, 2.0, 18)
X, LAM   = np.meshgrid(x_vals, lam_vals)
DX   = -r * X + Phi(LAM, cb)
DLAM = (r + rho) * LAM - Uprime(X)
norm = np.sqrt(DX**2 + DLAM**2)
norm[norm == 0] = 1
ax.quiver(X, LAM, DX/norm, DLAM/norm,
          alpha=0.4, color='gray', scale=25)

# Isoclines
x_iso = np.linspace(0.01, 2.0, 300)
lam_iso1 = Dprime(r * x_iso, cb)
lam_iso2 = Uprime(x_iso) / (r + rho)
ax.plot(x_iso, lam_iso1, 'b--', linewidth=2,
        label=r'$\frac{dx}{dt}=0$ : $\lambda=D\'(rx)$')
ax.plot(x_iso, lam_iso2, 'r--', linewidth=2,
        label=r'$\frac{d\lambda}{dt}=0$ : $\lambda=\frac{U\'(x)}{r+\rho}$')

# Seuils
ax.axhline(y=-cb, color='orange', linestyle=':',
           linewidth=1.5,
           label=f"$D'(0) = {-cb:.2f}$")
ax.axvline(x=xmin, color='purple', linestyle=':',
           linewidth=1.5,
           label=f'$x_{{min}} = {xmin:.2f}$')

# Point d'equilibre
if x_eq is not None:
    ax.plot(x_eq, lam_eq, 'wo', markersize=10,
            label=f'$(x^*,\\lambda^*)=({x_eq:.2f},{lam_eq:.2f})$',
            zorder=5)
    ax.annotate(f'$(x^*, \\lambda^*)$',
                xy=(x_eq, lam_eq),
                xytext=(x_eq + 0.05, lam_eq + 0.12),
                fontsize=10, color='white')

    # Effort gap
    ax.annotate('', xy=(x_eq, lam_eq),
                xytext=(x_eq, -cb),
                arrowprops=dict(arrowstyle='<->',
                                color='lime', lw=2))
    ax.text(x_eq + 0.03, (lam_eq - cb) / 2,
            'effort gap', color='lime', fontsize=9)

# Trajectoires
t_span = np.linspace(0, 4, 2000)

if x_eq is not None:
    # Convergentes
    first_conv = True
    for delta in [0.08, 0.18, -0.06]:
        lam0 = lam_eq + delta
        sol = odeint(system, [x0_traj, lam0], t_span,
                     args=(r, rho, cb))
        mask = sol[:, 0] > 0.01
        label = '$V^s$ — relation durable' if first_conv else ''
        ax.plot(sol[mask, 0], sol[mask, 1],
                color='lime', alpha=0.8,
                linewidth=1.8, label=label)
        first_conv = False

    # Divergentes
    first_div = True
    for delta in [-0.25, -0.40]:
        lam0 = lam_eq + delta
        sol = odeint(system, [x0_traj, lam0], t_span,
                     args=(r, rho, cb))
        mask = sol[:, 0] > 0.01
        label = 'Trajectoire de rupture' if first_div else ''
        ax.plot(sol[mask, 0], sol[mask, 1],
                color='violet', alpha=0.8,
                linewidth=1.8, label=label)
        first_div = False

# Mise en forme
ax.set_xlim(0, 2)
ax.set_ylim(-0.5, 2)
ax.set_xlabel('$x$ (niveau de sentiment)', fontsize=13,
              color='white')
ax.set_ylabel('$\\lambda$ (variable adjointe)', fontsize=13,
              color='white')
ax.set_title(
    f'Portrait de phase — $r={r}$, $\\rho={rho}$, '
    f'$\\bar{{c}}={cb}$, $x_{{min}}={xmin}$',
    fontsize=13, color='white')
ax.tick_params(colors='white')
ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['top'].set_color('gray')
ax.spines['right'].set_color('gray')
legend = ax.legend(loc='upper right', fontsize=9,
                   facecolor='#1e2130', edgecolor='gray',
                   labelcolor='white')
ax.grid(True, alpha=0.2, color='gray')

st.pyplot(fig)

# ============================================================
# INTERPRETATION AUTOMATIQUE
# ============================================================
st.markdown("---")
st.markdown("### 📊 Interprétation automatique")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("**État de la relation :**")
    if x_eq is not None:
        if x_eq > xmin:
            st.success(f"✅ L'équilibre $x^* = {x_eq:.2f}$ est "
                       f"au-dessus du seuil $x_{{min}} = {xmin}$ "
                       f": la relation est **viable**.")
        else:
            st.error(f"❌ L'équilibre $x^* = {x_eq:.2f}$ est "
                     f"en-dessous du seuil $x_{{min}} = {xmin}$ "
                     f": la relation est **vouée à l'échec**.")

with col_b:
    st.markdown("**Effort gap :**")
    if x_eq is not None:
        if effort_gap > 0.3:
            st.warning(f"⚠️ L'effort gap $c^* - \\bar{{c}} = "
                       f"{effort_gap:.3f}$ est **élevé** : "
                       f"le couple devra fournir un effort "
                       f"significativement supérieur à son "
                       f"niveau naturel.")
        elif effort_gap > 0:
            st.success(f"✅ L'effort gap $c^* - \\bar{{c}} = "
                       f"{effort_gap:.3f}$ est **tolérable** : "
                       f"la relation a de bonnes chances "
                       f"de durer.")
        else:
            st.error("❌ Effort gap négatif : vérifier "
                     "les paramètres.")

st.markdown("---")
st.markdown("""
### 📚 Références
- Rey J-M (2010). *A Mathematical Model of Sentimental Dynamics 
  Accounting for Marital Dissolution*. PLoS ONE 5(3): e9881.
- Chazel F. (2025). *L'amour en équation(s)*. Bureau d'études 
  3MIC-MA, INSA Toulouse.

*Application réalisée sous Python/Streamlit dans le cadre 
du Bureau d'Études 3MIC-MA, INSA Toulouse, 2025-2026.*
""")
