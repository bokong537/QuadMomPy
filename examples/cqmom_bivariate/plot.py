"""
This script can be used to plot the results from `run.py`.

Panel 1 shows the conditional quadrature nodes recovered by CQMOM against the
exact bivariate log-normal density, panel 2 shows how accurately the sparse
quadrature represents each moment order.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import multivariate_normal

# Input files written by `run.py`
quad_file = "quad_out.dat"
err_file = "mom_err.dat"

# Parameters of the exact density, has to match `run.py`
mu = np.array([0.0, 0.0])       # log-mean
sigma = np.array([0.2, 0.2])    # log-standard deviation
rho = 0.6                       # correlation coefficient of the log-variables

x_ref = 1.0e-4      # crystal size scale [m]

xi, psi, w = np.genfromtxt(quad_file, unpack=True)
err = np.genfromtxt(err_file)

# Exact density in the dimensional coordinates, from the log-normal density of
# (ln xi, ln psi) by change of variables
cov = np.array([[sigma[0]**2, rho*sigma[0]*sigma[1]],
                [rho*sigma[0]*sigma[1], sigma[1]**2]])
grid_x = np.linspace(0.55*x_ref, 1.9*x_ref, 200)
grid_p = np.linspace(0.5, 1.8, 200)
X, P = np.meshgrid(grid_x, grid_p)
lnpdf = multivariate_normal(mean=mu, cov=cov).pdf(
        np.column_stack((np.log(X/x_ref).ravel(), np.log(P).ravel())))
pdf = lnpdf.reshape(X.shape)/((X/x_ref)*P)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))

cont = ax[0].contourf(X, P, pdf, levels=14, cmap="viridis", alpha=0.75)
ax[0].scatter(x_ref*xi, psi, s=200*w/w.max(), c="w", edgecolors="k",
              linewidths=1.2, zorder=3, label="CQMOM nodes")
ax[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: "%.1f" % (v/x_ref)))
ax[0].set_xlabel("size $x$ [$10^{-4}$ m]")
ax[0].set_ylabel("composition $y$ [-]")
ax[0].set_title("Conditional quadrature (%d nodes)" % len(w))
ax[0].legend(loc="upper right", framealpha=0.9)
ax[0].grid(which="both", ls=":", lw=0.5, alpha=0.6)
fig.colorbar(cont, ax=ax[0], label="number density")

im = ax[1].imshow(np.log10(np.maximum(err, 1e-17)), origin="lower",
                  cmap="magma", aspect="auto")
ax[1].set_xlabel("order in composition $y$")
ax[1].set_ylabel("order in size $x$")
ax[1].set_title("Relative moment error")
ax[1].set_xticks(range(err.shape[1]))
ax[1].set_yticks(range(err.shape[0]))
for a in range(err.shape[0]):
    for b in range(err.shape[1]):
        ax[1].text(b, a, "%.0e" % err[a, b], ha="center", va="center",
                   fontsize=7, color="w")
fig.colorbar(im, ax=ax[1], label="log10 relative error")

plt.tight_layout()
plt.show()
