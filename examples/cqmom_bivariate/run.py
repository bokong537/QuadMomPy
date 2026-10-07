"""
Demonstration of the conditional quadrature method of moments (CQMOM) for a
two-dimensional internal coordinate, see Ref. [1].

The test case is a bivariate log-normal distribution in (size, composition)
space, which is a common stand-in for the number density function of a
crystallizer operating under non-uniform supersaturation or mixing. Since the
raw moments of a log-normal distribution are available in closed form, the
accuracy of the reconstructed sparse quadrature can be checked against an exact
reference for every moment order.

Two numerical properties of CQMOM matter in practice and are reported by this
script:

* The moment set has to be non-dimensionalised. Moment tensors of physical
  distributions span many orders of magnitude (crystal size in metres gives
  m_5 ~ 1e-20), and the Hankel/Jacobi inversion underlying QMOM loses all
  precision in that case - it fails with `LinAlgError: Eigenvalues did not
  converge` long before the quadrature itself would be inaccurate. Scale each
  coordinate by a reference value (`x_ref`, `y_ref` below) and convert the
  nodes back after the inversion.

* A conditional quadrature with N nodes per dimension does not reproduce every
  entry of a moment tensor of order 2N-1. The per-order error table printed here
  shows the structure: the marginal and low-order mixed moments are recovered to
  machine precision, while the highest orders carry the residual that the sparse
  quadrature cannot represent.

References
----------
.. [1] Yuan, Fox, "Conditional quadrature method of moments for kinetic
       equations", J. Chem. Phys. 139 (2013) 114103.

"""
import numpy as np
from quadmompy import qbmm

# Input file defining `n_dims`, `qbmm_type` and the per-dimension inversions
setup_file = "cqmom_setup.py"

# Output files
quad_out_file = "quad_out.dat"
err_out_file = "mom_err.dat"

# Parameters of the bivariate log-normal distribution, given in the
# non-dimensional coordinates xi = x / x_ref, psi = y / y_ref
mu = np.array([0.0, 0.0])       # log-mean
sigma = np.array([0.2, 0.2])    # log-standard deviation
rho = 0.6                       # correlation coefficient of the log-variables

# Reference scales used to render the dimensionless result dimensional again
x_ref = 1.0e-4      # crystal size scale [m]
y_ref = 1.0         # composition scale [-]

# Quadrature nodes per dimension. `2 * n_nodes[i]` moments are required in
# dimension i.
n_nodes = [3, 3]
n_mom = [2*nn for nn in n_nodes]


# Raw moments E[xi^a psi^b] of the bivariate log-normal distribution
mom = np.empty(n_mom)
for a in range(n_mom[0]):
    for b in range(n_mom[1]):
        va, vb = a*sigma[0], b*sigma[1]
        var = va**2 + vb**2 + 2.0*rho*va*vb
        mom[a, b] = np.exp(a*mu[0] + b*mu[1] + 0.5*var)

# Normalise the number density; m[0, 0] = 1 afterwards
mom /= mom[0, 0]

# Create the CQMOM object from file and invert the moment set
cqmom = qbmm.from_file(setup_file)
quad = cqmom.moment_inversion(mom)

# Compare every moment of the reconstructed quadrature with the exact value
err = np.empty(n_mom)
for a in range(n_mom[0]):
    for b in range(n_mom[1]):
        err[a, b] = abs(quad.reconstruct((a, b))/mom[a, b] - 1.0)

print(f"{quad.xi.shape[0]} quadrature nodes for {n_mom} moments per dimension")
print("max relative moment error  : %.2e" % err.max())
print("relative error of m[0,0]   : %.2e" % err[0, 0])
print("relative error of mixed m[1,1] : %.2e" % err[1, 1])
print()
print("relative moment error, rows = size order, columns = composition order")
np.set_printoptions(precision=1, suppress=False, linewidth=100)
print(err)
print()

# Some distribution properties, recovered from the quadrature and from the
# closed-form moments, to check the physical content
mom_x_1 = quad.reconstruct((1, 0))
mom_y_1 = quad.reconstruct((0, 1))
mom_x_2 = quad.reconstruct((2, 0))
mom_y_2 = quad.reconstruct((0, 2))
mom_xy = quad.reconstruct((1, 1))
sd_x = (mom_x_2 - mom_x_1**2)**0.5
sd_y = (mom_y_2 - mom_y_1**2)**0.5
corr = (mom_xy - mom_x_1*mom_y_1)/(sd_x*sd_y)

# Exact mean, standard deviation and covariance of a log-normal variable pair
ex_mx = np.exp(mu[0] + 0.5*sigma[0]**2)
ex_my = np.exp(mu[1] + 0.5*sigma[1]**2)
ex_sdx = ex_mx*(np.exp(sigma[0]**2) - 1.0)**0.5
ex_sdy = ex_my*(np.exp(sigma[1]**2) - 1.0)**0.5
ex_corr = (np.exp(rho*sigma[0]*sigma[1]) - 1.0) \
        / ((np.exp(sigma[0]**2) - 1.0)*(np.exp(sigma[1]**2) - 1.0))**0.5

print("mean size            : %.6e m   (exact %.6e m)"
      % (mom_x_1*x_ref, ex_mx*x_ref))
print("std. deviation size  : %.6e m   (exact %.6e m)"
      % (sd_x*x_ref, ex_sdx*x_ref))
print("mean composition     : %.6f       (exact %.6f)" % (mom_y_1, ex_my))
print("size-composition corr: %.6f       (exact %.6f)" % (corr, ex_corr))

# Write nodes and weights, dimensional, for `plot.py` and further analysis
header = "xi_x xi_y weight    # non-dimensional nodes and weights, multiply" \
         + " column 1 by x_ref = %.3e and column 2 by y_ref = %.3e" \
         % (x_ref, y_ref)
np.savetxt(quad_out_file, np.column_stack((quad.xi, quad.w)), header=header)
np.savetxt(err_out_file, err, header="relative moment error, rows = order in"\
           " x, columns = order in y")
