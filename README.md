# Galaxy cluster mass profiles from weak gravitational lensing assuming spherical symmetry

Julia package to infer galaxy cluster mass profiles from weak-lensing data assuming only spherical symmetry (and not assuming a specific mass profile such as an NFW profile).
See [arXiv:2408.07026](https://arxiv.org/abs/2408.07026) for details.

If you want to use this from Python, you can do so with [juliacall](https://juliapy.github.io/PythonCall.jl/stable/), see the [examples folder](./examples/python).

## Installation

You can install this package using the julia package manager

```julia
pkg> add SphericalClusterMass
```

## Usage

First import the package. We also import `Unitful` and `UnitfulAstro` for later use of units and `LinearAlgebra` to conveniently construct diagonal matrices.

```julia
using SphericalClusterMass
using Unitful
using UnitfulAstro
using LinearAlgebra
```

Then you can get the deprojected mass from the azimuthally averaged tangential reduced shear $G_+ = \langle g_+ \Sigma_{\mathrm{crit}} \rangle$ and the azimuthally averaged inverse critical surface density $f_c = \langle \Sigma_{\mathrm{crit}}^{-1} \rangle$
(alternatively, if individual source redshifts are not available, you can use $G_+ = \langle g_+ \rangle / \langle \Sigma_{\mathrm{crit}}^{-1} \rangle$ and $f_c = \langle \Sigma_{\mathrm{crit}}^{-2} \rangle / \langle \Sigma_{\mathrm{crit}}^{-1} \rangle$).
The following code calculates the deprojected mass profile and the associated covariance matrix,

```julia
R=[.2, .5, .7] .* u"Mpc"
result = calculate_M_and_covariance_in_bins(
    # Observational data.
    R=R,
    G=1e3 .* [.3, .2, .1] .* u"Msun/pc^2",
    f=1e-3 .* [.9, .9, .9] ./ u"Msun/pc^2",
    # Covariance matrix.
    # This corresponds to no actual covariance, just 10% statistical uncertainties on G
    G_covariance=diagm((1e2 .* [.3, .2, .1] .* u"Msun/pc^2") .^ 2),
    # Extrapolate beyond last data point assuming G ~ 1/R.
    # Corresponds to a singular isothermal sphere.
    # To extrapolate assuming an NFW profile, see `ExtrapolateNFW` below.
    extrapolate=ExtrapolatePowerDecay(1),
    # Interpolate linearly between discrete data points in "R-space".
    # - Quadratic interpolation is also a good choice, but a bit slower.
    # - To interpolate in "ln(R)-space", use `InterpolateLnR(..)`.
    #   This may be the better choice for logarithmic bins.
    interpolate=InterpolateR(1),
)
```

This will run for a few seconds on the first run to compile the code.
Subsequent runs will be fast, unless the number of data points changes, which requires recompilation.
`result` is a named tuple with fields `M`, `M_stat_cov` and `M_stat_err` where `M` refers to the deprojected 3D mass $M(r)$.
Let's have a look this mass profile and its error bars,
```julia
using Plots
plot(R, result.M, yerror=result.M_stat_err)
```
and its correlation matrix
```julia
heatmap(result.M_stat_cov ./ (result.M_stat_err * result.M_stat_err'))
```

### Faster calculation assuming constant $f_c$

In practice, $f_c$ is often reasonably constant as a function of radius.
A constant $f_c$ makes the calculation simpler and faster.
This faster calculation will automatically be done if $f_c$ is passed as a scalar instead of a vector,

```julia
result = calculate_M_and_covariance_in_bins(
    R=R,
    G=1e3 .* [.3, .2, .1] .* u"Msun/pc^2",
    f=1e-3 * .9 / u"Msun/pc^2", # <-- This is now a scalar instead of a vector
    G_covariance=diagm((1e2 .* [.3, .2, .1] .* u"Msun/pc^2") .^ 2),
    extrapolate=ExtrapolatePowerDecay(1),
    interpolate=InterpolateR(1),
)
```

### Faster calculation without covariance matrix

If one is not interested in the statistical uncertainties and covariances, one can use `calculate_M`

```julia
result = calculate_M(
    R=R,
    G=1e3 .* [.3, .2, .1] .* u"Msun/pc^2",
    f=1e-3 .* [.9, .9, .9] ./ u"Msun/pc^2",
    extrapolate=ExtrapolatePowerDecay(1),
    interpolate=InterpolateR(1),
).(R ./ u"Mpc")
```

If one has $f_c = \mathrm{const}$, one can also pass `f` as a scalar

```julia
result = calculate_M(
    R=R,
    G=1e3 .* [.3, .2, .1] .* u"Msun/pc^2",
    f=1e-3 * .9 / u"Msun/pc^2", # <-- This is now a scalar instead of a vector
    extrapolate=ExtrapolatePowerDecay(1),
    interpolate=InterpolateR(1),
).(R ./ u"Mpc")
```

### Correct for miscentering

To correct for miscentering (to leading order in $R_{\mathrm{mc}}/R$ where $R_{\mathrm{mc}}$ is the miscentering radius), you can use the `miscenter_correct` argument:

```julia
result = calculate_M_and_covariance_in_bins(
    R=R,
    G=1e3 .* [.3, .2, .1] .* u"Msun/pc^2",
    f=1e-3 .* [.9, .9, .9] ./ u"Msun/pc^2",
    G_covariance=diagm((1e2 .* [.3, .2, .1] .* u"Msun/pc^2") .^ 2),
    extrapolate=ExtrapolatePowerDecay(1),
    interpolate=InterpolateR(1),
    # This will (approximately) correct for miscentering by `Rmc²`.
    # The uncertainty `σ_Rmc²` on `Rmc²` will be propagated into the covariance matrix.
    # The default is `MiscenterCorrectNone()`, which does not correct for miscentering.
    #
    # Note: The "naive" way to implement this miscentering correction (just following
    #       equations (22)+(23) in the paper) involves calculating numerical 2nd order
    #       derivatives, which can be dicey. Therefore, the implementation here avoids
    #       this. This implementation is not exactly equivalent to equations (22)+(23)
    #       from the paper. It is, however, equivalent up to terms of order κ*(Rmc/R)^2
    #       (which are beyond the order of approximation of (22)+(23)). Details will be
    #       published in future work, but I'm happy to privately explain more in the
    #       meantime.
    #       To use the "naive" way of calculating the miscentering correction, use
    #       `MiscenterCorrectSmallRmcPreprocessG` instead of `MiscenterCorrectSmallRmc`.
    miscenter_correct=MiscenterCorrectSmallRmc(
        Rmc²=(.16u"Mpc")^2,
        σ_Rmc²=(.16u"Mpc")^2,
    )
)
```

The function `calculate_M` also supports the `miscenter_correct` argument.

### Extrapolate assuming NFW profile

To extrapolate assuming an NFW profile use `ExtrapolateNFW(cm)` where `cm` describes a mass-concentration relation.
For now, only one relation from Maccio et al. 2008 is supported, but others can be added easily.

```julia
h=.7
ρcrit = 1.85e11u"Msun/Mpc^3"
result = calculate_M_and_covariance_in_bins(
    R=R,
    G=1e3 .* [.3, .2, .1] .* u"Msun/pc^2",
    f=1e-3 .* [.9, .9, .9] ./ u"Msun/pc^2",
    G_covariance=diagm((1e2 .* [.3, .2, .1] .* u"Msun/pc^2") .^ 2),
    # This will continue with an NFW-like profile matched to the last data point,
    # assuming the Maccio et al 2008 mass-concentration relation for a specific Hubble
    # constant h = H0/(100 km s⁻¹ Mpc⁻¹) and critical density at the redshift of interest.
    extrapolate=ExtrapolateNFW(CMRelationMaccio2008(ρcrit, h)),
    interpolate=InterpolateR(1),
)
```

