# Galaxy cluster mass profiles from gravitational lensing assuming spherical symmetry

Package to infer galaxy cluster mass profiles from lensing data assuming only spherical symmetry (and not assuming a specific mass profile such as an NFW profile).
See [arXiv:2408.07026](https://arxiv.org/abs/2408.07026) for details.

## Installation

To use this package, first install it using the julia package manager.
It's not yet uploaded to the official julia package repository.
As a workaround it's possible to install like this

```julia
pkg> add https://github.com/tmistele/SphericalClusterMass.jl#master
```

## Usage

First import the package. We also import `Unitful` and `UnitfulAstro` for later use of units and `LinearAlgebra` to conveniently construct diagonal matrices.

```julia
using SphericalClusterMass
using Unitful
using UnitfulAstro
using LinearAlgebra
```

Then you can get the deprojected mass given the azimuthally averaged tangential reduced shear $G_+ = \langle g_+ \Sigma_{\mathrm{crit}} \rangle$ and azimuthally averaged inverse critical surface density $f_c = \langle \Sigma_{\mathrm{crit}}^{-1} \rangle$.
The following code calculates (a form of) the deprojected mass profile and the associated covariance matrix,

```julia
R=[.2, .5, .7] .* u"Mpc"
result = calculate_gobs_and_covariance_in_bins(
    # Observational data.
    R=R,
    G=1e3 .* [.3, .2, .1] .* u"Msun/pc^2",
    f=1e-3 .* [.9, .9, .9] ./ u"Msun/pc^2",
    # Covariance matrix.
    # This corresponds to no actual covariance, just 10% statistical uncertainties on G
    G_covariance=diagm((1e2 .* [.3, .2, .1] .* u"Msun/pc^2") .^ 2),
    # Extrapolate beyond last data point assuming G ~ 1/R.
    # Corresponds to a singular isothermal sphere.
    extrapolate=ExtrapolatePowerDecay(1),
    # Interpolate linearly between discrete data points.
    # Quadratic interpolation is also a good choice, but a bit slower.
    interpolation_order=1,
)
```

This will run for a few seconds on the first run to compile the code.
Subsequent runs will be fast, unless the number of data points changes, which requires recompilation.
`result` is a named tuple with fields `gobs`, `gobs_stat_cov` and `gobs_stat_err` where `gobs` refers to the acceleration $G M/r^2$.
The deprojected mass $M$ and its statistical uncertainties and covariance matrix can be inferred from this.
For example,

```julia
M = result.gobs .* R .^ 2 ./ u"G" .|> u"Msun"
M_stat_err = result.gobs_stat_err .* R .^ 2 ./ u"G" .|> u"Msun"

using Plots
plot(R, M, yerror=M_stat_err)
```

### Faster calculation assuming constant $f_c$

In practice, $f_c$ is often reasonably constant as a function of radius.
A constant $f_c$ makes the calculation simpler and faster.
This faster calculation will automatically be done if $f_c$ is passed as a scalar instead of a vector,

```julia
result = calculate_gobs_and_covariance_in_bins(
    R=R,
    G=1e3 .* [.3, .2, .1] .* u"Msun/pc^2",
    f=1e-3 * .9 / u"Msun/pc^2", # <-- This is now a scalar instead of a vector
    G_covariance=diagm((1e2 .* [.3, .2, .1] .* u"Msun/pc^2") .^ 2),
    extrapolate=ExtrapolatePowerDecay(1),
    interpolation_order=1,
)
```

### Faster calculation without covariance matrix

If one is not interested in the statistical uncertainties and covariances, one can use `calculate_gobs_fgeneral`

```julia
result = calculate_gobs_fgeneral(
    R=R,
    G=1e3 .* [.3, .2, .1] .* u"Msun/pc^2",
    f=1e-3 .* [.9, .9, .9] ./ u"Msun/pc^2",
    extrapolate=ExtrapolatePowerDecay(1),
    interpolation_order=1,
).(R ./ u"Mpc")
```

or, for the case assuming $f_c = \mathrm{const}$, one can use  `calculate_gobs_fconst`,


```julia
result = calculate_gobs_fconst(
    R=R,
    G=1e3 .* [.3, .2, .1] .* u"Msun/pc^2",
    f=1e-3 * .9 / u"Msun/pc^2", # <-- This is now a scalar instead of a vector
    extrapolate=ExtrapolatePowerDecay(1),
    interpolation_order=1,
).(R ./ u"Mpc")
```
