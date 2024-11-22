# Galaxy cluster mass profiles from weak gravitational lensing assuming spherical symmetry

Package to infer galaxy cluster mass profiles from weak-lensing data assuming only spherical symmetry (and not assuming a specific mass profile such as an NFW profile).
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

Then you can get the deprojected mass from the azimuthally averaged tangential reduced shear $G_+ = \langle g_+ \Sigma_{\mathrm{crit}} \rangle$ and the azimuthally averaged inverse critical surface density $f_c = \langle \Sigma_{\mathrm{crit}}^{-1} \rangle$
(alternatively, if individual source redshifts are not available, you can use $G_+ = \langle g_+ \rangle / \langle \Sigma_{\mathrm{crit}}^{-1} \rangle$ and $f_c = \langle \Sigma_{\mathrm{crit}}^{-2} \rangle / \langle \Sigma_{\mathrm{crit}}^{-1} \rangle$).
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
    # Interpolate linearly between discrete data points in "R-space".
    # - Quadratic interpolation is also a good choice, but a bit slower.
    # - To interpolate in "ln(R)-space", use `InterpolateLnR(..)`.
    #   This may be the better choice for logarithmic bins.
    interpolate=InterpolateR(1),
)
```

This will run for a few seconds on the first run to compile the code.
Subsequent runs will be fast, unless the number of data points changes, which requires recompilation.
`result` is a named tuple with fields `gobs`, `gobs_stat_cov` and `gobs_stat_err` where `gobs` refers to the acceleration $G M(r) /r^2$.
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
    interpolate=InterpolateR(1),
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
    interpolate=InterpolateR(1),
).(R ./ u"Mpc")
```

or, for the case assuming $f_c = \mathrm{const}$, one can use  `calculate_gobs_fconst`,


```julia
result = calculate_gobs_fconst(
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
result = calculate_gobs_and_covariance_in_bins(
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
    # Note: The "naive" (just following the formula in the paper) way to implement this
    #       miscentering correction involves calculating numerical 2nd order derivatives.
    #       By default, the implementation here avoids this. This is equivalent to the
    #       "naive" way up to terms of order κ*(Rmc/R)^2 (that are neglected anyway).
    #       To use the "naive" way of calculating the miscentering correction, use
    #       `MiscenterCorrectSmallRmcPreprocessG` instead of `MiscenterCorrectSmallRmc`.
    #       (The optional parameter ϵ is not available in this case.)
    miscenter_correct=MiscenterCorrectSmallRmc(
        Rmc²=(.16u"Mpc")^2,
        σ_Rmc²=(.16u"Mpc")^2,
        # Optional parameter to control numerical procedure. Make smaller for a slower
        # and more precise calculation (but keep larger than available numerical precision!)
        # ϵ=.001
    )
)
```

The functions  `calculate_gobs_fconst` and `calculate_gobs_fgeneral` also support the `miscenter_correct` argument.
