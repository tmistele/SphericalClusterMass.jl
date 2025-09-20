import astropy.units as u
import numpy as np
from juliacall import Main as jl, convert as jlconvert
from collections import namedtuple


# Set up julia
jl.seval("using Unitful")
jl.seval("using UnitfulAstro")
jl.seval("using SphericalClusterMass")
jl_Mpc = jl.seval('u"Mpc"')
jl_Msun_pc2 = jl.seval('u"Msun/pc^2"')
jl_Msun = jl.seval('u"Msun"')


MAndCovarianceInBins = namedtuple('MAndCovarianceInBins', [
    'M', 'M_stat_err', 'M_stat_cov'
])


def calculate_M_and_covariance_in_bins(
    R, G, f, G_covariance, extrapolate, interpolate
):
    # Convert from astropy units to julia Unitful units
    jl_R = jlconvert(jl.Vector, R.to(u.Mpc).value) * jl_Mpc
    jl_G = jlconvert(jl.Vector, G.to(u.Msun / u.pc**2).value) * jl_Msun_pc2
    if len(f.shape) == 0:
        # Constant f
        jl_f = jlconvert(jl.Float64, f.to(u.pc**2 / u.Msun).value) / jl_Msun_pc2
    else:
        # Non-constant f
        jl_f = jlconvert(jl.Vector, f.to(u.pc**2 / u.Msun).value) / jl_Msun_pc2
    jl_G_covariance = (
        jlconvert(jl.Matrix, G_covariance.to((u.Msun / u.pc**2) ** 2).value)
        * jl_Msun_pc2
        * jl_Msun_pc2
    )

    result = jl.calculate_M_and_covariance_in_bins(
        R=jl_R,
        G=jl_G,
        f=jl_f,
        G_covariance=jl_G_covariance,
        extrapolate=extrapolate,
        interpolate=interpolate,
    )

    return MAndCovarianceInBins(
        M=(result.M / jl_Msun).to_numpy() * u.Msun,
        M_stat_err=(result.M_stat_err / jl_Msun).to_numpy() * u.Msun,
        M_stat_cov=(result.M_stat_cov / jl_Msun / jl_Msun).to_numpy() * u.Msun**2
    )
