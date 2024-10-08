module SphericalClusterMass

include("cluster_deproject_notebook.jl")

export calculate_gobs_fconst, calculate_gobs_fgeneral, calculate_gobs_and_covariance_in_bins
export ExtrapolatePowerDecay, InterpolateR, InterpolateLnR, MiscenterCorrectNone, MiscenterCorrectSmallRmc

end # module SphericalClusterMass
