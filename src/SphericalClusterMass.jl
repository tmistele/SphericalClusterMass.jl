module SphericalClusterMass

include("cluster_deproject_notebook.jl")

export calculate_M, calculate_M_and_covariance_in_bins, calculate_ρ_and_covariance_in_bins
export ExtrapolatePowerDecay, ExtrapolateNFW, CMRelationMaccio2008, InterpolateR, InterpolateLnR, MiscenterCorrectNone, MiscenterCorrectSmallRmc, NoNeglectKappa, NeglectKappa

end # module SphericalClusterMass
