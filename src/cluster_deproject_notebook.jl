### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 4bdde00d-5d78-45e5-8c4e-a790f7431a3c
let
	# @plutoonly doesn't work here.
	# (The `using ...` cell below is evaluated before this `Pkg` cell if I use it)
	# So just directly use `@isdefined PlutoRunner`
	if @isdefined PlutoRunner
		# https://github.com/fonsp/Pluto.jl/wiki/%F0%9F%8E%81-Package-management#pattern-the-shared-environment
		import Pkg
		# activate the shared project environment
		Pkg.activate(Base.current_project())
		# instantiate, i.e. make sure that all packages are downloaded
		Pkg.instantiate()
	end
end

# ╔═╡ d7ce151b-4732-48ea-a8a5-5bfbe94d119b
begin
	using OrdinaryDiffEq
	using StaticArrays
	import BSplineKit
	using QuadGK
	using Unitful
	using UnitfulAstro
	using ForwardDiff
	using LinearAlgebra
	using Plots
end

# ╔═╡ 1ae70636-b3ce-4ac7-b827-e8ec615bde29
module old_gobs_from_ΔΣ
	using Unitful
	using UnitfulAstro
	using Dierckx

	abstract type AbstractExtrapolateΔΣ end
	struct ExtrapolateΔΣSIS <: AbstractExtrapolateΔΣ end

	gobs_analytical_tail_factor(ex::ExtrapolateΔΣSIS;
			RoverRmax, RMpcMax, last_RMpc_bin_edge) = let
		(1/RoverRmax)*(1 - sqrt(1-RoverRmax^2))
	end

	function calculate_gobs(; R, ΔΣ̂, last_RMpc_bin_edge,
			interpolation_order=1, extrapolate=ExtrapolateΔΣSIS())
	
		RMpcvals = R ./ u"Mpc" .|> NoUnits
		RMpcMax = RMpcvals[end]
	
		@assert RMpcMax <= last_RMpc_bin_edge "Rmax is larger than last bin edge?!"
	
		prefactors = 4*u"G"*u"Msun/pc^2" .|> u"m/s^2"
	
		if length(ΔΣ̂) == 1
			# Cannot construct ΔΣ̂func below if there's only one data point
			return [prefactors * ΔΣ̂[end] * gobs_analytical_tail_factor(extrapolate;
				RoverRmax=1.0, RMpcMax=RMpcMax, last_RMpc_bin_edge=last_RMpc_bin_edge)]
		end
	
		ΔΣ̂func = Spline1D(
			RMpcvals, ΔΣ̂,
			# Just b/c of floating point inexactness. We don't actually need ΔΣ̂ outside
			# where it's defined
			bc="nearest",
			k=interpolation_order
		)
	
		integrals = [
			let
				RMpc = RMpcvals[i]
				RoverRmax = RMpc/RMpcMax
	
				# The interal up to Rmax
				res_numerical_integral = if i == length(RMpcvals)
					# At the last data point, ther's only the tail left
					# Exlicit branch here b/c `Spline1D(...)` fails when we
					# give just a single value in θs (I think?)
					0
				else
					# From where on we do the numerical integral
					# (where R/sin θ = Rmax)
					θmin = asin(RoverRmax)
	
					# Where we evaluate the integrand
					# (More points mean smaller error, but slower)
					θs = LinRange(θmin, π/2, 20)
	
					# ΔΣ(R/sin θ)
					integrand = Spline1D(
						θs, ΔΣ̂func.(RMpc ./ sin.(θs)),
						bc="error",
						k=1 # Keep this k=1 even for larger `interpolation_order`
					)
					
					# This (from DierckX) is faster than quadgk
					integrate(integrand, θs[begin], θs[end])
				end
				
				# Analytical rest of the integral beyond Rmax
				res_analytical_tail = ΔΣ̂[end] * gobs_analytical_tail_factor(extrapolate;
					RoverRmax=RoverRmax, RMpcMax=RMpcMax,
					last_RMpc_bin_edge=last_RMpc_bin_edge,
				)
	
				# Return the full integral
				res_numerical_integral+res_analytical_tail
			end
			for i in 1:length(RMpcvals)
		]
		prefactors .* integrals
	end

	function fast_gobs_Cαi(; R::typeof([1.0u"Mpc"]), C, α::Int64,
			extrapolate::E,
			last_RMpc_bin_edge::Float64
		) where {E <: AbstractExtrapolateΔΣ}
		Δθ_αi(α, i) = asin(R[α]/R[i]) - asin(R[α]/R[i+1])
	
		f_αi(α, i) = (
			-R[α] * atanh(sqrt(1 - (R[α]/R[i])^2 ))
			+R[α] * atanh(sqrt(1 - (R[α]/R[i+1])^2 ))
			-R[i] * Δθ_αi(α, i)
		) / (R[i+1] - R[i])
	
		f_cont(α) = gobs_analytical_tail_factor(
			extrapolate;
			RoverRmax=R[α]/R[end],
			RMpcMax=R[end] / u"Mpc" |> NoUnits,
			last_RMpc_bin_edge=last_RMpc_bin_edge
		)
		
		@assert length(C) == length(R)
		
		# Leave the C[begin:α-1] part of C untouched for perf (don't zero out or so)
		if α < length(C)
			C[α] = Δθ_αi(α, α) - f_αi(α, α)
		end
		for i in α+1:length(C)-1
			C[i] = Δθ_αi(α, i) - f_αi(α, i) + f_αi(α, i-1)
		end
		if α == length(C)
			C[length(C)] = f_cont(α)
		else
			C[length(C)] = f_cont(α) + f_αi(α, length(C)-1)
		end
		nothing
	end
	function calculate_gobs_fast(;  R, ΔΣ̂,
			extrapolate=ExtrapolateΔΣSIS(), last_RMpc_bin_edge::Float64)
		prefactor = 4*u"G"*u"Msun/pc^2" |> u"m/s^2"
		gobs = fill(NaN*u"m/s^2", length(R))
		C = zeros(length(R)) # Don't allocate in loop
		for α in eachindex(gobs)
			fast_gobs_Cαi(
				R=R, C=C, α=α,
				extrapolate=extrapolate, last_RMpc_bin_edge=last_RMpc_bin_edge
			)
			gobs[α] = prefactor * sum(C[i]*ΔΣ̂[i] for i in α:length(C))
		end
		gobs
	end
	function calculate_gobs_staterr_fast(; R, σ²_ΔΣ̂,
			# NOTE: Here `extrapolate` is that of the `gobs` for which we now shall
			#       calculate the error.
			extrapolate=ExtrapolateΔΣSIS(), last_RMpc_bin_edge::Float64)
		prefactor = 4*u"G"*u"Msun/pc^2" |> u"m/s^2"
		σ_gobs = fill(NaN*u"m/s^2", length(R))
		C = zeros(length(R)) # Don't allocate in loop
		for α in eachindex(σ_gobs)
			fast_gobs_Cαi(
				R=R, C=C, α=α,
				extrapolate=extrapolate, last_RMpc_bin_edge=last_RMpc_bin_edge
			)
			σ_gobs[α] = prefactor * sqrt(sum( C[i]^2*σ²_ΔΣ̂[i] for i in α:length(C)) )
		end
		σ_gobs
	end
	function to_bin_centers(edges)
		widths = circshift(edges, -1) .- edges
		( edges .+ (widths./2) )[1:end-1]
	end
	function calculate_gobs_covariance_fast(;
			# NOTE: Here `extrapolate` is that of the `gobs` for which we now shall
			#       calculate the error.
			extrapolate=ExtrapolateΔΣSIS(),
			σ²_ΔΣ̂_l,
			w̄l_unnormalized,
			∑ₗ_w̄l_unnormalized,
			l_r_bin_edges,
			out::Matrix{Float64}
		)
	
		prefactor = (4*u"G"*u"Msun/pc^2")^2 ./ u"(m/s^2)^2" |> NoUnits
	
		l_len = size(σ²_ΔΣ̂_l, 1)
		Nbins = size(σ²_ΔΣ̂_l, 2)
		@assert size(out) == (Nbins, Nbins)
	
		cov_αβ(α, β) = let
			# These need to be inside cov_αβ so different threads have their own buffers
			# to write into
			Cα = zeros(Nbins) # Don't allocate in hot loop
			Cβ = zeros(Nbins)
			
			result = 0.0
			for l in 1:l_len
	
				# Zero weight means there were not sources at some radial bin for this l.
				# So we skipped that radial bin in the gobs calculation for that l.
				# Thus, the gobs_l in that radial bin cannot co-vary with anything.
				# It was just left out.
				if w̄l_unnormalized[l, α] == 0 || w̄l_unnormalized[l, β] == 0
					continue
				end
	
				# Radial bins where we found sources (may not be the case at small radii)
				# In the gobs calculation, we interpolated between only the radial bins
				# that do have signal. For the others we just pretended there was no 
				# radial bin there.
				# So do the same here.
				idx = (@view w̄l_unnormalized[l, :]) .> 0
	
				R = let
					r_bin_centers = to_bin_centers(@view l_r_bin_edges[l, :])
					r_bin_centers[idx]
				end
				last_RMpc_bin_edge = l_r_bin_edges[l, findlast(idx)+1] ./ u"Mpc"
	
				num_missing_before_α = count(x -> !x, @view idx[begin:α])
				num_missing_before_β = count(x -> !x, @view idx[begin:β])
	
				Cα[:] .= 0.0
				Cβ[:] .= 0.0
				fast_gobs_Cαi(
					R=R, C=(@view Cα[idx]),
					# The `α` named parameter must now denote the position of the original
					# radial bin in the reduced set of bins R.
					α=α-num_missing_before_α,
					extrapolate=extrapolate, last_RMpc_bin_edge=last_RMpc_bin_edge
				)
				fast_gobs_Cαi(
					R=R, C=(@view Cβ[idx]),
					# The `α` named parameter must now denote the position of the original
					# radial bin in the reduced set of bins R.
					α=β-num_missing_before_β,
					extrapolate=extrapolate, last_RMpc_bin_edge=last_RMpc_bin_edge
				)
				term = w̄l_unnormalized[l, α]*w̄l_unnormalized[l, β]
				term *= sum(Cα[i]*Cβ[i]*σ²_ΔΣ̂_l[l, i] for i in max(α, β):length(Cα))
				
				result += term
			end
			
			prefactor*result / (∑ₗ_w̄l_unnormalized[α] * ∑ₗ_w̄l_unnormalized[β])
		end
	
		Threads.@sync for (α, β) in Iterators.product(1:Nbins, 1:Nbins)
			Threads.@spawn let
				# Don't unnecessarily calculate off-diagonal elements twice.
				# It's symmetric.
				if α < β
					# intent is `continue`. Because of Threads.@spawn this then needs to
					# be a `return` instead
					return
				end
				result = cov_αβ(α, β)
				out[α, β] = result
				out[β, α] = result
			end
		end
		nothing
	end
end

# ╔═╡ 3f004698-b952-462f-8824-5c78ab1e08ad
module __demo
	using Unitful
	using UnitfulAstro

	# This is copied from the demo code I wrote for the paper.
	# Perhaps put the demo code properly into a sub-module or so here.
	abstract type AbstractProfile end
	function calculate_γ12(p::P, x, y; Σcrit) where P<:AbstractProfile
		R = sqrt( (x-p.x0)^2 + (y-p.y0)^2 )
		dϕ = calculate_dR_lenspot(p, R; Σcrit=Σcrit)
		ddϕ = calculate_ddR_lenspot(p, R; Σcrit=Σcrit)
		tmp = (R*ddϕ - dϕ)/R^3
		
		γ1 = .5 * ( (x-p.x0)^2 - (y-p.y0)^2 ) * tmp
		γ2 = (x-p.x0)*(y-p.y0) * tmp
		(γ1, γ2) .|> NoUnits
	end
	function calculate_γtx(p, x, y; Σcrit)
		(γ1, γ2) = calculate_γ12(p, x, y; Σcrit=Σcrit)
	
		R = sqrt(x^2 + y^2)
		cosφ = x/R
		sinφ = y/R
	
		# Angle sum identities (or use Mathematica, Cos[2φ]//TrigReduce)
		cos2φ = cosφ^2 - sinφ^2
		sin2φ = 2*cosφ*sinφ
	
		γt = - cos2φ * γ1 - sin2φ * γ2
		γx =   sin2φ * γ1 - cos2φ * γ2
	
		(γt, γx)
	end
	function calculate_κ(p::P, x, y; Σcrit) where P <:AbstractProfile
		R = sqrt( (x-p.x0)^2 + (y-p.y0)^2 )
		calculate_κ(p, R; Σcrit=Σcrit)
	end
	function calculate_azimuthally_averaged_gt(R, p; Σcritinv)
		
		Σcrit = 1/Σcritinv
		
		# 50 samples
		# `end-1` to make sure 2π is not included (do not double-count φ = 0 = 2π)
		# TODO: Make that depend on radius?? (constant source number density!)
		φvals = LinRange(0, 2π, 50+1)[1:end-1] |> collect
	
		xvals = R .* cos.(φvals)
		yvals = R .* sin.(φvals)
	
		# Get the (dimensionless) γ and κ
		γtx = calculate_γtx.(Ref(p), xvals, yvals; Σcrit=Σcrit)
		γt = map(x -> x[1], γtx)
		γx = map(x -> x[2], γtx)
		
		κ = calculate_κ.(Ref(p), xvals, yvals; Σcrit=Σcrit)
	
		# Don't shoot ourselves in the foot
		@assert all(κ .< .9) "Oups -- don't let κ be so big :) $(R) $(κ)"
	
		# Reduced shear
		gt = γt ./ (1 .- κ)
		gx = γx ./ (1 .- κ)
	
		# Azimuthal average
		gt = sum(gt) / length(gt)
		gx = sum(gx) / length(gx)
	
		@assert abs(gx) < 1e-15 "Cross-check: γx and gx should be zero"
	
		# Reduced tangential shear
		gt
	end
	struct ProfileSIS <: AbstractProfile
		M1Mpc::typeof(1.0u"Msun")
		x0::typeof(1.0u"Mpc")
		y0::typeof(1.0u"Mpc")
	end
	ProfileSIS(; M1Mpc, x0, y0) = ProfileSIS(
		M1Mpc,
		x0,
		y0
	)
	function calculate_κ(p::ProfileSIS, R; Σcrit)
		# ρ = M1Mpc/(4π*1Mpc*r^2)
		# NB: κ= Σ/Σcrit -- but lens equations have factor of 2: Δ ϕ = 2 κ
		a = p.M1Mpc/(4π*1u"Mpc") / Σcrit
		a*π/R |> NoUnits
	end
	function calculate_dR_lenspot(p::ProfileSIS, R; Σcrit)
		# ρ = M1Mpc/(4π*1Mpc*r^2)
		a = p.M1Mpc/(4π*1u"Mpc")
		
		2π * a/Σcrit
	end
	function calculate_ddR_lenspot(p::ProfileSIS, R; Σcrit)
		# ρ = M1Mpc/(4π*1Mpc*r^2)
		prefactor = p.M1Mpc/(4π*1u"Mpc")/Σcrit
		0*prefactor/1u"Mpc"
	end
end

# ╔═╡ 8f04c59b-a109-4032-9235-1acc6f8ad9b4
# We use this notebook both as a library for other code
# and a pluto notebook to have nice visual tests directly next to the code
# This macro can be used to eliminate the "tests" when this is used a a library.
macro plutoonly(block)
	if @isdefined PlutoRunner
		:($(esc(block)))
	end
end

# ╔═╡ 2bd8f9c4-ed93-406f-974e-3539d44f21c4
@plutoonly let
	import PlutoUI
	PlutoUI.TableOfContents()
end

# ╔═╡ 9269044a-217b-48ef-b6f1-266a75890956
begin
	# How to extrapolate G beyond last data point
	abstract type AbstractExtrapolate end
	struct ExtrapolatePowerDecay <: AbstractExtrapolate
		# Only positive values are allowed!
		n::Float64
	end

	# Hot to interpolate G, Gf, f, in between the discrete data points
	# NB: This also controls whether the integrals I(R) and J(R) are calculated from
	#     ODEs of the form dI/dR = ... (for linear-spce) or from ODEs of the form 
	#     dI/d ln R = ... (for log-space)
	abstract type AbstractInterpolate end
	struct InterpolateR <: AbstractInterpolate
		order::UInt8
	end
	struct InterpolateLnR <: AbstractInterpolate
		order::UInt8
	end

	# If and how to correct for miscentering
	abstract type AbstractMiscenterCorrect end
	# No correction
	struct MiscenterCorrectNone <: AbstractMiscenterCorrect end
	# Correct up to (including) O((Rmc/R)^2)
	# (but not including O(κ (Rmc/R)^2))
	struct MiscenterCorrectSmallRmc <: AbstractMiscenterCorrect
		Rmc²::typeof(1.0u"Mpc^2")
		# Note: That's the uncertainty on (Rmc^2)
		σ_Rmc²::typeof(1.0u"Mpc^2")
	end
	MiscenterCorrectSmallRmc(; Rmc², σ_Rmc²) = MiscenterCorrectSmallRmc(
		Rmc²,
		σ_Rmc²
	)
end

# ╔═╡ 52cadcf0-a9ae-4e91-ac44-21e6fd25dabc
md"""
# Input quantities / Notation

See for example the Umetsu (2020) lensing review, around equation (99)

$G \equiv \langle g_+ \Sigma_{\mathrm{cr}} \rangle \equiv \langle \Delta \Sigma_+\rangle$ 

and 

$f \equiv \langle\Sigma_{\mathrm{cr}}^{-1} \rangle$ 

where $\langle \dots \rangle$ denotes source average over all sources at sam projected distance $R$ (azimuthal average).

This is for a *single* cluster (not multiple clusters stacked).
"""

# ╔═╡ 6cc0e536-970a-4e94-8449-6c358c31b3ec
md"""
# Miscentering correction

For a given miscentering radius $R_{mc}$, we can correct for miscentering (by expanding in $R_{mc}/R$) using

$G_+ \to G_+ + \frac14 \left(\frac{R_{mc}}{R}\right)^2 (4 G_+(R) - R G_+'(R) - R^2 G_+''(R) )$

Note that:

$R G_+'(R) + R^2 G_+''(R) = \partial_{\ln R}^2 G_+$
"""

# ╔═╡ 61671b5e-9a09-49ed-ba69-37852662f803
begin
	function miscenter_correct_G(
		miscenter_correct::MiscenterCorrectNone, interpolate::I; R, G
	) where I<: AbstractInterpolate
		G
	end
	function miscenter_correct_G_covariance(
		miscenter_correct::MiscenterCorrectNone, interpolate::I; R, G, G_covariance
	) where I <: AbstractInterpolate
		G_covariance
	end

	function __calculate_miscenter_corrected_GMsunpc2_small_Rmc(
		interpolate::InterpolateR; RMpc, Rmc²Mpc, GMsunpc2
	)
		@assert length(RMpc) == length(GMsunpc2)
		@assert length(RMpc) >= 3 "Need at least 3 data points for `MiscenterCorrectSmallRmc`"
		
		# Interpolate for diff
		Gint = BSplineKit.interpolate(
			RMpc, GMsunpc2,
			# At least 3 (qudratic) b/c we diff 2 times
			BSplineKit.BSplineOrder(max(3, interpolate.order)) 
		)
		dGint = BSplineKit.Derivative(1) * Gint
		ddGint = BSplineKit.Derivative(2) * Gint

		# The actual NLO correction formula
		# Correct up to order O((Rmc/R)^2) (but not up to O(κ (Rmc/R)^2)!)
		GMsunpc2 .- (1 ./ 4) .* (Rmc²Mpc ./ RMpc .^ 2) .* (
			.- 4 .* GMsunpc2
			.+ RMpc .* dGint.(RMpc)
			.+ (RMpc .^ 2) .* ddGint.(RMpc)
		)
	end

	function __calculate_miscenter_corrected_GMsunpc2_small_Rmc(
		interpolate::InterpolateLnR; RMpc, Rmc²Mpc, GMsunpc2
	)
		@assert length(RMpc) == length(GMsunpc2)
		@assert length(RMpc) >= 3 "Need at least 3 data points for `MiscenterCorrectSmallRmc`"
		
		# Interpolate for diff
		Gint = BSplineKit.interpolate(
			# NB: log = ln here. _Not_ log10!
			log.(RMpc), GMsunpc2,
			# At least 3 (qudratic) b/c we diff 2 times
			BSplineKit.BSplineOrder(max(3, interpolate.order)) 
		)
		ddGint = BSplineKit.Derivative(2) * Gint

		# The actual NLO correction formula
		# Correct up to order O((Rmc/R)^2) (but not up to O(κ (Rmc/R)^2)!)
		GMsunpc2 .- (1 ./ 4) .* (Rmc²Mpc ./ RMpc .^ 2) .* (
			.- 4 .* GMsunpc2
			# R df/dR + R^2 d^2 f/dR^2 = d^2 f / (d lnR)^2
			.+ ddGint.(log.(RMpc))
		)
	end

	function miscenter_correct_G(
		miscenter_correct::MiscenterCorrectSmallRmc, interpolate::I; R, G
	) where I<:AbstractInterpolate
		__calculate_miscenter_corrected_GMsunpc2_small_Rmc(
			interpolate;
			RMpc=R ./ u"Mpc" .|> NoUnits,
			Rmc²Mpc=miscenter_correct.Rmc² / u"Mpc^2" |> NoUnits,
			GMsunpc2=G ./ u"Msun/pc^2" .|> NoUnits
		) .* u"Msun/pc^2"
	end	
	
	function miscenter_correct_G_covariance(
		miscenter_correct::MiscenterCorrectSmallRmc, interpolate::I; R, G, G_covariance
	) where I<:AbstractInterpolate
		@assert length(R) == length(G) == size(G_covariance)[1] == size(G_covariance)[2]
		RMpc = R ./ u"Mpc" .|> NoUnits
		
		# For ForwardDiff, all inputs in one big vector etc.
		__calc_corrected_GMsunpc2(GMsunpc2_and_Rmc²Mpc2) = let

			# Unpack the vector
			Rmc²Mpc = GMsunpc2_and_Rmc²Mpc2[end]
			GMsunpc2 = GMsunpc2_and_Rmc²Mpc2[1:end-1]
	
			__calculate_miscenter_corrected_GMsunpc2_small_Rmc(
				interpolate; RMpc=RMpc, Rmc²Mpc=Rmc²Mpc, GMsunpc2=GMsunpc2
			)
		end

		# For ForwardDiff.jl all input in one big vector
		input = fill(NaN, length(G)+1)
		input[1:end-1] .= G ./ u"Msun/pc^2" .|> NoUnits
		input[end] = miscenter_correct.Rmc² / u"Mpc^2" |> NoUnits

		G_corrected = __calc_corrected_GMsunpc2(input) .* u"Msun/pc^2"
		jac = ForwardDiff.jacobian(__calc_corrected_GMsunpc2, input)
		G_cov_corrected = let
			input_cov = zeros(length(G)+1, length(G)+1)
			input_cov[1:end-1, 1:end-1] .= G_covariance ./ u"(Msun/pc^2)^2"
			input_cov[end, end] = (miscenter_correct.σ_Rmc² ^2) ./ u"(Mpc^2)^2"
			# See here https://juliadiff.org/ForwardDiff.jl/stable/user/api/
			# jac[α, i] = ∂ function(α)/∂x[i]
			# So the correct thing is jac * Cov * transpose(jac)
			# (and _not_ transpose(jac) * Cov * jac)
			out_cov = jac * input_cov * jac'
			# Units
			out_cov .* u"(Msun/pc^2)^2"
		end
		
		G_cov_corrected
	end
end

# ╔═╡ bb1aa65a-90b7-4d39-8f54-e1b306d506bb
md"""
# Actual deprojection
"""

# ╔═╡ ca33d61e-018e-4976-8c0b-0aba837a2af4
md"""
$I(R) \equiv \int_R^\infty dR' \frac{2}{R'} \frac{G f(R')}{1 - G f(R')}$

which satisfies the equation

$I'(R) = - \frac{2}{R} \frac{G f(R)}{1 - G f(R)}$

Boundary condition:

- Extrapolate $1/R^n$:

  $I(R_{\mathrm{max}}) = \int_{R_{\mathrm{max}}}^\infty dR' \frac{2}{R'} \frac{Gf(R_{\mathrm{max}}) \left(\frac{R_{\mathrm{max}}}{R}\right)^n}{1 - Gf(R_{\mathrm{max}}) \left(\frac{R_{\mathrm{max}}}{R}\right)^n} = - \frac2n \ln(1 - Gf(R_{\mathrm{max}}))$

  For later:

  $I(R \geq R_{\mathrm{max}}) = -\frac{2}{n} \ln\left(1 - Gf(R_{\mathrm{max}}) \left(\frac{R_{\mathrm{max}}}{R}\right)^n\right)$
"""

# ╔═╡ 3e5aa347-e19e-4107-a85e-30aa2515fb3a
begin
	function calculate_I_R∞(
		extrapolate::ExtrapolatePowerDecay, interpolate::InterpolateR; RMpc, Gf
	)
		# Solve in terms of X = Rmax - R so we can impose I(RMax) as an *initial*
		# condition (rather than a final condition). Because that's what `ODEProblem`
		# wants
		n = extrapolate.n
		RMpcMax = maximum(RMpc)
		RMpcMin = minimum(RMpc)
		prob = ODEProblem(
			(I, p, X) -> let
				RMpc = RMpcMax - X
				SA[(2/RMpc)*Gf(RMpc)/(1 - Gf(RMpc))] # RHS of I'(X) = ...
			end,
			SA[-(2/n)*log(1 - Gf(RMpcMax))], # Initial condition
			(0, RMpcMax-RMpcMin) # R interval where to solve
		)
		s = solve(prob, Tsit5())
		RMpc -> let
			@assert RMpc <= RMpcMax "I(R) only calculated up to last bin center!"
			s(RMpcMax - RMpc, idxs=1)
		end
	end
	
	function calculate_I_R∞(
		extrapolate::ExtrapolatePowerDecay, interpolate::InterpolateLnR; RMpc, Gf
	)
		# Solve in terms of X=ln(Rmax)-ln(R) so we can impose I(RMax) as an *initial*
		# condition (rather than a final condition). Because that's what `ODEProblem`
		# wants
		n = extrapolate.n
		RMpcMax = maximum(RMpc)
		RMpcMin = minimum(RMpc)
		prob = ODEProblem(
			(I, p, X) -> let
				RMpc = RMpcMax*exp(-X) # same as exp(ln(Rmax) - X)
				SA[2*Gf(RMpc)/(1 - Gf(RMpc))] # RHS of I'(X) = ... NB: no 1/RMpc
			end,
			SA[-(2/n)*log(1 - Gf(RMpcMax))], # Initial condition
			(0, log(RMpcMax)-log(RMpcMin)) # R interval where to solve
		)
		s = solve(prob, Tsit5())
		RMpc -> let
			@assert RMpc <= RMpcMax "I(R) only calculated up to last bin center!"
			s(log(RMpcMax/RMpc), idxs=1)
		end
	end
end

# ╔═╡ c8046b24-dfe7-4bf2-8787-b33d855e586f
md"""
$J(R) = \int_R^\infty dR'' \frac{2}{R''} \frac{G(R'')}{1 - Gf(R'')} \frac{1}{e^{-I(R'')}}$

which satisfies the diff eq

$J'(R) = - \frac{2}{R} \frac{G(R)}{1 - Gf(R)} \frac{1}{e^{-I(R)}}$

with boundary condition

$J(R_{\mathrm{max}}) = \int_{R_{\mathrm{max}}}^\infty dR'' \frac{2}{R''} \frac{G(R'')}{1 - Gf(R'')} \frac{1}{e^{-I(R'')}}$

For ..
- extrapolate $1/R^n$ (see Mathematica):

  $J(R_{\mathrm{max}}) = \frac{1}{f_\infty} \left((1 - Gf(R_{\mathrm{max}}))^{-\frac2n} - 1\right)$
"""

# ╔═╡ 64e5f173-11be-4dbf-b9ab-f652c50d9c09
begin
	function calculate_J_R∞(
		extrapolate::ExtrapolatePowerDecay, interpolate::InterpolateR;
		RMpc::typeof([1.0]), Gf, Ĝ, I
	)
		# Solve in terms of X = Rmax - R so we can impose J(RMax) as an *initial*
		# condition (rather than a final condition). Because that's what `ODEProblem`
		# wants
		n = extrapolate.n
		RMpcMax = maximum(RMpc)
		RMpcMin = minimum(RMpc)
		f̂∞ = Gf(RMpcMax)/Ĝ(RMpcMax)
		prob = ODEProblem(
			(J, p, X) -> let
				RMpc = RMpcMax - X
				# RHS of I'(X) = ...
				SA[(2/RMpc)*(1/exp(-I(RMpc)))*Ĝ(RMpc)/(1 - Gf(RMpc))]
			end,
			# Initial condition
			SA[(1/f̂∞)*((1 - Gf(RMpcMax))^(-2/n) - 1)], 
			(0, RMpcMax-RMpcMin) # R interval where to solve
		)
		s = solve(prob, Tsit5())
		RMpc -> let
			@assert RMpc <= RMpcMax "J(R) only calculated up to last bin center!"
			s(RMpcMax - RMpc, idxs=1)
		end
	end
	function calculate_J_R∞(
		extrapolate::ExtrapolatePowerDecay, interpolate::InterpolateLnR;
		RMpc::typeof([1.0]), Gf, Ĝ, I
	)
		# Solve in terms of X=ln(Rmax)-ln(R) so we can impose J(RMax) as an *initial*
		# condition (rather than a final condition). Because that's what `ODEProblem`
		# wants
		n = extrapolate.n
		RMpcMax = maximum(RMpc)
		RMpcMin = minimum(RMpc)
		f̂∞ = Gf(RMpcMax)/Ĝ(RMpcMax)
		prob = ODEProblem(
			(J, p, X) -> let
				RMpc = RMpcMax*exp(-X) # same as exp(ln(Rmax) - X)
				# RHS of I'(X) = ... NB: no 1/RMpc
				SA[2*(1/exp(-I(RMpc)))*Ĝ(RMpc)/(1 - Gf(RMpc))]
			end,
			# Initial condition
			SA[(1/f̂∞)*((1 - Gf(RMpcMax))^(-2/n) - 1)], 
			(0, log(RMpcMax)-log(RMpcMin)) # R interval where to solve
		)
		s = solve(prob, Tsit5())
		RMpc -> let
			@assert RMpc <= RMpcMax "J(R) only calculated up to last bin center!"
			s(log(RMpcMax) - log(RMpc), idxs=1)
		end
	end
end

# ╔═╡ 49397343-2023-4627-89e6-74170976c890
begin
	function get_interpolation_RMpc(
		extrapolate::ExtrapolatePowerDecay, interpolate::InterpolateR;
		RMpc::typeof([1.0]), values
	)
		func = BSplineKit.extrapolate(
			BSplineKit.interpolate(
				RMpc, values,
				# Linear interpolation = BSplineOrder(2), so +1
				BSplineKit.BSplineOrder(interpolate.order+1)
			),
			# Just b/c of floating point inexactness. We don't actually need values 
			# outside where they're defined
			BSplineKit.Flat()
		)

		maxRMpc = maximum(RMpc)
		boundaryVal = func(maxRMpc)
		n = extrapolate.n
		RMpc -> if RMpc > maxRMpc
			# Asymptotically we fall off as 1/R^n
			boundaryVal * (maxRMpc/RMpc)^n
		else
			func(RMpc)
		end
	end
	function get_interpolation_RMpc(
		extrapolate::ExtrapolatePowerDecay, interpolate::InterpolateLnR;
		RMpc::typeof([1.0]), values
	)
		logfunc = BSplineKit.extrapolate(
			BSplineKit.interpolate(
				log.(RMpc), values,
				# Linear interpolation = BSplineOrder(2), so +1
				BSplineKit.BSplineOrder(interpolate.order+1)
			),
			# Just b/c of floating point inexactness. We don't actually need values 
			# outside where they're defined
			BSplineKit.Flat()
		)

		maxRMpc = maximum(RMpc)
		boundaryVal = logfunc(log(maxRMpc))
		n = extrapolate.n
		RMpc -> if RMpc > maxRMpc
			# Asymptotically we fall off as 1/R^n
			boundaryVal * (maxRMpc/RMpc)^n
		else
			logfunc(log(RMpc))
		end
	end
	function get_interpolation_RMpc_flat(
		interpolate::InterpolateR;
		RMpc::typeof([1.0]), values
	)
		BSplineKit.extrapolate(
			BSplineKit.interpolate(
				RMpc, values,
				# Linear interpolation = BSplineOrder(2), so +1
				BSplineKit.BSplineOrder(interpolate.order+1)
			),
			BSplineKit.Flat()
		)
	end
	function get_interpolation_RMpc_flat(
		interpolate::InterpolateLnR;
		RMpc::typeof([1.0]), values
	)
		logfunc = BSplineKit.extrapolate(
			BSplineKit.interpolate(
				log.(RMpc), values,
				# Linear interpolation = BSplineOrder(2), so +1
				BSplineKit.BSplineOrder(interpolate.order+1)
			),
			BSplineKit.Flat()
		)
		RMpc -> logfunc(log(RMpc))
	end
end

# ╔═╡ dfe40541-396b-485b-bcb6-d70730a24867
md"""
The $\Delta \Sigma$ tail beyond $R = R_{\mathrm{max}}$is the *same* for both $f = \mathrm{\mathrm{const}}$ and $f \neq \mathrm{const}$.

That's because we assume $f = f(R_{\mathrm{max}})$ at $R > R_{\mathrm{max}}$ even when $f$ is not constant.

So we can use the simpler $f = \mathrm{const}$ formulas to calculate the $g_{\mathrm{obs}}$ tail.

- First, consider $1/R^n$ extrapolation. This gives

  $\begin{aligned}
  \Delta \Sigma(R > R_{\mathrm{max}})
   &= \frac{1}{f_\infty} \frac{G f(R)}{1 - Gf (R)} e^{-I(R)}\\
   &\stackrel{Mathematica}{=} \frac{1}{f_\infty} (G f(R_{\mathrm{max}})) \left(\frac{R_{\mathrm{max}}}{R}\right)^n \left(1 - G f(R_{\mathrm{max}}) \left(\frac{R_{\mathrm{max}}}{R}\right)^n \right)^{\frac2n - 1}
  \end{aligned}$

  We then have for $n=1$

  $\begin{aligned}
  &g_{\mathrm{obs}}^{\mathrm{tail}}(R) \\
  &= \int_0^{\theta_{m}} d\theta \Delta \Sigma(R/\sin \theta)\\
  &= \frac{4 G_N}{f_\infty} G f(R_{\mathrm{max}}) \frac{R_{\mathrm{max}}}{R} \bigg(
  \\
  & \quad \quad 1 - \frac12 G f(R_{\mathrm{max}}) \frac{R_{\mathrm{max}}}{R} \theta_{m} - \cos(\theta_{m}) + \frac12 G f(R_{\mathrm{max}}) \frac{R_{\mathrm{max}}}{R} \cos(\theta_{m}) \sin(\theta_{m}) \\
  &\quad \bigg)
  \end{aligned}$

  For $n=2$:

  $\begin{aligned}
  &g_{\mathrm{obs}}^{\mathrm{tail}}(R) \\
  &= \int_0^{\theta_{m}} d\theta \Delta \Sigma(R/\sin \theta)\\
  &= \frac{4 G_N}{f_\infty} \frac12  G f(R_{\mathrm{max}}) \left(\frac{R_{\mathrm{max}}}{R}\right)^2 \left(
   \theta_m - \cos(\theta_{m}) \sin(\theta_{m}) 
  \right)
  \end{aligned}$

  For other $n$: Just do it numerically, works quite well (see comments in code below) :)
"""

# ╔═╡ c86ab391-86c3-44f8-b0b9-20fb70c4dc87
function calculate_gobs_tail(
	extrapolate::ExtrapolatePowerDecay; θlim, f∞, GfTail, RMpc, RMpcTail
)
	n = extrapolate.n
	RMpcMax = RMpcTail
	Gfmax = GfTail

	if n == 1
		(4*u"G"/f∞)*Gfmax*(RMpcMax/RMpc)*(
			1
			- (1/2)*Gfmax*(RMpcMax/RMpc)*θlim
			- cos(θlim)
			+ (1/2)*Gfmax*(RMpcMax/RMpc)*cos(θlim)*sin(θlim)
		) |> u"m/s^2"
	elseif n == 2
		(4*u"G"/f∞)*(1/2)*Gfmax*(RMpcMax/RMpc)^2*(
			θlim - cos(θlim)*sin(θlim)
		) |> u"m/s^2"
	else
		# We could use that integral for the other n as well. It works well.
		# But I've implemented them already and they're faster of course, so
		# let's keep them for now.
		ΔΣtail(RMpc) = (1/f∞)*Gfmax*(RMpcMax/RMpc)^n*(1 - Gfmax*(RMpcMax/RMpc)^n)^(2/n-1)

		quadgk_result = 4*u"G*Msun/pc^2"*quadgk(
			θ -> ΔΣtail(RMpc/sin(θ))/u"Msun/pc^2" |> NoUnits,
			0, θlim
		)[1] |> u"m/s^2"
	end
end

# ╔═╡ c449a9c8-1739-481f-87d5-982532c2955c
function calculate_gobs_from_ΔΣ(
	extrapolate::E; ΔΣ, RMpc, f∞, Gf
) where E<:AbstractExtrapolate
	# The tail needs to be calculated analytically. Reason: The tail goes to R -> ∞.
	# That's ok for Gf(R) and f(R) because those we just extrapolate. But it's not ok
	# for I(R) and J(R) which also enter ΔΣ(R), because those we solved numerically
	# only up to R=Rmax (and the `ODESolution` extrapolation beyond last data point
	# is often completely off).

	RMpcTail = maximum(RMpc)
	GfTail = Gf(RMpcTail)
	
	gobs(RMpc) = if RMpc < RMpcTail
		θlim = asin(RMpc/RMpcTail)
		numeric = 4*u"G*Msun/pc^2"*quadgk(
			θ -> ΔΣ(RMpc/sin(θ))/u"Msun/pc^2" |> NoUnits,
			θlim, π/2
		)[1] |> u"m/s^2"
		analytical_tail = calculate_gobs_tail(
			extrapolate;
			θlim=θlim, f∞=f∞, GfTail=GfTail, RMpc=RMpc, RMpcTail=RMpcTail
		)
		numeric + analytical_tail
	else
		calculate_gobs_tail(
			extrapolate;
			θlim=π/2, f∞=f∞, GfTail=GfTail, RMpc=RMpc, RMpcTail=RMpcTail
		)
	end
end

# ╔═╡ 18dccd90-f99f-11ee-1bf6-f1ca60e4fcd0
begin
	# Non-constant f = <Σ_crit^(-1)>
	function calculate_gobs_fgeneral(;
		# Type omitted b/c of ForwardDiff which requires allowing Dual numbers
		G, # typeof([1.0*u"Msun/pc^2"]),
		f::typeof([1.0/u"Msun/pc^2"]),
		R::typeof([1.0*u"Mpc"]),
		interpolate::I,
		extrapolate::E,
		miscenter_correct::MC=MiscenterCorrectNone(),
	) where {
		E<:AbstractExtrapolate,
		I<:AbstractInterpolate,
		MC<:AbstractMiscenterCorrect
	}
		@assert length(G) == length(f) == length(R) "G, f, R must have same length"
		
		# We do miscenter correction before running the actual deprojection, at the
		# level of the input G
		G = miscenter_correct_G(miscenter_correct, interpolate; R=R, G=G)
		
		@assert !any(G .* f .>= 1.0) "G*f must be < 1"
	
		RMpc = R ./ u"Mpc"
		Gf_unchecked = get_interpolation_RMpc(
			extrapolate, interpolate;
			RMpc=RMpc, values=G .* f
		)
		Ĝ = get_interpolation_RMpc(
			extrapolate, interpolate;
			RMpc=RMpc, values=G ./ u"Msun/pc^2"
		)
		f̂ = get_interpolation_RMpc_flat(
			interpolate;
			RMpc=RMpc, values=f .* u"Msun/pc^2"
		)
		# Make sure that the interpolated Gf is not larger than 1.
		# Can happen with e.g. quadratic interpolation despite us having checked that
		# G .* f .< 1.0 holds.
		Gf = if interpolate.order > 1
			RMpc -> let
				value = Gf_unchecked(RMpc)
				@assert value < 1.0 "interpolated G*f must be < 1. Problem are typically too large fluctuations."
				value
			end
		else
			Gf_unchecked
		end

		IR∞ = calculate_I_R∞(extrapolate, interpolate; RMpc=RMpc, Gf=Gf)
		JR∞ = calculate_J_R∞(extrapolate, interpolate; RMpc=RMpc, Gf=Gf, Ĝ=Ĝ, I=IR∞)

		ΔΣ(RMpc) = (u"Msun/pc^2")*(Ĝ(RMpc)/(1 - Gf(RMpc)))*(
			1 - exp(-IR∞(RMpc))*f̂(RMpc)*JR∞(RMpc)
		)

		calculate_gobs_from_ΔΣ(extrapolate; ΔΣ=ΔΣ, RMpc=RMpc, f∞=f[end], Gf=Gf)
	end
	
	# Constant f = <Σ_crit^(-1)>
	function calculate_gobs_fconst(;
		# Type omitted b/c of ForwardDiff which requires allowing Dual numbers
		G, # typeof([1.0*u"Msun/pc^2"])
		f::typeof(1.0/u"Msun/pc^2"),
		R::typeof([1.0*u"Mpc"]),
		interpolate::I,
		extrapolate::E,
		miscenter_correct::MC=MiscenterCorrectNone(),
	) where {
		E<:AbstractExtrapolate,
		I<:AbstractInterpolate,
		MC<:AbstractMiscenterCorrect
	}
		@assert length(G) == length(R) "G, R must have same length"
		
		# We do miscenter correction before running the actual deprojection, at the
		# level of the input G
		G = miscenter_correct_G(miscenter_correct, interpolate; R=R, G=G)
		
		@assert !any(G .* f .>= 1.0) "G*f must be < 1"
		
		RMpc = R ./ u"Mpc"
		Gf_unchecked = get_interpolation_RMpc(
			extrapolate, interpolate;
			RMpc=RMpc, values=G .* f
		)
		# Make sure that the interpolated Gf is not larger than 1.
		# Can happen with e.g. quadratic interpolation despite us having checked that
		# G .* f .< 1.0 holds.
		Gf = if interpolate.order > 1
			RMpc -> let
				value = Gf_unchecked(RMpc)
				@assert value < 1.0 "interpolated G*f must be < 1. Problem are typically too large fluctuations."
				value
			end
		else
			Gf_unchecked
		end

		IR∞ = calculate_I_R∞(extrapolate, interpolate; RMpc=RMpc, Gf=Gf)
		ΔΣ(RMpc) = (1/f)*(Gf(RMpc)/(1 - Gf(RMpc)))*exp(-IR∞(RMpc))

		calculate_gobs_from_ΔΣ(extrapolate; ΔΣ=ΔΣ, RMpc=RMpc, f∞=f, Gf=Gf)
	end
end

# ╔═╡ 2e3d91f1-6b0f-4f5e-9761-e6a359585653
function calculate_gobs_and_covariance_in_bins(;
		G::typeof([1.0*u"Msun/pc^2"]),
		f,
		R::typeof([1.0*u"Mpc"]),
		G_covariance::typeof([1.0 1.0] .* u"(Msun/pc^2)^2"),
		interpolate::I,
		extrapolate::E,
		miscenter_correct::MC=MiscenterCorrectNone(),
	) where {
		E<:AbstractExtrapolate,
		I<:AbstractInterpolate,
		MC<:AbstractMiscenterCorrect
	}

	# We do miscenter correction before running the actual deprojection, at the level
	# of the input G and its covariance
	(G, G_covariance) = (
		miscenter_correct_G(
			miscenter_correct, interpolate; R=R, G=G),
		miscenter_correct_G_covariance(
			miscenter_correct, interpolate; R=R, G=G, G_covariance=G_covariance
		),
	)

	RMpc = R ./ u"Mpc" .|> NoUnits

	__get_gobs(f::typeof([1.0/u"Msun/pc^2"]); G_no_units) = calculate_gobs_fgeneral(;
		G=G_no_units .* u"Msun/pc^2",
		f=f,
		R=R,
		interpolate=interpolate,
		extrapolate=extrapolate,
		miscenter_correct=MiscenterCorrectNone(), # already done above
	)
	__get_gobs(f::typeof(1.0/u"Msun/pc^2"); G_no_units) = calculate_gobs_fconst(;
		G=G_no_units .* u"Msun/pc^2",
		f=f,
		R=R,
		interpolate=interpolate,
		extrapolate=extrapolate,
		miscenter_correct=MiscenterCorrectNone(), # already done above
	)

	# Forward-diff
	# - requires a single argument as input
	# - no units as input or output
	gobs_func = G_no_units -> let
		gobs = __get_gobs(f; G_no_units=G_no_units)
		gobs.(RMpc) ./ u"m/s^2"
	end

	# We could just `DiffResults` to avoid calculating `value` ourselves. That is
	# also done during the jacobian calculation anyway.
	# But: I tried that and in some cases the `value` was then off. Only by <1% but
	# still. Don't like that it's off at all. So let's just call gobs_fun(...)
	# once ourselves and lose a little perf :)
	value = gobs_func(G ./ u"Msun/pc^2" .|> NoUnits)
	jac = ForwardDiff.jacobian(
		gobs_func,
		G ./ u"Msun/pc^2" .|> NoUnits,
	)

	# `gobs_func` doesn't have units. So we have to put them back ourselves.
	gobs = value .* u"m/s^2"
	# See here https://juliadiff.org/ForwardDiff.jl/stable/user/api/
	# jac[α, i] = ∂gobs(α)/∂x[i]
	# So the correct thing is jac * Cov * transpose(jac)
	# (and _not_ transpose(jac) * Cov * jac)
	gobs_stat_cov = jac * (G_covariance ./ u"(Msun/pc^2)^2") * jac' .* u"(m/s^2)^2"
	gobs_stat_err = sqrt.(diag(gobs_stat_cov))
		
	(gobs=gobs, gobs_stat_cov=gobs_stat_cov, gobs_stat_err=gobs_stat_err)
end

# ╔═╡ f4311bdf-db19-4886-93f2-51143e6845bc
md"""
# Tests
## f=const and f!=const versions agree

when f is constant
"""

# ╔═╡ 9dd1c7c4-a44c-4b5c-a810-b6b171ac2569
@plutoonly let
	# Test: For f = const, both methods should agree
	gobs1overR = calculate_gobs_fconst(
		R=[.2, .5, .7] .* u"Mpc",
		G=[.3, .2, .1] .* u"Msun/pc^2",
		f=.9 ./ u"Msun/pc^2",
		extrapolate=ExtrapolatePowerDecay(1),
		interpolate=InterpolateR(1),
	)
	gobs1overRinterpLnR = calculate_gobs_fconst(
		R=[.2, .5, .7] .* u"Mpc",
		G=[.3, .2, .1] .* u"Msun/pc^2",
		f=.9 ./ u"Msun/pc^2",
		extrapolate=ExtrapolatePowerDecay(1),
		interpolate=InterpolateLnR(1),
	)
	plot(RMpc -> gobs1overR(RMpc), .2, 1.3, label="Extrapolate 1/R, interpolateR(1)")
	plot!(RMpc -> gobs1overRinterpLnR(RMpc), .2, 1.3, label="Extrapolate 1/R, interpolateLnR(1)")

	gobs1overR = calculate_gobs_fgeneral(
		R=[.2, .5, .7] .* u"Mpc",
		G=[.3, .2, .1] .* u"Msun/pc^2",
		f=[.9, .9, .9] ./ u"Msun/pc^2",
		extrapolate=ExtrapolatePowerDecay(1),
		interpolate=InterpolateR(1),
	)
	gobs1overRinterpLnR = calculate_gobs_fgeneral(
		R=[.2, .5, .7] .* u"Mpc",
		G=[.3, .2, .1] .* u"Msun/pc^2",
		f=[.9, .9, .9] ./ u"Msun/pc^2",
		extrapolate=ExtrapolatePowerDecay(1),
		interpolate=InterpolateLnR(1),
	)
	plot!(RMpc -> gobs1overR(RMpc), .2, 1.3, label="(general) Extrapolate 1/R, interpolateR(1)", ls=:dash)
	plot!(RMpc -> gobs1overRinterpLnR(RMpc), .2, 1.3, label="(general) Extrapolate 1/R, interpolateLnR(1)", ls=:dash)
end

# ╔═╡ 2dbc3c0b-8050-448b-b836-aafc21a7f189
md"""
## Covariance matrix in $Gf \ll 1$ limit = what we had for $C_{\alpha i}$ thing

Because in this limit $\Delta \Sigma$ is just $G$ and $f$ drops out and we can apply our simpler calculation from galaxy-galaxy weak lensing (where $\kappa$ is negligible).
"""

# ╔═╡ 2754de10-f637-46a4-ae6c-5e897206233a
@plutoonly let
	R_bin_edges = [.0, .4, .6, .8, 1.0] .* u"Mpc"
	R = old_gobs_from_ΔΣ.to_bin_centers(R_bin_edges)
	G=[.3, .2, .1, .05] .* u"Msun/pc^2"
	G_covariance=let
		σ_G = [.03, .02, .01, .005] * u"Msun/pc^2"
		diagm(σ_G .^ 2)
	end

	function do_test(; f, extrapolate, extrapolate_old, allowed_difference_factor=1.0)
		@info "Testing with" nameof(typeof(f)) nameof(typeof(extrapolate))
		new = calculate_gobs_and_covariance_in_bins(
			R=R, f=f, G=G, G_covariance=G_covariance,
			interpolate=InterpolateR(1),
			extrapolate=extrapolate,
		)
	
		let
			# Compare gobs
			old_gobs = old_gobs_from_ΔΣ.calculate_gobs_fast(
				R=R,
				ΔΣ̂=G ./ u"Msun/pc^2",
				extrapolate=extrapolate_old,
				last_RMpc_bin_edge=1.0, # Doesn't matter for SIS extrapolation
			)
			max_difference = maximum(abs.((new.gobs .- old_gobs) ./ old_gobs))
			@info "Test gobs" new.gobs old_gobs max_difference
			@assert max_difference < 5e-8*allowed_difference_factor "old gobs != new gobs?!"
		end
	
		let
			# Compare gobs stat error
			old_gobs_staterr = old_gobs_from_ΔΣ.calculate_gobs_staterr_fast(
				R=R,
				σ²_ΔΣ̂=diag(G_covariance) ./ u"(Msun/pc^2)^2",
				extrapolate=extrapolate_old,
				last_RMpc_bin_edge=1.0, # Doesn't matter for SIS extrapolation
			)
			max_difference = maximum(abs.((new.gobs_stat_err .- old_gobs_staterr) ./ old_gobs_staterr))
			@info "Test gobs stat err" new.gobs_stat_err old_gobs_staterr max_difference
			@assert max_difference < 5e-8*allowed_difference_factor "old gobs stat err != new gobs stat err?!"
		end
	
		let
			# Compare gobs covariance matrix
			old_gobs_cov = fill(NaN, length(R), length(R))
			old_gobs_from_ΔΣ.calculate_gobs_covariance_fast(;
				# NOTE: Here `extrapolate` is that of the `gobs` for which we now shall
				#       calculate the error.
				σ²_ΔΣ̂_l=(diag(G_covariance) ./ u"(Msun/pc^2)^2")',
				w̄l_unnormalized=ones(length(R))',
				∑ₗ_w̄l_unnormalized=ones(length(R))',
				l_r_bin_edges=(R_bin_edges)',
				extrapolate=extrapolate_old,
				out=old_gobs_cov
			)
			old_gobs_cov = old_gobs_cov .* u"(m/s^2)^2"
			max_difference = maximum(abs.((new.gobs_stat_cov .- old_gobs_cov) ./ old_gobs_cov))
			@info "Test gobs cov" new.gobs_stat_cov old_gobs_cov max_difference
			@assert max_difference < 1e-6*allowed_difference_factor "old gobs stat cov != new gobs stat cov?!"	
		end
	end

	# Make this f very small so we're in the Gf << 1 limit
	do_test(
		f = .1e-6 .* [.9, 1.5, 1.9, .9] ./ u"Msun/pc^2",
		extrapolate=ExtrapolatePowerDecay(1),
		extrapolate_old=old_gobs_from_ΔΣ.ExtrapolateΔΣSIS(),
	)
	# Same test but with the code that assumes f=const
	do_test(
		f = .1e-6 .* .9 ./ u"Msun/pc^2",
		extrapolate=ExtrapolatePowerDecay(1),
		extrapolate_old=old_gobs_from_ΔΣ.ExtrapolateΔΣSIS(),
	)

	@info "all good :)"
end

# ╔═╡ 981960ac-5f53-4175-a93d-660285acc372
md"""
## Reconstruction: Looks good out to $\sim$ Mpc

This is for $\rho \sim 1/r^2(1 + r^2)$ essentially. So SIS core, then $1/r^4$ fall off (reason: that's what Mathematica could do analytically)

Asymptotic behavior of $\Delta \Sigma$ (and thus of $Gf$) is $1/R^2$ for this profile. So at large radii, our $1/R^2$ extrapolation should give a gobs close to the real gobs. And indeed it does!

We're asssuming 10% fake errors on $G$ measurements, just so we can test the error calculation a bit.
"""

# ╔═╡ da111d64-a4f5-4637-986c-3b26027c058b
@plutoonly function test_reconstruction_SIS_quartic_fall_off(;
		logRMpc_bin_width, interpolate=InterpolateR(1)
)
	# Test: Do we actually recontruct the correct gobs?

	# See `check-ESD-to-RAR-for-explicit-examples.nb` in `lensing-RAR/`for analytic 
	# formulas.
	#  SIS core + faster - 1/r^4 fall-off at larger  radii
	r0 = .01u"Mpc"
	ΔΣ_real(R) = 100u"Msun/pc^2"*r0 * (
		1/R + 1/sqrt(R^2+r0^2) + 2*(r0 - sqrt(r0^2+R^2))/R^2
	)
	Σ_real(R) = 100u"Msun/pc^2"*r0 * (
		1/R - 1/sqrt(R^2+r0^2)
	)
	gobs_real(R) = 4*u"G"*(100*r0*u"Msun/pc^2")*r0*atan(R/r0)/R^2 |> u"m/s^2"

	# Remove the exponential to test the constant f case
	f(R) = (1/(2u"Msun/pc^2"))*(1 + exp(-R/200u"kpc"))
	G(R) = ΔΣ_real(R)/(1 - f(R)*Σ_real(R))
	# Make up fake 10% measurement errors on G
	σ_G(R) = 0.1 * G(R)

	pGf = plot(R -> G(R)*f(R)*(R/u"Mpc")^2, .1u"Mpc", 3u"Mpc", ylabel="G * f * R^2")

	Rbins = 10 .^ (log10(.15):logRMpc_bin_width:log10(3.5)) .* u"Mpc"
	last_RMpc_bin_edge = 3.8
	p = plot(
		RMpc -> gobs_real(RMpc*u"Mpc") * RMpc^2,
		.15, 3.5, label="gobs real * R^2",
		xscale=:log10,
		xlabel="R in Mpc",
		ylabel="gobs * R^2",
	)
	p_gobs_ratio = plot(
		RMpc -> 1.0,
		.15, 3.5, label="",
		color=:black,
		xscale=:log10,
		xlabel="R in Mpc",
		ylabel="gobs reconstructed / gobs real",
	)
	for n in [1/2, 1, 2, 4]
		gobs_reconstructed = calculate_gobs_and_covariance_in_bins(
			R=Rbins,
			G=G.(Rbins),
			G_covariance=diagm(σ_G.(Rbins) .^ 2),
			f=f.(Rbins),
			extrapolate=ExtrapolatePowerDecay(n),
			interpolate=interpolate,
		)
		plot!(p,
			Rbins ./ u"Mpc",
			gobs_reconstructed.gobs .* (Rbins ./ u"Mpc") .^ 2,
			yerror=gobs_reconstructed.gobs_stat_err .* (Rbins ./ u"Mpc") .^ 2,
			label="gobs reconstructed * R^2, 1/R^$(n) extrapolate",
			marker=:diamond,
		)
		plot!(p_gobs_ratio,
			Rbins ./ u"Mpc",
			gobs_reconstructed.gobs ./ gobs_real.(Rbins),
			yerror=gobs_reconstructed.gobs_stat_err ./ gobs_real.(Rbins),
			label="1/R^$(n) extrapolate",
			marker=:diamond,
		)
	end

	plot(p, p_gobs_ratio, pGf, size=(600, 400*3), layout=(3,1), left_margin=(15, :mm))
end

# ╔═╡ 53eb9c24-1113-4a78-a006-6487e5d8f732
@plutoonly test_reconstruction_SIS_quartic_fall_off(logRMpc_bin_width=.05)

# ╔═╡ a97f453d-1794-4302-958b-d06b98a1a9cb
md"""
## Small bias if sampling too small (few percent)
"""

# ╔═╡ 3da7711f-7335-4498-891b-fe6ac1e81d7c
@plutoonly test_reconstruction_SIS_quartic_fall_off(logRMpc_bin_width=.1)

# ╔═╡ d06064c8-5bc9-4e29-ba21-62925fca0104
md"""
## ... better with quadratic interpolation or interpolation in $\ln(R)$ space
"""

# ╔═╡ f5b99cdf-daf7-4419-9259-57f07b3d9fdf
@plutoonly test_reconstruction_SIS_quartic_fall_off(logRMpc_bin_width=.1, interpolate=InterpolateR(2))

# ╔═╡ 11c0a007-08a4-446b-81ff-961ab62b9051
@plutoonly test_reconstruction_SIS_quartic_fall_off(logRMpc_bin_width=.1, interpolate=InterpolateLnR(1))

# ╔═╡ 9dcd6d67-90f6-4cd2-843c-02b3f6d196cd
md"""
## SIS reconstruction works also in tail

(as it must, that's what this test is for to check)
"""

# ╔═╡ cda4a385-3c68-430d-8e86-abd54374dffa
@plutoonly function test_reconstruction_SIS(; logRMpc_bin_width, interpolate)
	# Test: Do we actually recontruct the correct gobs?

	# See `check-ESD-to-RAR-for-explicit-examples.nb` in `lensing-RAR/`for analytic 
	# formulas.
	# Just SIS. Very simple. Reason to test this: Should be correct in the 1/R tail!
	r0 = .01u"Mpc"
	ΔΣ_real(R) = 100u"Msun/pc^2"*r0 * (
		1/R
	)
	Σ_real(R) = 100u"Msun/pc^2"*r0 * (
		1/R
	)
	gobs_real(R) = 4*u"G"*(100*r0*u"Msun/pc^2")/R |> u"m/s^2"

	# Remove the exponential to test the constant f case
	f(R) = (1/(50u"Msun/pc^2"))*(1 + exp(-R/200u"kpc"))
	G(R) = ΔΣ_real(R)/(1 - f(R)*Σ_real(R))

	pGf = plot(R -> G(R)*f(R)*(R/u"Mpc")^2, .1u"Mpc", 3u"Mpc", ylabel="G * f * R^2")

	Rbins = 10 .^ (log10(.15):logRMpc_bin_width:log10(3.5)) .* u"Mpc"
	last_RMpc_bin_edge = 3.8
	p = plot(
		RMpc -> gobs_real(RMpc*u"Mpc") * RMpc^2,
		.15, 3.5, label="gobs real * R^2 ",
		xscale=:log10,
		xlabel="R in Mpc",
		ylabel="gobs * R^2",
	)
	p_gobs_ratio = plot(
		RMpc -> 1.0,
		.15, 3.5, label="",
		color=:black,
		xscale=:log10,
		xlabel="R in Mpc",
		ylabel="gobs reconstructed / gobs real",
	)
	for n in [1/2, 1, 2, 4]
		gobs_reconstructed = calculate_gobs_fgeneral(
			R=Rbins,
			G=G.(Rbins),
			f=f.(Rbins),
			extrapolate=ExtrapolatePowerDecay(n),
			interpolate=interpolate
		)
		plot!(p,
			Rbins ./ u"Mpc",
			gobs_reconstructed.(Rbins ./ u"Mpc") .* (Rbins ./ u"Mpc") .^ 2,
			label="gobs reconstructed * R^2, 1/R^$(n) extrapolate",
			marker=:diamond,
		)
		plot!(p_gobs_ratio,
			Rbins ./ u"Mpc",
			gobs_reconstructed.(Rbins ./ u"Mpc")  ./ gobs_real.(Rbins),
			label="1/R^$(n) extrapolate",
			marker=:diamond,
		)
	end

	plot(p, p_gobs_ratio, pGf, size=(600, 400*3), layout=(3,1), left_margin=(15, :mm))
end

# ╔═╡ dd9afde0-ea99-41c5-8b86-da1c91a09fc4
@plutoonly test_reconstruction_SIS(logRMpc_bin_width=.12, interpolate=InterpolateR(2))

# ╔═╡ e3401c57-3fe6-4526-a128-387672b33863
md"""
## Miscentering correction
"""

# ╔═╡ bfd8b4e9-4b43-4720-bcc2-9263ac2d2362
@plutoonly let
	# Test: R and ln R interpolation agree with small bins

	R = (.4:.003:3.0) .* u"Mpc"
	Gfunc = R -> 300u"Msun/pc^2" / (R/u"Mpc") |> u"Msun/pc^2"
	G = Gfunc.(R)

	mc = MiscenterCorrectSmallRmc(1.0u"Mpc^2", 0.5u"Mpc^2")
	G_corr_R =  miscenter_correct_G(mc, InterpolateR(1), R=R, G=G)
	G_corr_lnR =  miscenter_correct_G(mc, InterpolateLnR(1), R=R, G=G)

	@assert all(abs.(G_corr_R ./ G_corr_lnR .- 1) .< .02) "R and ln R interpolations should agree well for small bins"

	G_cov = diagm( (.1 .* G) .^ 2)
	G_corr_cov_R = miscenter_correct_G_covariance(
		mc, InterpolateR(1);
		R=R, G=G, G_covariance=G_cov
	)
	G_corr_cov_lnR = miscenter_correct_G_covariance(
		mc, InterpolateLnR(1);
		R=R, G=G, G_covariance=G_cov
	)
	@assert all(abs.(G_corr_cov_R ./ G_corr_cov_lnR .- 1) .< .2) "R and ln R interpolations should agree well for small bins (covariance)"
end

# ╔═╡ c91c0b29-b152-4c49-b90d-5447b2908f0d
@plutoonly let
	# Test: Miscentering correction correctly corrects

	p_original = __demo.ProfileSIS(
		M1Mpc=4e14u"Msun",
		x0=0u"Mpc", # Centered
		y0=0u"Mpc",
	)
	p_miscentered = __demo.ProfileSIS(
		M1Mpc=4e14u"Msun",
		x0=.16u"Mpc", # Not centered
		y0=0u"Mpc",
	)
	Σcrit = 3000u"Msun/pc^2"

	R = collect(.4:.01:3.0) .* u"Mpc"

	calc_gobs = (p, miscenter_correct) -> let 
		gt = __demo.calculate_azimuthally_averaged_gt.(R, Ref(p); Σcritinv=1/Σcrit)
		f = 1/Σcrit
		G = gt*Σcrit
		res = calculate_gobs_and_covariance_in_bins(
			R=R, G=G, f=f,
			G_covariance=zeros(length(R), length(R)) .* u"(Msun/pc^2)^2",
			interpolate=InterpolateR(2),
			extrapolate=ExtrapolatePowerDecay(1), # SIS (not exact here!)
			miscenter_correct=miscenter_correct
		)

		(res.gobs, res.gobs_stat_err)
	end

	(gobs_centered, _) = calc_gobs(p_original, MiscenterCorrectNone())
	(gobs_uncorrected, _) = calc_gobs(p_miscentered, MiscenterCorrectNone())
	(gobs_corrected, gobs_uncorrected_stat_err) = calc_gobs(
		p_miscentered,
		MiscenterCorrectSmallRmc(
			# Correct by the actual Rmc
			Rmc²=(.16u"Mpc")^2,
			# For later: use (uncertainty of Rmc^2) = Rmc^2
			σ_Rmc²=(.16u"Mpc")^2
		)
	)
	gobs_true = (R -> (u"G"/R^2)*R*4e14u"Msun"/1u"Mpc" |> u"m/s^2").(R)

	# Assert some stuff
	# 1) reconstruction works for correctly centered profile (permill)
	@assert all(abs.(gobs_centered ./ gobs_true .- 1) .< .003)
	# 2a) reconstruction _doesn't_ work for miscentered profile at small radii (>5%)
	sel = R .< .5u"Mpc"
	@assert all(abs.(gobs_uncorrected[sel] ./ gobs_true[sel] .- 1) .> .05)
	# 2b) at large radii it slowly gets better (naturally) (<1.5% here)
	sel = R .> 1.0u"Mpc"
	@assert all(abs.(gobs_uncorrected[sel] ./ gobs_true[sel] .- 1) .< .015)
	# 3a) Miscentering correction helps at small radii! (btter than 1.1%)
	sel = R .< .5u"Mpc"
	@assert all(abs.(gobs_corrected[sel] ./ gobs_true[sel] .- 1) .< .011)
	# 3b) Miscentering correction also helps at large radii (now permill!)
	sel = R .> 1.0u"Mpc"
	@assert all(abs.(gobs_corrected[sel] ./ gobs_true[sel] .- 1) .< .003)

	# 4) Linearity in Rmc^2 means: Uncertainty in gobs induced by Rmc^2 = gobs[Rmc^2-Rmc^2] - gobs[Rmc^2=0] (most are actually much better than 9% -- but apparently some outlier is there?)
	@assert all(abs.(abs.(gobs_uncorrected .- gobs_corrected) ./gobs_uncorrected_stat_err .- 1) .< .09)

	# Plots that show this visually
	plot(
		R, gobs_centered ./ gobs_true,
		label="M reconst. original / M true", color=:black,
		ylim=(:auto, 1.05)
	)
	plot!(R,
		gobs_corrected ./ gobs_true,
		ribbon=gobs_uncorrected_stat_err ./ gobs_true,
		label="M reconst. miscentered, corrected / M true", color=3, marker=:diamond, ms=2,
	)
	plot!(
		R, gobs_uncorrected ./ gobs_true,
		label="M reconst. miscentered / M true", color=1, marker=:diamond, ms=2,
	)
end

# ╔═╡ Cell order:
# ╠═8f04c59b-a109-4032-9235-1acc6f8ad9b4
# ╠═4bdde00d-5d78-45e5-8c4e-a790f7431a3c
# ╠═d7ce151b-4732-48ea-a8a5-5bfbe94d119b
# ╠═2bd8f9c4-ed93-406f-974e-3539d44f21c4
# ╠═9269044a-217b-48ef-b6f1-266a75890956
# ╟─52cadcf0-a9ae-4e91-ac44-21e6fd25dabc
# ╟─6cc0e536-970a-4e94-8449-6c358c31b3ec
# ╠═61671b5e-9a09-49ed-ba69-37852662f803
# ╟─bb1aa65a-90b7-4d39-8f54-e1b306d506bb
# ╟─ca33d61e-018e-4976-8c0b-0aba837a2af4
# ╠═3e5aa347-e19e-4107-a85e-30aa2515fb3a
# ╟─c8046b24-dfe7-4bf2-8787-b33d855e586f
# ╠═64e5f173-11be-4dbf-b9ab-f652c50d9c09
# ╠═49397343-2023-4627-89e6-74170976c890
# ╟─dfe40541-396b-485b-bcb6-d70730a24867
# ╠═c86ab391-86c3-44f8-b0b9-20fb70c4dc87
# ╠═c449a9c8-1739-481f-87d5-982532c2955c
# ╠═18dccd90-f99f-11ee-1bf6-f1ca60e4fcd0
# ╠═2e3d91f1-6b0f-4f5e-9761-e6a359585653
# ╟─f4311bdf-db19-4886-93f2-51143e6845bc
# ╠═9dd1c7c4-a44c-4b5c-a810-b6b171ac2569
# ╟─2dbc3c0b-8050-448b-b836-aafc21a7f189
# ╠═2754de10-f637-46a4-ae6c-5e897206233a
# ╠═1ae70636-b3ce-4ac7-b827-e8ec615bde29
# ╟─981960ac-5f53-4175-a93d-660285acc372
# ╠═53eb9c24-1113-4a78-a006-6487e5d8f732
# ╠═da111d64-a4f5-4637-986c-3b26027c058b
# ╟─a97f453d-1794-4302-958b-d06b98a1a9cb
# ╠═3da7711f-7335-4498-891b-fe6ac1e81d7c
# ╟─d06064c8-5bc9-4e29-ba21-62925fca0104
# ╠═f5b99cdf-daf7-4419-9259-57f07b3d9fdf
# ╠═11c0a007-08a4-446b-81ff-961ab62b9051
# ╟─9dcd6d67-90f6-4cd2-843c-02b3f6d196cd
# ╠═dd9afde0-ea99-41c5-8b86-da1c91a09fc4
# ╠═cda4a385-3c68-430d-8e86-abd54374dffa
# ╟─e3401c57-3fe6-4526-a128-387672b33863
# ╠═bfd8b4e9-4b43-4720-bcc2-9263ac2d2362
# ╠═c91c0b29-b152-4c49-b90d-5447b2908f0d
# ╟─3f004698-b952-462f-8824-5c78ab1e08ad
