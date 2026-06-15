### A Pluto.jl notebook ###
# v1.0.1

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
	import Roots
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
	import Cosmology # For NFW tests
	using Plots
	import PlutoUI
	PlutoUI.TableOfContents()
end

# ╔═╡ 9269044a-217b-48ef-b6f1-266a75890956
begin
	# Mass-concentration relations for NFW extrapolation
	abstract type AbstractMassConcentrationRelation end
	struct CMRelationMaccio2008 <: AbstractMassConcentrationRelation
		ρcrit::typeof(1.0u"Msun/Mpc^3")
		h::Float64
	end
	
	# How to extrapolate G beyond last data point
	abstract type AbstractExtrapolate end
	struct ExtrapolatePowerDecay <: AbstractExtrapolate
		# Only positive values are allowed!
		n::Float64
	end
	struct ExtrapolateNFW{
		CM<:AbstractMassConcentrationRelation
	} <: AbstractExtrapolate
		cm::CM
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
	copy_with_other_Rmc²(::MiscenterCorrectNone, Rmc²) = MiscenterCorrectNone()
	
	# Correct up to (including) O((Rmc/R)^2)
	# (but not including O(κ (Rmc/R)^2))
	# This applies the correction when calculating M from ΔΣ.
	# Advantage: Numerically stable with noisy data. Disadvantage: Slightly slower
	struct MiscenterCorrectSmallRmc{R} <: AbstractMiscenterCorrect 
		Rmc²::R # Not typeof(1.0u"Mpc^2") b/c of ForwardDiff
		# Note: That's the uncertainty on (Rmc^2)
		σ_Rmc²::typeof(1.0u"Mpc^2")
	end
	MiscenterCorrectSmallRmc(; Rmc²::typeof(1.0u"Mpc^2"), σ_Rmc²) = MiscenterCorrectSmallRmc(
		Rmc²,
		σ_Rmc²
	)
	copy_with_other_Rmc²(mc::MiscenterCorrectSmallRmc, Rmc²) = MiscenterCorrectSmallRmc(
		Rmc²,
		mc.σ_Rmc²
	)
	
	# Correct up to (including) O((Rmc/R)^2)
	# (but not including O(κ (Rmc/R)^2))
	# This applies the correction as a preprocessing step on G.
	# Advantage: Very fast. Disadvantage: Requires up to 2nd order numerical
	# derivatives of G_+.
	struct MiscenterCorrectSmallRmcPreprocessG{R} <: AbstractMiscenterCorrect
		Rmc²::R # Not typeof(1.0u"Mpc^2") b/c of ForwardDiff
		# Note: That's the uncertainty on (Rmc^2)
		σ_Rmc²::typeof(1.0u"Mpc^2")
	end
	MiscenterCorrectSmallRmcPreprocessG(; Rmc²::typeof(1.0u"Mpc^2"), σ_Rmc²) = MiscenterCorrectSmallRmcPreprocessG(
		Rmc²,
		σ_Rmc²,
	)
	copy_with_other_Rmc²(mc::MiscenterCorrectSmallRmcPreprocessG, Rmc²) = MiscenterCorrectSmallRmcPreprocessG(
		Rmc²,
		mc.σ_Rmc²
	)
end

# ╔═╡ 1ae70636-b3ce-4ac7-b827-e8ec615bde29
module old_gobs_from_ΔΣ
	using Unitful
	using UnitfulAstro
	using Dierckx
	import HypergeometricFunctions
	import ..MiscenterCorrectNone, ..MiscenterCorrectSmallRmc
	import ..ExtrapolatePowerDecay

	function to_bin_centers(edges)
		widths = circshift(edges, -1) .- edges
		( edges .+ (widths./2) )[1:end-1]
	end

	gobs_analytical_tail_factor(ex::ExtrapolatePowerDecay; RoverRmax, RMpcMax) = let
		x = RoverRmax
		((1/x)^ex.n)*(if ex.n == 1
			1 - sqrt(1-x^2)
		elseif ex.n == 2
			.5*(-x*sqrt(1-x^2) + asin(x))
		else
			hyp = HypergeometricFunctions.:_₂F₁(1/2, (1+ex.n)/2, (3+ex.n)/2, x^2)
			x^(1+ex.n)*hyp/ (1+ex.n)
		end)			
	end

	function fast_gobs_∑_i_Cαi²_σ²_ΔΣ̂(;
		Rmc²::typeof(1.0u"Mpc^2"),
		R::typeof([1.0u"Mpc"]),
		σ²_ΔΣ̂,
		extrapolate,
	)
		N = length(R)
		out = fill(NaN, N)
		C = zeros(N) # Don't allocate in loop
		for α in eachindex(out)
			fast_gobs_Cαi(; Rmc², R, C, α, extrapolate)
			out[α] = sum(C[i]^2*σ²_ΔΣ̂[i] for i in α:N)
		end
		out
	end
	function fast_gobs_Mpc²_∑_i_∂Cαi∂Rmc²_ΔΣ̂(;
		R::typeof([1.0u"Mpc"]),
		ΔΣ̂,
		extrapolate,
	)
		dummy_Rmc² = 1.0u"Mpc^2"
		zero_Rmc² = 0.0u"Mpc^2"
	
		Mpc = 1.0u"Mpc"
		
		N = length(R)
		out = fill(NaN, N)
		C_dummy = zeros(N) # Don't allocate in loop
		C_zero = zeros(N) # Don't allocate in loop
		for α in eachindex(out)
			# We know that Cαi = linear in Rmc^2.
			# So:
			#    Mpc^2 * (∂ Cαi / ∂ Rmc²)
			#  = (Cαi|_(Rmc² = dummy^2) - Cαi|_(Rmc²=0)) / ( (dummy/Mpc)^2)
			fast_gobs_Cαi(; Rmc²=dummy_Rmc², R, C=C_dummy, α, extrapolate)
			fast_gobs_Cαi(; Rmc²=zero_Rmc², R, C=C_zero, α, extrapolate)
			out[α] = sum(
				((C_dummy[i] - C_zero[i])/(dummy_Rmc²/Mpc^2))*ΔΣ̂[i]
				for i in α:N
			)
		end
		out
	end
	function fast_gobs_Cαi(;
		Rmc²::typeof(1.0u"Mpc^2"),
		R::typeof([1.0u"Mpc"]), C, α::Int64,
		extrapolate::ExtrapolatePowerDecay,
	)
	
		# θ_(α i) = asin(R_α/R_i)
		cosθ(α, i) = sqrt(1 - (R[α]/R[i])^2)
		sinθ(α, i) = R[α]/R[i]
		tanθ(α, i) = sinθ(α, i) / cosθ(α, i)

		# θlim_α = asin(R_α/R_N)
		N = length(R)
		cosθlim(α) = sqrt(1 - (R[α]/R[N])^2)
		sinθlim(α) = R[α]/R[N]
		tanθlim(α) = sinθlim(α) / cosθlim(α)
	
		# Integrals ∫_lower^upper go from lower = θ_(α,i+1) up to upper = θ_(α, i)
		# NOTE: the order is i+1 -> i b/c R/θ grow in opposite directions!
		# 
		# Zap diverging terms using `zero_at_αα`
		zero_at_αα(α, i) = val -> α == i  ? 0 : val
		Δθ_αi(α, i) = asin(R[α]/R[i]) - asin(R[α]/R[i+1])
		a_αi(α, i) = -atanh(cosθ(α, i)) - (-atanh(cosθ(α, i+1)))
		b_αi(α, i) = 2Δθ_αi(α, i) + (
			(- cosθ(α, i)  *sinθ(α, i)   - (tanθ(α, i)   |> zero_at_αα(α, i))) -
			(- cosθ(α, i+1)*sinθ(α, i+1) -  tanθ(α, i+1)                   )
		)
		c_minus_dαi(α, i) = (
			(-cosθ(α, i)   - (2/cosθ(α, i)  |> zero_at_αα(α, i))) -
			(-cosθ(α, i+1) -  2/cosθ(α, i+1)                  )
		)

		A_αi(α, i) = (
			Δθ_αi(α, i) + (1/4)*(Rmc²/R[α]^2) * b_αi(α, i)
			+ if α == i
				(1/4)*(Rmc²/R[α]^2) * (
					(-cosθ(α, α)*sinθ(α, α) + (tanθ(α, α) |> zero_at_αα(α, α))) - 
					(-cosθ(α, N)*sinθ(α, N) +  tanθ(α, N))
				)
			else
				0.0
			end
		)
		B_αi(α, i) = (
			- Δθ_αi(α, i)*R[i]
			+ a_αi(α, i)*R[α]
			- (1/4)*(Rmc²/R[α]^2) * b_αi(α, i) * R[i]
			+ (1/4)*(Rmc²/R[α]^2) * c_minus_dαi(α, i) * R[α]
			+ if α == i
				(1/4)*(Rmc²/R[α]^2) * R[α] * (
					# From "bulk"
					(sinθ(α, α) * (tanθ(α, α) |> zero_at_αα(α, α))) -
					(sinθ(α, N) *  tanθ(α, N))
					# From "boundary"
					+sinθlim(α)*tanθlim(α)
				)
			else
				0.0 * u"Mpc"
			end
			+ if N-1 == i
				# From "boundary"
				(1/4)*(Rmc²/R[α]^2) * R[α] * (
					-sinθlim(α)*tanθlim(α)
				)
			else
				0.0 * u"Mpc"
			end
		) / (R[i+1] - R[i])
	
		# The (1+ (1/4) ...) corrects the last data point's ΔΣ using the "naive"
		# miscnetering correction formula that needs 2nd derivatives. It's ok b/c
		# we can do the calculation analytically for the SIS tail.
		f_cont(α) = gobs_analytical_tail_factor(
			extrapolate;
			RoverRmax=R[α]/R[end],
			RMpcMax=R[end] / u"Mpc" |> NoUnits,
		)*(1 + (1/4)*(Rmc²/R[end]^2)*(4-extrapolate.n^2))
		
		@assert length(C) == length(R)
		
		# Leave the C[begin:α-1] part of C untouched for perf (don't zero out or so)
		# (Same formulas as in `MiscenterCorrectNone` case, just with
		#  Δθ -> A, f -> B)
		if α < length(C)
			C[α] = A_αi(α, α) - B_αi(α, α)
		end
		for i in α+1:length(C)-1
			C[i] = A_αi(α, i) - B_αi(α, i) + B_αi(α, i-1)
		end
		if α == length(C)
			C[length(C)] = f_cont(α)
		else
			C[length(C)] = f_cont(α) + B_αi(α, length(C)-1)
		end

		# "Boundary" terms proportional to ΔΣ_α and ΔΣ_N.
		# One may be tempted to absorb them into A_αi. But that doesn't go up to
		# i=N so cannot be used.
		# For α = N they cancel.
		if α < N
			C[α] += (1/4)*(Rmc²/R[α]^2)*sinθlim(α)^2*tanθlim(α)
			C[N] -= (1/4)*(Rmc²/R[α]^2)*sinθlim(α)^2*tanθlim(α)
		end
		
		nothing
	end
	function calculate_gobs_staterr_fast(;
		w̄l_unnormalized::AbstractMatrix,
	 	∑ₗ_w̄l_unnormalized::AbstractMatrix,
		∑_i_Cαi²_σ²_ΔΣ̂_l::AbstractMatrix,
		Mpc²_∑_i_∂Cαi∂Rmc²_ΔΣ̂_l::AbstractMatrix,
		σ_Rmc²::typeof(1.0u"Mpc^2")
	)
		term1 = sum((w̄l_unnormalized .^2) .* ∑_i_Cαi²_σ²_ΔΣ̂_l, dims=1)
		term2 = sum(w̄l_unnormalized .* Mpc²_∑_i_∂Cαi∂Rmc²_ΔΣ̂_l, dims=1) .^ 2
	
		FourG² = (4u"G*Msun/pc^2")^2 ./ u"(m/s^2)^2" |> NoUnits
		num = sqrt.(FourG² .* (term1 .+ (σ_Rmc²/u"Mpc^2")^2 .* term2))
		num ./ ∑ₗ_w̄l_unnormalized
	end
	function calculate_gobs_covariance_fast(;
			Rmc²::typeof(1.0u"Mpc^2"), σ_Rmc²::typeof(1.0u"Mpc^2"),
			Mpc²_∑_i_∂Cαi∂Rmc²_ΔΣ̂_l::AbstractMatrix,
			extrapolate::ExtrapolatePowerDecay,
			σ²_ΔΣ̂_l::AbstractMatrix,
			w̄l_unnormalized::AbstractMatrix,
			∑ₗ_w̄l_unnormalized::AbstractMatrix,
			l_r_bin_edges::AbstractMatrix,
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
			
			term1 = 0.0
			term2_α = 0.0
			term2_β = 0.0
			for l in 1:l_len
				# CARE: Do these two Rmc² related terms _before_ skipping below!
				#       The condition (w̄[l, α] == 0 || w̄[l, β] == 0) is ok to skip for
				#       the rest b/c that's proportional to the *product* of w̄[l, α] and
				#       w̄[l, β]. But these two are sensitive to both _individually_. 
				term2_α += (
					w̄l_unnormalized[l, α]*
					Mpc²_∑_i_∂Cαi∂Rmc²_ΔΣ̂_l[l, α]
				)
				term2_β += (
					w̄l_unnormalized[l, β]*
					Mpc²_∑_i_∂Cαi∂Rmc²_ΔΣ̂_l[l, β]
				)
				
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
				fast_gobs_Cαi(;
					R, C=(@view Cα[idx]), Rmc²,
					# The `α` named parameter must now denote the position of the original
					# radial bin in the reduced set of bins R.
					α=α-num_missing_before_α,
					extrapolate,
				)
				fast_gobs_Cαi(;
					R, C=(@view Cβ[idx]), Rmc²,
					# The `α` named parameter must now denote the position of the original
					# radial bin in the reduced set of bins R.
					α=β-num_missing_before_β,
					extrapolate
				)
				
				term1 += (
					w̄l_unnormalized[l, α]*w̄l_unnormalized[l, β]*
					sum(Cα[i]*Cβ[i]*σ²_ΔΣ̂_l[l, i] for i in max(α, β):length(Cα))
				)
			end
			
			prefactor*(
				term1 +
				(σ_Rmc²/u"Mpc^2")^2 * term2_α*term2_β
			) / (∑ₗ_w̄l_unnormalized[α] * ∑ₗ_w̄l_unnormalized[β])
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
	function calculate_gobs_fast(; Rmc²::typeof(1.0u"Mpc^2"), R, ΔΣ̂, extrapolate)
		prefactor = 4*u"G"*u"Msun/pc^2" |> u"m/s^2"
		gobs = fill(NaN*u"m/s^2", length(R))
		C = zeros(length(R)) # Don't allocate in loop
		for α in eachindex(gobs)
			fast_gobs_Cαi(; Rmc², R, C, α, extrapolate)
			gobs[α] = prefactor * sum(C[i]*ΔΣ̂[i] for i in α:length(C))
		end
		gobs
	end
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
# Miscentering correction at level of $G$

(*Problem: This requires numerical derivatives up to 2nd order which can be dicy in practice*)



For a given miscentering radius $R_{mc}$, we can correct for miscentering (by expanding in $R_{mc}/R$) using

$G_+ \to G_+ + \frac14 \left(\frac{R_{mc}}{R}\right)^2 (4 G_+(R) - R G_+'(R) - R^2 G_+''(R) )$

Note that:

$R G_+'(R) + R^2 G_+''(R) = \partial_{\ln R}^2 G_+$
"""

# ╔═╡ 61671b5e-9a09-49ed-ba69-37852662f803
begin
	function miscenter_correct_G(
		miscenter_correct::Union{MiscenterCorrectNone, MiscenterCorrectSmallRmc}, interpolate::I; R, G
	) where I<: AbstractInterpolate
		G
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
		miscenter_correct::MiscenterCorrectSmallRmcPreprocessG, interpolate::I;
		R, G
	) where I<:AbstractInterpolate
		__calculate_miscenter_corrected_GMsunpc2_small_Rmc(
			interpolate;
			RMpc=R ./ u"Mpc" .|> NoUnits,
			Rmc²Mpc=miscenter_correct.Rmc² / u"Mpc^2" |> NoUnits,
			GMsunpc2=G ./ u"Msun/pc^2" .|> NoUnits
		) .* u"Msun/pc^2"
	end	
end

# ╔═╡ bb1aa65a-90b7-4d39-8f54-e1b306d506bb
md"""
# Actual deprojection

## $I(R)$ and $J(R)$ integrals
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

- For NFW extrapolation: Keep our lives simple: Just do the integral numerically.
"""

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

- For NFW extrapolate, we exploit that a) $J$ is related in a simple way to $\Delta \Sigma$ and 2) we _know_ the NFW parameters for the extrapolation so we _know_ $\Delta \Sigma$ and $G$ for the extraplation -- so we can just solve the following equation for $J$ algebraically (Eq. (B5) in Mistele&Durakovic 2024)

  $\Delta \Sigma = \frac{1}{f_c} \frac{G f}{1 - G f} (1 - e^{-I} f_c J)$

  with *everything* evaluated at $R = R_{\mathrm{max}}$. The result is:

  $J = \frac{e^{+I} }{f_c} \left(1 - \frac{f \Delta \Sigma}{G f} (1 - G f) \right)$

  This gives $J(R_{\mathrm{max}})$ with the RHS evaluated for the NFW continuation.
  Using $G = \Delta \Sigma/(1 - f \Sigma)$ (approximately in the regime we're interested int), this becomes

  $J = e^{+I} (\Delta \Sigma + \Sigma)$

  For NFW, this is just (use $\Delta \Sigma = 2 \rho_s r_s(g + f)$)

  $J(R_{\mathrm{max}}) = e^{I(R_{\mathrm{max}})} \cdot 2 \rho_s r_s g_{\mathrm{NFW}}\left(\frac{R_{\mathrm{max}}}{r_s}\right)$
"""

# ╔═╡ fa01d0c3-f793-44a8-a406-776b77786aa9
md"""
## Interpolations
"""

# ╔═╡ 49397343-2023-4627-89e6-74170976c890
begin
	function get_interpolation_RMpc(
		interpolate::InterpolateR; RMpc::typeof([1.0]), values
	)
		func = BSplineKit.extrapolate(
			BSplineKit.interpolate(
				RMpc, values,
				# Linear interpolation = BSplineOrder(2), so +1
				BSplineKit.BSplineOrder(interpolate.order+1)
			),
			# Just because of floating point inexactness. We don't actually need
			# values outside the range of RMpc. To avoid accidental mistakes,
			# we actually enforce at the upper boundary. But at the lower boundary
			# we don't and _sometimes_ that seems to be an issue in practice.
			# (`interpolate` itself gives 0 outsides `RMpc`...)
			BSplineKit.Flat()
		)

		maxRMpc = maximum(RMpc)
		RMpc -> let
			RMpc <= maxRMpc || throw("interpolation must be evaluated at <= maxRMpc")
			func(RMpc)
		end
	end
	function get_interpolation_RMpc(
		interpolate::InterpolateLnR; RMpc::typeof([1.0]), values
	)
		logfunc = BSplineKit.extrapolate(
			BSplineKit.interpolate(
				log.(RMpc), values,
				# Linear interpolation = BSplineOrder(2), so +1
				BSplineKit.BSplineOrder(interpolate.order+1)
			),
			# See comment above for why
			BSplineKit.Flat()
		)

		maxRMpc = maximum(RMpc)
		RMpc -> let
			RMpc <= maxRMpc || throw("interpolation must be evaluated at <= maxRMpc")
			logfunc(log(RMpc))
		end
	end	

	function get_interpolation_RMpc_dlog(
		interpolate::InterpolateR; RMpc::typeof([1.0]), values
	)
		func = BSplineKit.extrapolate(
			BSplineKit.interpolate(
				RMpc, values,
				# Linear interpolation = BSplineOrder(2), so +1
				BSplineKit.BSplineOrder(interpolate.order+1)
			),
			# See comment above
			BSplineKit.Flat()
		)
		dfunc = BSplineKit.Derivative(1) * func
		maxRMpc = maximum(RMpc)
		RMpc -> let
			RMpc <= maxRMpc || throw("interpolation must be evaluated at <= maxRMpc")
			RMpc*dfunc(RMpc)
		end
	end
	function get_interpolation_RMpc_dlog(
		interpolate::InterpolateLnR; RMpc::typeof([1.0]), values
	)
		logfunc = BSplineKit.extrapolate(
			BSplineKit.interpolate(
				log.(RMpc), values,
				# Linear interpolation = BSplineOrder(2), so +1
				BSplineKit.BSplineOrder(interpolate.order+1)
			),
			# See comment above
			BSplineKit.Flat()
		)
		dlogfunc = BSplineKit.Derivative(1) * logfunc
		maxRMpc = maximum(RMpc)
		RMpc -> let
			RMpc <= maxRMpc || throw("interpolation must be evaluated at <= maxRMpc")
			dlogfunc(log(RMpc))
		end
	end
end

# ╔═╡ 42855db1-3956-429e-afe7-46d385e5148c
md"""
## $M$ and covariance
"""

# ╔═╡ 6bfbe740-2993-4ae1-ad30-54ea923e0e1c
md"""
## $M$ from $\Delta \Sigma$ incl. miscentering correction
"""

# ╔═╡ dfe40541-396b-485b-bcb6-d70730a24867
md"""
### Power decay tail

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
  &M^{\mathrm{tail}}(R) \\
  &= 4 R^2\int_0^{\theta_{m}} d\theta \Delta \Sigma(R/\sin \theta)\\
  &= \frac{4 R^2}{f_\infty} G f(R_{\mathrm{max}}) \frac{R_{\mathrm{max}}}{R} \bigg(
  \\
  & \quad \quad 1 - \frac12 G f(R_{\mathrm{max}}) \frac{R_{\mathrm{max}}}{R} \theta_{m} - \cos(\theta_{m}) + \frac12 G f(R_{\mathrm{max}}) \frac{R_{\mathrm{max}}}{R} \cos(\theta_{m}) \sin(\theta_{m}) \\
  &\quad \bigg)
  \end{aligned}$

  For $n=2$:

  $\begin{aligned}
  &M^{\mathrm{tail}}(R) \\
  &= 4 R^2 \int_0^{\theta_{m}} d\theta \Delta \Sigma(R/\sin \theta)\\
  &= \frac{4 R^2}{f_\infty} \frac12  G f(R_{\mathrm{max}}) \left(\frac{R_{\mathrm{max}}}{R}\right)^2 \left(
   \theta_m - \cos(\theta_{m}) \sin(\theta_{m}) 
  \right)
  \end{aligned}$

  For other $n$: Just do it numerically, works quite well (see comments in code below) :)
"""

# ╔═╡ fa506a97-1c00-488d-a4d1-18b878bc3640
md"""
### NFW tail

Let's assume a fixed concentration-mass relation, say $c = c(M_{200})$.
This implies we also know $r_s$ as a function of $M_{200}$.

$r_s(M_{200}) \equiv \frac{r_{200}(M_{200})}{c(M_{200})}$ 

Then, we can write the NFW $\Delta \Sigma$ as:

$\Delta \Sigma_{\mathrm{NFW}} = \Delta \Sigma_{\mathrm{NFW}}(R|r_s(M_{200}), M_{200})$

i.e. the NFW excess-surface density depends on only a single parameter, in this case $M_{200}$ (at a given projected radius $R$).

Thus, we can now find an *NFW tail* in a very simple way from the *last measured data point* at $R_{\mathrm{max}}$. We just need to solve this equation

$\Delta \Sigma(R_{\mathrm{max}})_{\mathrm{measured}} = \Delta \Sigma_{\mathrm{NFW}}(R_{\mathrm{max}}|r_s(M_{200}), M_{200})$

for $M_{200}$. Or the same for our (approximate) formula $G = \Delta \Sigma/(1 - f_c \Sigma)$.
"""

# ╔═╡ 0134ff7b-b627-4016-9a4b-d686207111b3
begin
	fNFW(x) = if x < 1
		(1/(1-x^2))*( -1 + (2/sqrt(1-x^2)) * atanh(sqrt( (1-x)/(1+x) )) )
	elseif x == 1
		1/3
	else
		(1/(x^2-1))*( 1 - (2/sqrt(x^2-1)) * atan(sqrt( (x-1)/(x+1) )) )
	end

	gNFW(x) = if x < 1
		(2/x^2)*( (2/sqrt(1-x^2)) * atanh(sqrt( (1-x)/(1+x) )) + log(x/2) )
	elseif x == 1
		2*(1+log(1/2))
	else
		(2/x^2)*( (2/sqrt(x^2-1)) * atan(sqrt( (x-1)/(x+1) )) + log(x/2) )
	end

	# That's a job for AutoDiff -- so I don't have to put the long derivative
	# expression here (but I did cross-check with Mathematica)
	d_gNFW(x) = if x == 1
		-(10/3)+log(16)
	else
		ForwardDiff.derivative(gNFW, x)
	end

	ΔΣ_NFW(R; rs, ρs) = 2*ρs*rs*(gNFW(R/rs) - fNFW(R/rs))
	Σ_NFW(R; rs, ρs) = 2*ρs*rs*fNFW(R/rs)

	Gf_NFW(R; rs, ρs, f∞) = (
		f∞*ΔΣ_NFW(R; ρs, rs) |> NoUnits
	) / (
		# Without that `NoUnits`, ForwardDiff.jl isn't happy...
		1 - (f∞*Σ_NFW(R; ρs, rs) |> NoUnits)
	)
	
	# Simple continuity smoke test
	@plutoonly let
		@assert abs(fNFW(1.0)/fNFW(1.00001)  -1 ) < 1e-4
		@assert abs(fNFW(1.0)/fNFW(0.99999)  -1 ) < 1e-4
		@assert abs(gNFW(1.0)/gNFW(1.00001)  -1 ) < 1e-5
		@assert abs(gNFW(1.0)/gNFW(0.99999)  -1 ) < 1e-5

		# From Mathematica
		@assert abs(d_gNFW(.1)/(-9.70774) - 1) < 1e-6
		@assert abs(d_gNFW(1.1)/(-0.481767) - 1) < 1e-6
		@assert d_gNFW(1.0) == -(10/3)+log(16)
		@assert abs(d_gNFW(1.0)/d_gNFW(0.99999)  -1 ) < 1e-4
		@assert abs(d_gNFW(1.0)/d_gNFW(1.00001)  -1 ) < 1e-4
	end
end

# ╔═╡ 3e5aa347-e19e-4107-a85e-30aa2515fb3a
begin
	function I_R∞_tail(extrapolate::ExtrapolatePowerDecay, pre; RMpcMax)
		# ∫ dR' ... from Rmax to ∞
		GfTail = pre
		n = extrapolate.n
		-(2/n)*log(1 - GfTail)
	end

	function I_R∞_tail(extrapolate::ExtrapolateNFW, pre; RMpcMax)
		# ∫ dR' ... from Rmax to ∞
		(rs, ρs, f∞) = pre
		quadgk(RMpc -> let
			Gf = Gf_NFW(RMpc*u"Mpc"; rs, ρs, f∞)
			(2/RMpc)*Gf/(1 - Gf)
		end, RMpcMax, Inf)[1]
	end
	
	function calculate_I_R∞(
		extrapolate, interpolate::InterpolateR, pre; RMpc, Gf,
	)
		# Solve in terms of X = Rmax - R so we can impose I(RMax) as an *initial*
		# condition (rather than a final condition). Because that's what `ODEProblem`
		# wants
		RMpcMax = maximum(RMpc)
		RMpcMin = minimum(RMpc)
		prob = ODEProblem(
			# RHS of I'(X) = ...
			(I, p, X) -> let
				RMpc = RMpcMax - X
				SA[(2/RMpc)*Gf(RMpc)/(1 - Gf(RMpc))]
			end,
			# Initial condition, i.e. ∫ dR' ... from Rmax to ∞
			SA[I_R∞_tail(extrapolate, pre; RMpcMax)],
			# R interval where to solve
			(0, RMpcMax-RMpcMin)
		)
		s = solve(prob, Tsit5())
		RMpc -> let
			@assert RMpc <= RMpcMax "I(R) only calculated up to last bin center!"
			s(RMpcMax - RMpc, idxs=1)
		end
	end
	
	function calculate_I_R∞(
		extrapolate, interpolate::InterpolateLnR, pre; RMpc, Gf,
	)
		# Solve in terms of X=ln(Rmax)-ln(R) so we can impose I(RMax) as an *initial*
		# condition (rather than a final condition). Because that's what `ODEProblem`
		# wants
		RMpcMax = maximum(RMpc)
		RMpcMin = minimum(RMpc)
		prob = ODEProblem(
			# RHS of I'(X) = ... NB: no 1/RMpc
			(I, p, X) -> let
				RMpc = RMpcMax*exp(-X) # same as exp(ln(Rmax) - X)
				SA[2*Gf(RMpc)/(1 - Gf(RMpc))] 
			end,
			# Initial condition
			SA[I_R∞_tail(extrapolate, pre; RMpcMax)],
			# R interval where to solve
			(0, log(RMpcMax)-log(RMpcMin))
		)
		s = solve(prob, Tsit5())
		RMpc -> let
			@assert RMpc <= RMpcMax "I(R) only calculated up to last bin center!"
			s(log(RMpcMax/RMpc), idxs=1)
		end
	end
end

# ╔═╡ 64e5f173-11be-4dbf-b9ab-f652c50d9c09
begin
	function J_R∞_tail(extrapolate::ExtrapolatePowerDecay, pre; RMpcMax, f̂∞, I)
		# ∫ dR' ... from Rmax to ∞
		GfTail = pre
		n = extrapolate.n
		(1/f̂∞)*((1 - GfTail)^(-2/n) - 1)
	end

	function J_R∞_tail(extrapolate::ExtrapolateNFW, pre; RMpcMax, f̂∞, I)
		# ∫ dR' ... from Rmax to ∞
		(rs, ρs, _) = pre
		exp(I(RMpcMax))*2*ρs*rs*gNFW(RMpcMax*u"Mpc"/rs) / u"Msun/pc^2" |> NoUnits
	end
	
	function calculate_J_R∞(
		extrapolate, interpolate::InterpolateR, pre;
		RMpc::typeof([1.0]), Gf, Ĝ, I
	)
		# Solve in terms of X = Rmax - R so we can impose J(RMax) as an *initial*
		# condition (rather than a final condition). Because that's what `ODEProblem`
		# wants
		RMpcMax = maximum(RMpc)
		RMpcMin = minimum(RMpc)
		f̂∞ = Gf(RMpcMax)/Ĝ(RMpcMax)
		prob = ODEProblem(
			# RHS of I'(X) = ...
			(J, p, X) -> let
				RMpc = RMpcMax - X
				SA[(2/RMpc)*(1/exp(-I(RMpc)))*Ĝ(RMpc)/(1 - Gf(RMpc))]
			end,
			# Initial condition
			SA[J_R∞_tail(extrapolate, pre; RMpcMax, f̂∞, I)],
			# R interval where to solve
			(0, RMpcMax-RMpcMin)
		)
		s = solve(prob, Tsit5())
		RMpc -> let
			@assert RMpc <= RMpcMax "J(R) only calculated up to last bin center!"
			s(RMpcMax - RMpc, idxs=1)
		end
	end
	function calculate_J_R∞(
		extrapolate, interpolate::InterpolateLnR, pre;
		RMpc::typeof([1.0]), Gf, Ĝ, I
	)
		# Solve in terms of X=ln(Rmax)-ln(R) so we can impose J(RMax) as an *initial*
		# condition (rather than a final condition). Because that's what `ODEProblem`
		# wants
		RMpcMax = maximum(RMpc)
		RMpcMin = minimum(RMpc)
		f̂∞ = Gf(RMpcMax)/Ĝ(RMpcMax)
		prob = ODEProblem(
			# RHS of I'(X) = ... NB: no 1/RMpc
			(J, p, X) -> let
				RMpc = RMpcMax*exp(-X) # same as exp(ln(Rmax) - X)
				SA[2*(1/exp(-I(RMpc)))*Ĝ(RMpc)/(1 - Gf(RMpc))]
			end,
			# Initial condition
			SA[J_R∞_tail(extrapolate, pre; RMpcMax, f̂∞, I)],
			# R interval where to solve
			(0, log(RMpcMax)-log(RMpcMin))
		)
		s = solve(prob, Tsit5())
		RMpc -> let
			@assert RMpc <= RMpcMax "J(R) only calculated up to last bin center!"
			s(log(RMpcMax) - log(RMpc), idxs=1)
		end
	end
end

# ╔═╡ 3f004698-b952-462f-8824-5c78ab1e08ad
module __demo
	using Unitful
	using UnitfulAstro
	import ..gNFW, ..fNFW, ..d_gNFW

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
	
		@assert abs(gx) < 1e-13 "Cross-check: γx and gx should be zero"
	
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

	struct ProfileNFW <: AbstractProfile
		ρ0::typeof(1.0u"Msun/Mpc^3")
		rs::typeof(1.0u"Mpc")
		x0::typeof(1.0u"Mpc")
		y0::typeof(1.0u"Mpc")
	end
	ProfileNFW(; ρ0, rs, x0, y0) = ProfileNFW(
		ρ0,
		rs,
		x0,
		y0
	)
	function calculate_κ(p::ProfileNFW, R; Σcrit)
		# NB: κ= Σ/Σcrit -- but lens equations have factor of 2: Δ ϕ = 2 κ
		# Σ from Umetsu2020, Eq. (118)/(119)
		x = R/p.rs
		Σ = 2*p.ρ0*p.rs*fNFW(x)
		Σ / Σcrit |> NoUnits
	end
	function calculate_dR_lenspot(p::ProfileNFW, R; Σcrit)
		x = R/p.rs
		Σ̄ = 2*p.ρ0*p.rs*gNFW(x)
		(R/Σcrit) * Σ̄
	end
	function calculate_ddR_lenspot(p::ProfileNFW, R; Σcrit)
		x = R/p.rs
		(2*p.ρ0*p.rs/Σcrit) * (gNFW(x) + x * d_gNFW(x) )
	end
end

# ╔═╡ 8193ca3f-749d-4734-b2c9-9db46a0458c0
const Gf_NFW_approx_miscentering_applied = let

	# That's a helper function defined as
	#   H(lnx; p) = p*(g - f)/(1 - p * f)
	# Compare:
	#   Gf_NFW(R) = fc*(2rsρs)*(g(x=r/Rs) - f(...))/(1 - f_c * (2rsρs) * f(...))
	# So we have:
	#   Gf_NFW(R) = H(ln(R/rs); p=f_c*2rsρs)
	#
	# We need H''(lnx; p).
	# 
	# We could just let ForwardDiff.jl calculate this (and it's actually not
	# horrible, we had this in a previous version). But we can speed this part
	# up a lot by actually working out the explicit formulas.
	# Formulas from Mathematica, see `faster-NFW-miscentering-formulas.nb`
	∂_lnx_fNFW(x) = let
		sq = sqrt(abs((x-1)/(x+1)))
		if x < 1
			-(
				(x-1)*(1+2x^2) + 6x^2*sq*atanh(sq)
			) / (
				(1+x)^2*(-1+x)^3
			)
		else
			-(1+2x^2)/(-1+x^2)^2 + 6x^2*atan(sq)/(x^2-1)^(5/2)
		end
	end

	∂²_lnx_fNFW(x) = let
		sq = sqrt(abs((x-1)/(x+1)))
		x^2 * if x < 1
			(
				(x-1)*(11+4x^2)
				+6*(2+3x^2)*sq*atanh(sq)
			) / (
				(1+x)^3*(x-1)^4
			)
		else
			(11+4x^2)/(x^2-1)^3 - 6*(2+3x^2)*atan(sq)/(x^2-1)^(7/2)
		end
	end

	∂_lnx_gNFW(x) = let
		sq = sqrt(abs((x-1)/(x+1)))
		sq_atan_sq = if x < 1
			-sq*atanh(sq)
		else
			sq*atan(sq)
		end
		(
			4*(2-3x^2)*sq_atan_sq + 2*(x-1)*(
				x^2+
				(x^2-1)*log(4)
				-2(x^2-1)*log(x)
			)
		) / (
			(x-1)^2*x^2*(1+x)
		)
	end

	∂²_lnx_gNFW(x) = let
		sq = sqrt(abs((x-1)/(x+1)))
		sq_atan_sq = if x < 1
			-sq*atanh(sq)
		else
			sq*atan(sq)
		end
		(
			4*sq_atan_sq*(4-10x^2+9x^4)+
			+2*(x-1)*(x^2-4x^4+4*(-1+x^2)^2*log(x/2))
		) / (
			(x-1)*x^2*(x^2-1)^2
		)
	end

	∂²_lnx_H(x; p) = if abs(x-1) < 1e-2
		# This becomes numerically tricky for x close to 1.
		# So just use linear Taylor expansion there
		a = 4p*(
			-6525 + 903p - 206p^2 +
			9450log(2) - 1710p*log(2) + 276p^2*log(2)
		)/( 175(-3 + p)^3)
		b = 16p*(
			675(-72+35*log(8))
			+ p*(
				(5310-5889p+338p^2)
				+ (-3150+(2325-166p)p)*log(8)
			)
		) / (875*(p-3)^4)
		a + b*log(x)
	else
		df = ∂_lnx_fNFW(x)
		dg = ∂_lnx_gNFW(x)
		
		d²f = ∂²_lnx_fNFW(x)
		d²g = ∂²_lnx_gNFW(x)
		
		A = gNFW(x) - fNFW(x)
		B = fNFW(x)

		dA = dg - df
		dB = df

		d²A = d²g - d²f
		d²B = d²f
		
		(
			2p^2*dA*dB/(1-p*B)^2
			+ 2p^3*A*dB^2/(1-p*B)^3
			+ p*d²A/(1 - p*B)
			+ p^2*A*d²B/(1-p*B)^2
		)
	end

	@plutoonly let
		@assert abs(∂_lnx_fNFW(.1)/(-0.948628645550817) - 1) < 1e-15
		@assert abs(∂_lnx_fNFW(1.1)/(-0.3677094931729954) - 1) < 1e-15
		
		# TODO: Why is this one so much worse?
		@assert abs(∂_lnx_gNFW(.1)/(-0.970774460018575) - 1) < 1e-12
		@assert abs(∂_lnx_gNFW(1.1)/(-0.5299439541930584) - 1) < 1e-15

		@assert abs(∂²_lnx_fNFW(.1)/0.07503419144630517 - 1) < 1e-15
		@assert abs(∂²_lnx_fNFW(1.1)/0.3343068883999718 - 1) < 1e-15

		# TODO: Why is this one so much worse?
		@assert abs(∂²_lnx_gNFW(.1)/0.04429162893533274 - 1) < 1e-11
		@assert abs(∂²_lnx_gNFW(1.1)/0.3244689220401106 - 1) < 1e-15

		f = lnx -> fNFW(exp(lnx))
		d_lnx_f(lnx) = ForwardDiff.derivative(f, lnx)
		dd_lnx_f(lnx) = ForwardDiff.derivative(d_lnx_f, lnx)
		@assert abs(d_lnx_f(log(.2)) / ∂_lnx_fNFW(.2) - 1) < 1e-15
		@assert abs(d_lnx_f(log(1.2)) / ∂_lnx_fNFW(1.2) - 1) < 1e-14
		@assert abs(dd_lnx_f(log(.2))/ ∂²_lnx_fNFW(.2) - 1) < 1e-15
		@assert abs(dd_lnx_f(log(1.2))/ ∂²_lnx_fNFW(1.2) - 1) < 1e-13
	
		g = lnx -> gNFW(exp(lnx))
		d_lnx_g(lnx) = ForwardDiff.derivative(g, lnx)
		dd_lnx_g(lnx) = ForwardDiff.derivative(d_lnx_g, lnx)
		@assert abs(d_lnx_g(log(.2)) / ∂_lnx_gNFW(.2) - 1) < 1e-13
		@assert abs(d_lnx_g(log(1.2)) / ∂_lnx_gNFW(1.2) - 1) < 1e-14
		@assert abs(dd_lnx_g(log(.2))/ ∂²_lnx_gNFW(.2) - 1) < 1e-12
		@assert abs(dd_lnx_g(log(1.2))/ ∂²_lnx_gNFW(1.2) - 1) < 1e-13
	
		H(lnx; p) = let
			x = exp(lnx)
			p*(gNFW(x) - fNFW(x)) / (1 - p*fNFW(x))
		end
		dH(lnx; p) = ForwardDiff.derivative(lny -> H(lny; p), lnx)
		d²H(lnx; p) = ForwardDiff.derivative(lny -> dH(lny; p), lnx)
	
		@assert abs(∂²_lnx_H(.1; p=.2)/d²H(log(.1); p=.2) - 1) < 1e-11
		@assert abs(∂²_lnx_H(1.1; p=.2)/d²H(log(1.1); p=.2) - 1) < 1e-11
	end

	# This is:
	#  (G₊_tail - ϵ² D G₊_tail)(R)
	# with ϵ = Rmc/Rmax and (D F)(R) ≡ (1/4) * (4F(R) - R F'(R) - R² F''(R))	
	# Note that D can be written as
	#   DF(R) = (1/4) * (4F - ∂²_(ln R) F)
	# And note ∂²_(ln R) ... = ∂²_(ln (R/rs)) ...
	# Instaed of removing miscentering using our O(ϵ²) formula,we're applying it.
	(
		R; Rmc², rs, ρs, f∞
	) -> let
		Gf = Gf_NFW(R; rs, ρs, f∞)

		fc2rsρs = f∞*2*rs*ρs |> NoUnits
		∂_terms = ∂²_lnx_H(R/rs; p=fc2rsρs)

		Gf - (1/4) * (Rmc²/R^2) * (4*Gf - ∂_terms)
	end
end

# ╔═╡ ae4b04aa-f4a2-4060-89a6-211eb40a1808
@plutoonly let
	# Simple checks against Mathematica of applying miscentering to NFW
	
	test1 = Gf_NFW_approx_miscentering_applied(
		1.0u"Mpc";
		Rmc²=(.2u"Mpc")^2, rs=.5u"Mpc", ρs=1e14u"Msun/Mpc^3", f∞=1e-3/u"Msun/pc^2"
	)
	# From Mathematica (didn't use the ln(R) form to cross-check that)
	@assert abs(test1/0.0166537 - 1) < 1e-5

	test2 = Gf_NFW_approx_miscentering_applied(
		1.0u"Mpc";
		Rmc²=(.2u"Mpc")^2, rs=1.0u"Mpc", ρs=1e14u"Msun/Mpc^3", f∞=1e-3/u"Msun/pc^2"
	)
	# From Mathematica at rs=R, where things are tricky
	@assert abs(test2/0.0577428-1) < 1e-6
end

# ╔═╡ f42c2a3a-ac7f-45cd-84dc-8eccd147ccab
const NFW_find_rs_ρs_from_last_Gf = let

	# M200 concentration relation from Maccio et al 2008, WMAP5
	# (formulas taken from LI et al 2020 Eq. (28) and Eq. (29))
	# NOTE: There is a typo in Li (it's a + b not a - b!)
	c200_from_M200 = let
		a = 0.830
		b = -0.098
		M0 = 1e12u"Msun"
		(cm::CMRelationMaccio2008, M200) -> 10^(a + b*log10(M200*cm.h/M0))
	end

	r200_from_M200(M200; ρcrit) = cbrt( M200 / ((4π/3)*200*ρcrit) ) |> u"Mpc"

	function ρs_from_M200_c200_rs(; M200, c200, rs)
		# From Wikipedia
		# M200 = 4π ρs rs^3 [ln(1+c200) - c200/(1+c200)]
		# So:
		# ρs = (M200/4πrs^3) * 1/[ln(1+c200) - c200/(1+c200)]
		(M200/(4π*rs^3)) / ( log(1+c200) - c200/(1+c200) ) |> u"Msun/Mpc^3"
	end

	function(cm::CM; Gf_NFW_func, GfTail, Rtail) where {
		CM <: AbstractMassConcentrationRelation
	}
		# What to do if the last shear data point is negative?
		# The standard NFW mass-concentration relation is defined
		# only for *positive* masses which give *positive* shear.
		# 
		# Very simple option implemented here:
		# - we match NFW to |Gf(Rmax)| which is always positive
		# - we give ρs the sign of Gf(Rmax)
		# 
		# Nice property: If we have a shear which is scattered around zero,
		# it gives, on average, a zero reconstructed mass.
		# (for non-zero signal it doesn't work quite as nicely because the NFW
		#  matching is non-linear.)
		sign_GfTail = sign(GfTail)
		abs_GfTail = abs(GfTail)
		
		# Find M200 from ΔΣ_NFW(Rmax|M200) = ΔΣ_measured(Rmax)
		get_rs_ρs = M200 -> let
			# Use c-M relation
			c200 = c200_from_M200(cm, M200)
			# Definitions of r200, rs, ρs (need cosmology in terms of ρcrit)
			r200 = r200_from_M200(M200; cm.ρcrit)
			rs = r200/c200
			ρs = ρs_from_M200_c200_rs(; M200, c200, rs)
			(rs, ρs)
		end
		log10_M200_matched = Roots.find_zero(
			log10_M200 -> let
				(rs, ρs) = get_rs_ρs(10 ^ log10_M200 * u"Msun")
				Gf_NFW_tail = Gf_NFW_func(Rtail; ρs, rs)
				Gf_NFW_tail > 0 || return 100.0 # Something larger than 1
				Gf_NFW_tail < 1 || return 10.0 
				Gf_NFW_tail  - abs_GfTail # match to |Gf(Rmax)|, see text above
			end,
			(8, 19), # (10^8 -- 10^19) Msun should cover everything realistic
			# Roots.Bisection() gives all-zeros for ForwardDiff! So use A42...
			# https://discourse.julialang.org/t/autodiff-ing-a-function-defined-by-the-result-of-roots-find-zero-fails/87753/3
			Roots.A42()
		)

		(rs, ρs) = get_rs_ρs(10 ^ log10_M200_matched * u"Msun")

		(rs, sign_GfTail*ρs #= sign of Gf(Rmax), see text above =#)
	end
end

# ╔═╡ ea9fc39e-ba29-4502-927f-d2ca77e3b4e7
md"""
### Bulk/$M$ itself
"""

# ╔═╡ 861b3ac9-14df-462a-9aa8-40ef9a521b81
md"""
## $\Delta \Sigma$ and more from $G_+$
"""

# ╔═╡ 6399685a-1e4d-41fc-a3cd-01c61bbf56cf
begin
    # Optionally, neglect κ when deprojecting
    # (i.e. leave out the non-linear step of going from G+ -> ΔΣ and instead just
    # assume G+ == ΔΣ)
    abstract type AbstractNeglectKappa end
    struct NeglectKappa <: AbstractNeglectKappa end
    struct NoNeglectKappa <: AbstractNeglectKappa end
end

# ╔═╡ c86ab391-86c3-44f8-b0b9-20fb70c4dc87
function calculate_M_tail(
	extrapolate::ExtrapolatePowerDecay, ::NoNeglectKappa, pre;
	θlim, f∞, rMpc, rMpcTail
)
	n = extrapolate.n
	rMpcMax = rMpcTail
	Gfmax = pre

	if n == 1
		(4*(rMpc*u"Mpc")^2/f∞)*Gfmax*(rMpcMax/rMpc)*(
			1
			- (1/2)*Gfmax*(rMpcMax/rMpc)*θlim
			- cos(θlim)
			+ (1/2)*Gfmax*(rMpcMax/rMpc)*cos(θlim)*sin(θlim)
		) |> u"Msun"
	elseif n == 2
		(4*(rMpc*u"Mpc")^2/f∞)*(1/2)*Gfmax*(rMpcMax/rMpc)^2*(
			θlim - cos(θlim)*sin(θlim)
		) |> u"Msun"
	else
		# We could use that integral for the other n as well. It works well.
		# But I've implemented them already and they're faster of course, so
		# let's keep them for now.
		ΔΣtail(rMpc) = (1/f∞)*Gfmax*(rMpcMax/rMpc)^n*(1 - Gfmax*(rMpcMax/rMpc)^n)^(2/n-1)

		4*(rMpc*u"Mpc")^2*u"Msun/pc^2"*quadgk(
			θ -> ΔΣtail(rMpc/sin(θ))/u"Msun/pc^2" |> NoUnits,
			0, θlim
		)[1] |> u"Msun"
	end
end

# ╔═╡ 5788ae15-34ca-4230-8d6f-52b2585123ce
function calculate_M_tail(
	extrapolate::ExtrapolatePowerDecay, ::NeglectKappa, pre;
	θlim, f∞, rMpc, rMpcTail
)
	n = extrapolate.n
	rMpcMax = rMpcTail
	Gfmax = pre

	if n == 1
		(4*(rMpc*u"Mpc")^2/f∞)*Gfmax*(rMpcMax/rMpc)*(
			1
			- cos(θlim)
		) |> u"Msun"
	elseif n == 2
		(4*(rMpc*u"Mpc")^2/f∞)*(1/2)*Gfmax*(rMpcMax/rMpc)^2*(
			θlim - cos(θlim)*sin(θlim)
		) |> u"Msun"
	else
		# We could use that integral for the other n as well. It works well.
		# But I've implemented them already and they're faster of course, so
		# let's keep them for now.
		ΔΣtail(rMpc) = (1/f∞)*Gfmax*(rMpcMax/rMpc)^n

		4*(rMpc*u"Mpc")^2*u"Msun/pc^2"*quadgk(
			θ -> ΔΣtail(rMpc/sin(θ))/u"Msun/pc^2" |> NoUnits,
			0, θlim
		)[1] |> u"Msun"
	end
end

# ╔═╡ 6f593629-bd08-44ad-8941-54c95f131908
function calculate_M_tail(
	extrapolate::ExtrapolateNFW, ::NoNeglectKappa, pre;
	θlim, f∞, rMpc, rMpcTail
)
	(rs, ρs, _) = pre
	r = rMpc*u"Mpc"
	4*r^2*u"Msun/pc^2"*quadgk(
		θ -> ΔΣ_NFW(r/sin(θ); ρs, rs)/u"Msun/pc^2" |> NoUnits,
		0, θlim
	)[1] |> u"Msun"
end

# ╔═╡ c449a9c8-1739-481f-87d5-982532c2955c
begin
	# The tail needs to be calculated analytically. Reason: The tail goes to R -> ∞.
	# That's ok for Gf(R) and f(R) because we could just extrapolate. But it's not ok
	# for I(R) and J(R) which also enter ΔΣ(R), because those we solved numerically
	# only up to R=Rmax (and the `ODESolution` extrapolation beyond last data point
	# is often completely off).
	
	function calculate_M_from_ΔΣ(
		extrapolate::E,
		interpolate::I,
		::Union{MiscenterCorrectNone, MiscenterCorrectSmallRmcPreprocessG},
		neglect_kappa,
		pre;
		ΔΣ, rMpc, f∞, Ĝvalues,
	) where {E<:AbstractExtrapolate, I<:AbstractInterpolate}
		rMpcTail = maximum(rMpc)
	
		M(rMpc) = if rMpc < rMpcTail
			θlim = asin(rMpc/rMpcTail)
			bulk = 4*(rMpc*u"Mpc")^2*u"Msun/pc^2"*quadgk(
				θ -> ΔΣ(rMpc/sin(θ))/u"Msun/pc^2" |> NoUnits,
				θlim, π/2
			)[1] |> u"Msun"
			tail = calculate_M_tail(
				extrapolate, neglect_kappa, pre; θlim, f∞, rMpc, rMpcTail)
			bulk + tail
		else
			calculate_M_tail(
				extrapolate, neglect_kappa, pre; θlim=π/2, f∞, rMpc, rMpcTail)
		end
	end
	
	function calculate_M_from_ΔΣ(
		extrapolate::Union{ExtrapolatePowerDecay, ExtrapolateNFW},
		interpolate::I,
		miscenter_correct::MiscenterCorrectSmallRmc,
		neglect_kappa,
		pre;
		ΔΣ, rMpc, f∞, Ĝvalues,
	) where I <: AbstractInterpolate
		Rmc²Mpc = miscenter_correct.Rmc² / u"Mpc^2" |> NoUnits
		ΔΣ̂(RMpc) = ΔΣ(RMpc)/u"Msun/pc^2" |> NoUnits
		
		rMpcTail = maximum(rMpc)

		dlog_Ĝ = get_interpolation_RMpc_dlog(interpolate; RMpc=rMpc, values=Ĝvalues)

		# For linear interpolation, these 
		# are discontinuous! That's however not a problem -- as long as we have this 
		# propertly: It's important that derivatives are like
		#   f'(x) = lim_{δ -> 0^+} = f'(x+δ)
		# Assert that! (but not needed at very last data point...)
		# Also check that this does _not_ hold if we take the limit the other way
		# (just to be sure we understand what's going on)
		__makesure_interpolation_ok(::InterpolateR) = let
			myrMpc = rMpc[begin:end-1]
			myrMpcLarger = 1.0001 .* rMpc[begin:end-1]
			myrMpcSmaller = .9999 .* rMpc[begin:end-1]
			@assert all(abs.(dlog_Ĝ.(myrMpc)./myrMpc .- dlog_Ĝ.(myrMpcLarger)./myrMpcLarger) .< 1e-12)
			@assert !any(abs.(dlog_Ĝ.(myrMpc)./myrMpc .- dlog_Ĝ.(myrMpcSmaller)./myrMpcSmaller) .< 1e-12)
		end
		__makesure_interpolation_ok(::InterpolateLnR) = let
			myrMpc = rMpc[begin:end-1]
			myrMpcLarger = 1.0001 .* rMpc[begin:end-1]
			myrMpcSmaller = .9999 .* rMpc[begin:end-1]
			@assert all(abs.(dlog_Ĝ.(myrMpc) .- dlog_Ĝ.(myrMpcLarger)) .< 1e-12)
			@assert !any(abs.(dlog_Ĝ.(myrMpc) .- dlog_Ĝ.(myrMpcSmaller)) .< 1e-12)
		end
		interpolate.order == 1 && __makesure_interpolation_ok(interpolate)

		# Basically, what we do is
		#  M ~ ∫dθ (ΔΣ_obs + ϵ² DΔΣ_obs)|R=r/sin θ
		# Which we split into a bulk and a tail.
		#  M ~ ∫_(R < Rmax) (ΔΣ_obs + ϵ² DΔΣ_obs)|R=r/sin θ
		#     + ∫_(R > Rmax) (ΔΣ_obs + ϵ² DΔΣ_obs)|R=r/sin θ
		# The split is why we need boundary terms below in the bulk integral
		# (the derivation has integration by parts which gives the boundary terms)!
		#
		# For the tail: We can just use the *same* formulas as without miscentering
		# correction. After all, The assumption we make (e.g. NFW tail) is on the
		# *miscentering-corrected* / *original* shear profile
		#
		# So, for NFW for example:
		#  M ~ ∫_(R < Rmax) (ΔΣ_obs + ϵ² DΔΣ_obs)|R=r/sin θ
		#     + ∫_(R > Rmax) (ΔΣ_NFW)|R=r/sin θ
		# 
		# So that's easy. We only need to be careful when *matching* the NFW tail
		# to the last measured data point that the observed data is miscentered,
		# while the NFW tail isn't. So in that matching we need to take that into
		# account.
		# That's done in `precompute(..)`.
		# Here, we don't need to worry about the tail at all.

		ΔΣ̂_integrand = (rMpc, θ) -> let
			RMpc = rMpc/sin(θ)
			ΔΣ̂val = ΔΣ̂(RMpc)
			# We need: r*f'(r/sin θ).
			# The dlog_ΔΣ̂(RMpc) thing is: (r/sin θ)*f'(r/sin θ)
			# So: r*f'(r/ sin θ) = "dlog_ΔΣ̂" * sin θ
			r_dĜ_of_r = sin(θ)*dlog_Ĝ(RMpc)
	
			# Correction from (Rmc/R)^2*4*ΔΣ(R)
			corr_0d = (Rmc²Mpc/RMpc^2)*4*ΔΣ̂val
	
			# Correction from (Rmc/R)^2*R*ΔΣ'(R)
			corr_1d = (Rmc²Mpc/rMpc^2)*(tan(θ)^2+2*sin(θ)^2)*(
				ΔΣ̂val - ΔΣ̂(rMpc)
			)
			
			# Correction from (Rmc/R)^2*R^2*ΔΣ''(R)
			corr_2d = (Rmc²Mpc/rMpc^2)*(sin(θ)+tan(θ)/cos(θ))*(
				# NOTE: I'm using Ĝ here instead of ΔΣ because that's much easier
				#       to compute in practice. Feels somewhat inconsistent with the
				#       other terms above (where I use ΔΣ), but: a) All the formulas
				#       are correct only up to O(κ) anyway, so formally these are 
				#       the same b) empirically, this actually seems to give better
				#       results in my "miscentered SIS" examples, so it probably
				#       at least doesn't hurt in practice.
				r_dĜ_of_r - dlog_Ĝ(rMpc)
			)

			ΔΣ̂val + (1/4)*(corr_0d - corr_1d - corr_2d)
		end
		
		M(rMpc) = if rMpc < rMpcTail
			θlim = asin(rMpc/rMpcTail)
			bulk = 4*(rMpc*u"Mpc")^2*u"Msun/pc^2"*quadgk(
				θ -> ΔΣ̂_integrand(rMpc, θ),
				θlim, π/2
			)[1] |> u"Msun"

			boundary = let
				RMpc = rMpcTail # = rMpc/sin(θlim)
				4*(rMpc*u"Mpc")^2*u"Msun/pc^2"* (-1/4) * (Rmc²Mpc/rMpc^2) * (
					(ΔΣ̂(RMpc) - ΔΣ̂(rMpc))*sin(θlim)^2*tan(θlim)
					# See `ΔΣ̂_integrand`. Using Ĝ here for derivative.
					# Care: This is supposed to be RMpc^-. For linear interpolation
					#       the derivative is non-continuous, so we must actually
					#       make sure to not take the wrong value.
					+ (
						sin(θlim)*dlog_Ĝ(.999999 * RMpc) - dlog_Ĝ(rMpc)
					)*sin(θlim)*tan(θlim)
				) |> u"Msun"
			end
			
			tail = calculate_M_tail(
				extrapolate, neglect_kappa, pre; θlim, f∞, rMpc, rMpcTail)
			
			bulk + boundary + tail
		else
			calculate_M_tail(
				extrapolate, neglect_kappa, pre; θlim=π/2, f∞, rMpc, rMpcTail)
		end
	end
end

# ╔═╡ df868364-b8c4-47f8-8f8f-860698b448b3
begin
	# Generic mechanism to pre-calculate something that is then passed to I/J
	# integral computation, and to M computation. Currently used to precompute
	# the parameters for the extrapolation/tail.
	
	function precompute(
		::ExtrapolatePowerDecay,
		::Union{MiscenterCorrectNone,MiscenterCorrectSmallRmcPreprocessG},
		::Union{NeglectKappa, NoNeglectKappa};
		RMpc, f∞, Gf
	)
		# Save the last Gf data point, which is the pre-factor for the 
		# G₊_tail ~ A * (Rmax/R)^n decay we assume.
		Gf(maximum(RMpc))
	end

	# Matching formula is (see `miscentering-correct-efficient-evaluation.typ`):
	#  G₊_observed (Rmax) = (G₊_tail - ϵ² D G₊_tail) (Rmax)
	# with ϵ = Rmc/Rmax and (D F)(R) ≡ (1/4) * (F(R) - R F'(R) - R² F''(R))
	# Our assumption here is G₊_tail ~ A (Rmax/R)^n after Rmax (NB: not ΔΣ_tail
	# ~ 1/R^n). This translates into
	#  G₊_observed (Rmax) = A (1 - ϵ² * (1/4) * (4 - n²))
	# Solving for A and expanding to order ϵ²
	#  A = G₊_observed(Rmax) ( 1 + ϵ² * (1/4) * (4 - n²) )
	function precompute(
		extrapolate::ExtrapolatePowerDecay,
		miscenter_correct::MiscenterCorrectSmallRmc,
		::Union{NeglectKappa, NoNeglectKappa};
		RMpc, f∞, Gf
	)
		n = extrapolate.n
		RMpcMax = maximum(RMpc)
		Rmc²Mpc = miscenter_correct.Rmc² / u"Mpc^2" |> NoUnits
		
		Gf(RMpcMax)*(1 + (1/4) * (Rmc²Mpc/RMpcMax^2)*(4-n^2))
	end
	
	function precompute(
		extrapolate::ExtrapolateNFW,
		::Union{MiscenterCorrectNone,MiscenterCorrectSmallRmcPreprocessG},
		::NoNeglectKappa;
		RMpc, f∞, Gf
	)
		RMpcMax = maximum(RMpc)
		(rs, ρs) = NFW_find_rs_ρs_from_last_Gf(
			extrapolate.cm;
			Gf_NFW_func=(R; rs, ρs) -> Gf_NFW(R; rs, ρs, f∞),
			GfTail=Gf(RMpcMax),
			Rtail=RMpcMax*u"Mpc"
		)
		(rs, ρs, f∞)
	end

	# Matching formula is (see `miscentering-correct-efficient-evaluation.typ`):
	#  G₊_observed (Rmax) = (G₊_tail - ϵ² D G₊_tail) (Rmax)
	# with ϵ = Rmc/Rmax and (D F)(R) ≡ (1/4) * (F(R) - R F'(R) - R² F''(R))	
	function precompute(
		extrapolate::ExtrapolateNFW,
		miscenter_correct::MiscenterCorrectSmallRmc,
		::NoNeglectKappa;
		RMpc, f∞, Gf
	)
		RMpcMax = maximum(RMpc)
		Rmc² = miscenter_correct.Rmc²
		(rs, ρs) = NFW_find_rs_ρs_from_last_Gf(
			extrapolate.cm;
			Gf_NFW_func=(R; rs, ρs) -> Gf_NFW_approx_miscentering_applied(
				R; Rmc², rs, ρs, f∞
			),
			GfTail=Gf(RMpcMax),
			Rtail=RMpcMax*u"Mpc"
		)
		(rs, ρs, f∞)
	end

	# TODO: Implement NFW with `NeglectKappa` (where matching to NFW is done
	#       assuming G == ΔΣ and then extrapolation also ignores κ)
end

# ╔═╡ 18dccd90-f99f-11ee-1bf6-f1ca60e4fcd0
begin
	# Non-constant f = <Σ_crit^(-1)>
	function __calculate_ΔΣ_fgeneral(
		from_ΔΣ_function, neglect_kappa::NoNeglectKappa;
		# Type omitted b/c of ForwardDiff which requires allowing Dual numbers
		G, # typeof([1.0*u"Msun/pc^2"]),
		f::typeof([1.0/u"Msun/pc^2"]),
		R::typeof([1.0*u"Mpc"]),
		interpolate::I,
		extrapolate::E,
		miscenter_correct::MC,
	) where {
		E<:AbstractExtrapolate,
		I<:AbstractInterpolate,
		MC<:AbstractMiscenterCorrect
	}
		@assert length(G) == length(f) == length(R) "G, f, R must have same length"

		# Optionally, do miscenter correction as a preprocessing step before running
		# the actual deprojection, at the level of the input G
		G = miscenter_correct_G(miscenter_correct, interpolate; R=R, G=G)
		
		@assert !any(G .* f .>= 1.0) "G*f must be < 1"
	
		RMpc = R ./ u"Mpc"
		Ĝvalues = G ./ u"Msun/pc^2"
		Gf_unchecked = get_interpolation_RMpc(interpolate; RMpc, values=G .* f)
		Ĝ = get_interpolation_RMpc(interpolate; RMpc, values=Ĝvalues)
		f̂ = get_interpolation_RMpc(interpolate; RMpc, values=f .* u"Msun/pc^2")
		
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

		f∞ = f[end]
		pre = precompute(extrapolate, miscenter_correct, neglect_kappa; RMpc, f∞, Gf)
		IR∞ = calculate_I_R∞(extrapolate, interpolate, pre; RMpc, Gf)
		JR∞ = calculate_J_R∞(extrapolate, interpolate, pre; RMpc, Gf, Ĝ, I=IR∞)

		ΔΣ(RMpc) = (u"Msun/pc^2")*(Ĝ(RMpc)/(1 - Gf(RMpc)))*(
			1 - exp(-IR∞(RMpc))*f̂(RMpc)*JR∞(RMpc)
		)

		from_ΔΣ_function(
			extrapolate, interpolate, miscenter_correct, neglect_kappa, pre;
			ΔΣ, rMpc=RMpc, f∞, Ĝvalues,
		)
	end
	function __calculate_ΔΣ_fgeneral(
		from_ΔΣ_function, neglect_kappa::NeglectKappa;
		# Type omitted b/c of ForwardDiff which requires allowing Dual numbers
		G, # typeof([1.0*u"Msun/pc^2"]),
		f::typeof([1.0/u"Msun/pc^2"]),
		R::typeof([1.0*u"Mpc"]),
		interpolate::I,
		extrapolate::E,
		miscenter_correct::MC,
	) where {
		E<:AbstractExtrapolate,
		I<:AbstractInterpolate,
		MC<:AbstractMiscenterCorrect
	}
		@assert length(G) == length(f) == length(R) "G, f, R must have same length"

		# Optionally, do miscenter correction as a preprocessing step before running
		# the actual deprojection, at the level of the input G
		G = miscenter_correct_G(miscenter_correct, interpolate; R=R, G=G)
	
		RMpc = R ./ u"Mpc"
		Ĝvalues = G ./ u"Msun/pc^2"
		Gf_unchecked = get_interpolation_RMpc(interpolate; RMpc, values=G .* f)
		Ĝ = get_interpolation_RMpc(interpolate; RMpc, values=Ĝvalues)
		
		# We neglect κ, so we don't need G*f < 1
		Gf = Gf_unchecked

		f∞ = f[end]
		pre = precompute(extrapolate, miscenter_correct, neglect_kappa; RMpc, f∞, Gf)
		# We neglect κ, so ΔΣ = G
		ΔΣ(RMpc) = (u"Msun/pc^2")*Ĝ(RMpc)

		from_ΔΣ_function(
			extrapolate, interpolate, miscenter_correct, neglect_kappa, pre;
			ΔΣ, rMpc=RMpc, f∞, Ĝvalues,
		)
	end
	
	# Constant f = <Σ_crit^(-1)>
	function __calculate_ΔΣ_fconst(
		from_ΔΣ_function, neglect_kappa::NoNeglectKappa;
		# Type omitted b/c of ForwardDiff which requires allowing Dual numbers
		G, # typeof([1.0*u"Msun/pc^2"])
		f::typeof(1.0/u"Msun/pc^2"),
		R::typeof([1.0*u"Mpc"]),
		interpolate::I,
		extrapolate::E,
		miscenter_correct::MC,
	) where {
		E<:AbstractExtrapolate,
		I<:AbstractInterpolate,
		MC<:AbstractMiscenterCorrect
	}
		@assert length(G) == length(R) "G, R must have same length"
		
		# Optionally, do miscenter correction as a preprocessing step before running
		# the actual deprojection, at the level of the input G
		G = miscenter_correct_G(miscenter_correct, interpolate; R=R, G=G)
		
		@assert !any(G .* f .>= 1.0) "G*f must be < 1"
		
		RMpc = R ./ u"Mpc"
		Gf_unchecked = get_interpolation_RMpc(interpolate; RMpc, values=G .* f)
		
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

		f∞ = f
		pre = precompute(extrapolate, miscenter_correct, neglect_kappa; RMpc, f∞, Gf)
		IR∞ = calculate_I_R∞(extrapolate, interpolate, pre; RMpc, Gf)
		ΔΣ(RMpc) = (1/f)*(Gf(RMpc)/(1 - Gf(RMpc)))*exp(-IR∞(RMpc))

		from_ΔΣ_function(
			extrapolate, interpolate, miscenter_correct, neglect_kappa, pre;
			ΔΣ, rMpc=RMpc, f∞, Ĝvalues=G ./ u"Msun/pc^2",
		)
	end
	function __calculate_ΔΣ_fconst(
		from_ΔΣ_function, neglect_kappa::NeglectKappa;
		# Type omitted b/c of ForwardDiff which requires allowing Dual numbers
		G, # typeof([1.0*u"Msun/pc^2"])
		f::typeof(1.0/u"Msun/pc^2"),
		R::typeof([1.0*u"Mpc"]),
		interpolate::I,
		extrapolate::E,
		miscenter_correct::MC,
	) where {
		E<:AbstractExtrapolate,
		I<:AbstractInterpolate,
		MC<:AbstractMiscenterCorrect
	}
		@assert length(G) == length(R) "G, R must have same length"
		
		# Optionally, do miscenter correction as a preprocessing step before running
		# the actual deprojection, at the level of the input G
		G = miscenter_correct_G(miscenter_correct, interpolate; R=R, G=G)
		
		RMpc = R ./ u"Mpc"
		Gf_unchecked = get_interpolation_RMpc(interpolate; RMpc, values=G .* f)
		# We neglect κ, so we don't need G*f < 1
		Gf = Gf_unchecked

		f∞ = f
		pre = precompute(extrapolate, miscenter_correct, neglect_kappa; RMpc, f∞, Gf)
		# We neglect κ, so ΔΣ = G
		ΔΣ(RMpc) = (1/f)*Gf(RMpc)

		from_ΔΣ_function(
			extrapolate, interpolate, miscenter_correct, neglect_kappa, pre;
			ΔΣ, rMpc=RMpc, f∞, Ĝvalues=G ./ u"Msun/pc^2",
		)
	end

	function calculate_from_ΔΣ(
		from_ΔΣ_function;
		# Type omitted b/c of ForwardDiff which requires allowing Dual numbers
		G, # typeof([1.0*u"Msun/pc^2"]),
		# No type for f since we allow both vector and scalar
		f,
		R::typeof([1.0*u"Mpc"]),
		interpolate::I,
		extrapolate::E,
		miscenter_correct::MC,
		neglect_kappa::NK,
	) where {
		E<:AbstractExtrapolate,
		I<:AbstractInterpolate,
		MC<:AbstractMiscenterCorrect,
		NK<:AbstractNeglectKappa,
	}
		__calc_ΔΣ(f::typeof([1.0/u"Msun/pc^2"])) = __calculate_ΔΣ_fgeneral(
			from_ΔΣ_function, neglect_kappa;
			G, f, R, interpolate, extrapolate, miscenter_correct
		)

		__calc_ΔΣ(f::typeof(1.0/u"Msun/pc^2")) = __calculate_ΔΣ_fconst(
			from_ΔΣ_function, neglect_kappa;
			G, f, R, interpolate, extrapolate, miscenter_correct
		)
		
		__calc_ΔΣ(f)
	end
end

# ╔═╡ 2c7ad8b1-4d4b-4117-82b2-79220746b769
function calculate_M(;
	# Type omitted b/c of ForwardDiff which requires allowing Dual numbers
	G, # typeof([1.0*u"Msun/pc^2"]),
	# No type for f since we allow both vector and scalar
	f,
	R::typeof([1.0*u"Mpc"]),
	interpolate::I,
	extrapolate::E,
	miscenter_correct::MC=MiscenterCorrectNone(),
	neglect_kappa::NK=NoNeglectKappa(),
) where {
	E<:AbstractExtrapolate,
	I<:AbstractInterpolate,
	MC<:AbstractMiscenterCorrect,
	NK<:AbstractNeglectKappa,
}
	calculate_from_ΔΣ(
		calculate_M_from_ΔΣ;
		G, f, R, interpolate, extrapolate, miscenter_correct, neglect_kappa
	)
end

# ╔═╡ 2e3d91f1-6b0f-4f5e-9761-e6a359585653
function calculate_M_and_covariance_in_bins(;
		G::typeof([1.0*u"Msun/pc^2"]),
		f,
		R::typeof([1.0*u"Mpc"]),
		G_covariance::typeof([1.0 1.0] .* u"(Msun/pc^2)^2"),
		interpolate::I,
		extrapolate::E,
		miscenter_correct::MC=MiscenterCorrectNone(),
		neglect_kappa::NK=NoNeglectKappa(),
	) where {
		E<:AbstractExtrapolate,
		I<:AbstractInterpolate,
		MC<:AbstractMiscenterCorrect,
		NK<:AbstractNeglectKappa,
	}

	__get_Rmc²(::MiscenterCorrectNone) = 0.0u"Mpc^2" # Unused dummy value
	__get_Rmc²(::MiscenterCorrectSmallRmc) = miscenter_correct.Rmc²
	__get_Rmc²(::MiscenterCorrectSmallRmcPreprocessG) = miscenter_correct.Rmc²
	
	__get_σ_Rmc²(::MiscenterCorrectNone) = 0.0u"Mpc^2" # Unused dummy value
	__get_σ_Rmc²(::MiscenterCorrectSmallRmc) = miscenter_correct.σ_Rmc²
	__get_σ_Rmc²(::MiscenterCorrectSmallRmcPreprocessG) = miscenter_correct.σ_Rmc²

	# Input for ForwardDiff -- everything in one vector and no units
	input = fill(NaN, length(G)+1)
	input[1:end-1] .= G ./ u"Msun/pc^2"
	input[end] = __get_Rmc²(miscenter_correct) / u"Mpc^2"

	# Covariance matrix matching this input
	input_cov = zeros(length(G)+1, length(G)+1)
	input_cov[1:end-1, 1:end-1] = G_covariance ./ u"(Msun/pc^2)^2"
	input_cov[end, end] = __get_σ_Rmc²(miscenter_correct)^2 / u"(Mpc^2)^2"

	RMpc = R ./ u"Mpc" .|> NoUnits

	# Forward-diff
	# - requires a single argument as input
	# - no units as input or output
	M_func = input -> let
		
		# The value of this `new_...` should be identical to `miscenter_correct`.
		# This is just to make it clear to `ForwardDiff.jl` where `Rmc²` is used.
		new_miscenter_correct = copy_with_other_Rmc²(
			miscenter_correct, input[end] .* u"Mpc^2"
		)
		
		M = calculate_M(;
			G=input[1:end-1] .* u"Msun/pc^2",
			f, R, interpolate, extrapolate, neglect_kappa,
			miscenter_correct=new_miscenter_correct, # _not_ the original one!
		)
		M.(RMpc) ./ u"Msun"
	end

	# We could just `DiffResults` to avoid calculating `value` ourselves. That is
	# also done during the jacobian calculation anyway.
	# But: I tried that and in some cases the `value` was then off. Only by <1% but
	# still. Don't like that it's off at all. So let's just call M_func(...)
	# once ourselves and lose a little perf :)
	value = M_func(input)
	jac = ForwardDiff.jacobian(M_func, input)

	# `M_func` doesn't have units. So we have to put them back ourselves.
	M = value .* u"Msun"
	# See here https://juliadiff.org/ForwardDiff.jl/stable/user/api/
	# jac[α, i] = ∂M(α)/∂x[i]
	# So the correct thing is jac * Cov * transpose(jac)
	# (and _not_ transpose(jac) * Cov * jac)
	M_stat_cov = jac * input_cov * jac' .* u"Msun^2"
	M_stat_err = sqrt.(diag(M_stat_cov))

	# When stacking multiple lenses one may want to weigh by 1/M_stat_err².
	# But: It's good to _not_ include the contributions from σ_Rmc² in that
	# M_stat_err in that case so as to ensure optimal cancellation of the shape noise
	# and also one doesn't want to negate the miscentering correction by just down-
	# -weighing everything that is miscentered.
	# In order for these to not have to calculate twice (once with and once without
	# σ_Rmc², we provide both versions here).
	M_stat_err_without_σ_Rmc² = if __get_σ_Rmc²(miscenter_correct) != 0.0u"Mpc^2"
		
		# Since this is the very last thing we do, we can over-write `input_cov`
		input_cov[end, end] = 0
		
		M_stat_cov_without_σ_Rmc² = jac * input_cov * jac' .* u"Msun^2"
		sqrt.(diag(M_stat_cov_without_σ_Rmc²))
	else
		M_stat_err
	end
		
	(
		M=M, M_stat_cov=M_stat_cov, M_stat_err=M_stat_err,
		M_stat_err_without_σ_Rmc²=M_stat_err_without_σ_Rmc²
	)
end

# ╔═╡ f4311bdf-db19-4886-93f2-51143e6845bc
md"""
# Tests
"""

# ╔═╡ f14ddc03-eb68-4029-a828-c78827482ead
md"""
## f=const and f!=const versions agree

when f is constant
"""

# ╔═╡ 9dd1c7c4-a44c-4b5c-a810-b6b171ac2569
@plutoonly let
	# Test: For f = const, both methods should agree
	M1overR = calculate_M(
		R=[.2, .5, .7] .* u"Mpc",
		G=[.3, .2, .1] .* u"Msun/pc^2",
		f=.9 ./ u"Msun/pc^2",
		extrapolate=ExtrapolatePowerDecay(1),
		interpolate=InterpolateR(1),
	)
	M1overRinterpLnR = calculate_M(
		R=[.2, .5, .7] .* u"Mpc",
		G=[.3, .2, .1] .* u"Msun/pc^2",
		f=.9 ./ u"Msun/pc^2",
		extrapolate=ExtrapolatePowerDecay(1),
		interpolate=InterpolateLnR(1),
	)
	plot(RMpc -> M1overR(RMpc)/(RMpc*u"Mpc"), .2, 1.3, label="Extrapolate 1/R, interpolateR(1)")
	plot!(RMpc -> M1overRinterpLnR(RMpc)/(RMpc*u"Mpc"), .2, 1.3, label="Extrapolate 1/R, interpolateLnR(1)")

	M1overR = calculate_M(
		R=[.2, .5, .7] .* u"Mpc",
		G=[.3, .2, .1] .* u"Msun/pc^2",
		f=[.9, .9, .9] ./ u"Msun/pc^2",
		extrapolate=ExtrapolatePowerDecay(1),
		interpolate=InterpolateR(1),
	)
	M1overRinterpLnR = calculate_M(
		R=[.2, .5, .7] .* u"Mpc",
		G=[.3, .2, .1] .* u"Msun/pc^2",
		f=[.9, .9, .9] ./ u"Msun/pc^2",
		extrapolate=ExtrapolatePowerDecay(1),
		interpolate=InterpolateLnR(1),
	)
	plot!(RMpc -> M1overR(RMpc)/(RMpc*u"Mpc"), .2, 1.3, label="(general) Extrapolate 1/R, interpolateR(1)", ls=:dash)
	plot!(RMpc -> M1overRinterpLnR(RMpc)/(RMpc*u"Mpc"), .2, 1.3, label="(general) Extrapolate 1/R, interpolateLnR(1)", ls=:dash)
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

	__get_old_Rmc²(mc::MiscenterCorrectNone) = 0.0u"Mpc^2"
	__get_old_Rmc²(mc::MiscenterCorrectSmallRmc) = mc.Rmc²
	__get_old_σ_Rmc²(mc::MiscenterCorrectNone) = 0.0u"Mpc^2"
	__get_old_σ_Rmc²(mc::MiscenterCorrectSmallRmc) = mc.σ_Rmc²

	function do_test(; f,  extrapolate, allowed_difference_factor=1.0, allowed_difference_factor_err=1.0, miscenter_correct=MiscenterCorrectNone())
		@info "Testing with" nameof(typeof(f)) extrapolate nameof(typeof(miscenter_correct))
		new_M = calculate_M_and_covariance_in_bins(;
			R=R, f=f, G=G, G_covariance=G_covariance,
			interpolate=InterpolateR(1),
			extrapolate, miscenter_correct
		)
		# Convert M to gobs to compare to old code
		new = (
			gobs=new_M.M .* u"G" ./ (R .^ 2) .|> u"m/s^2",
			gobs_stat_err=new_M.M_stat_err .* u"G" ./ (R .^ 2) .|> u"m/s^2",
			gobs_stat_cov=new_M.M_stat_cov .* u"G^2" ./ ( (R.^2) * (R.^2)') .|> u"(m/s^2)^2",
		)
		# Assert so I don't mess up R * R' vs R' * R
		@assert ( (R.^2) * (R.^2)').size == (length(R), length(R))

		Rmc² = __get_old_Rmc²(miscenter_correct)
		σ_Rmc² = __get_old_σ_Rmc²(miscenter_correct)
	
		let
			# Compare gobs
			old_gobs = old_gobs_from_ΔΣ.calculate_gobs_fast(;
				R, Rmc²,
				ΔΣ̂=G ./ u"Msun/pc^2",
				extrapolate,
			)
			max_difference = maximum(abs.((new.gobs .- old_gobs) ./ old_gobs))
			@info "Test gobs" new.gobs old_gobs max_difference
			@assert max_difference < 5e-8*allowed_difference_factor "old gobs != new gobs?!"
		end

		# Fake statcking 1 lens
		w̄l_unnormalized = ones(length(R))'
		∑ₗ_w̄l_unnormalized = ones(length(R))'
		σ²_ΔΣ̂_l = (diag(G_covariance) ./ u"(Msun/pc^2)^2")'
		∑_i_Cαi²_σ²_ΔΣ̂_l = old_gobs_from_ΔΣ.:fast_gobs_∑_i_Cαi²_σ²_ΔΣ̂(;
			Rmc², R, σ²_ΔΣ̂=σ²_ΔΣ̂_l, extrapolate
		)'
		Mpc²_∑_i_∂Cαi∂Rmc²_ΔΣ̂_l = old_gobs_from_ΔΣ.:fast_gobs_Mpc²_∑_i_∂Cαi∂Rmc²_ΔΣ̂(;
			R, ΔΣ̂=G ./ u"Msun/pc^2", extrapolate
		)'
	
		let
			# Compare gobs stat error
			old_gobs_staterr = old_gobs_from_ΔΣ.calculate_gobs_staterr_fast(;
				w̄l_unnormalized,
	 			∑ₗ_w̄l_unnormalized,
				∑_i_Cαi²_σ²_ΔΣ̂_l,
				Mpc²_∑_i_∂Cαi∂Rmc²_ΔΣ̂_l,
				σ_Rmc²,
			) .* u"m/s^2" |> vec
			max_difference = maximum(abs.((new.gobs_stat_err .- old_gobs_staterr) ./ old_gobs_staterr))
			@info "Test gobs stat err" new.gobs_stat_err old_gobs_staterr max_difference
			@assert max_difference < 5e-8*allowed_difference_factor_err "old gobs stat err != new gobs stat err?!"
		end
	
		let
			# Compare gobs covariance matrix
			old_gobs_cov = fill(NaN, length(R), length(R))
			old_gobs_from_ΔΣ.calculate_gobs_covariance_fast(;
				Rmc², σ_Rmc²,
				Mpc²_∑_i_∂Cαi∂Rmc²_ΔΣ̂_l,
				σ²_ΔΣ̂_l,
				w̄l_unnormalized,
				∑ₗ_w̄l_unnormalized,
				l_r_bin_edges=(R_bin_edges)',
				extrapolate,
				out=old_gobs_cov,
			)
			old_gobs_cov = old_gobs_cov .* u"(m/s^2)^2"
			max_difference = maximum(abs.((new.gobs_stat_cov .- old_gobs_cov) ./ old_gobs_cov))
			@info "Test gobs cov" new.gobs_stat_cov old_gobs_cov max_difference
			@assert max_difference < 1e-6*allowed_difference_factor_err "old gobs stat cov != new gobs stat cov?!"	
		end
	end

	# Make this f very small so we're in the Gf << 1 limit
	do_test(
		f = .1e-6 .* [.9, 1.5, 1.9, .9] ./ u"Msun/pc^2",
		extrapolate=ExtrapolatePowerDecay(1),
	)
	# Same test but with the code that assumes f=const
	do_test(
		f = .1e-6 .* .9 ./ u"Msun/pc^2",
		extrapolate=ExtrapolatePowerDecay(1),
	)

	# Same tests but with n=2 and n=1/2
	do_test(
		f = .1e-6 .* .9 ./ u"Msun/pc^2",
		extrapolate=ExtrapolatePowerDecay(1/2),
	)
	do_test(
		f = .1e-6 .* .9 ./ u"Msun/pc^2",
		extrapolate=ExtrapolatePowerDecay(2),
	)

	# Same test but with miscentering correction
	do_test(
		f = .1e-6 .* .9 ./ u"Msun/pc^2",
		extrapolate=ExtrapolatePowerDecay(1),
		miscenter_correct=MiscenterCorrectSmallRmc(
			Rmc²=(.16u"Mpc")^2,
			σ_Rmc²=(0.16u"Mpc")^2
		),
		allowed_difference_factor=2000,
		allowed_difference_factor_err=20000
	)

	@info "all good :)"
end

# ╔═╡ 981960ac-5f53-4175-a93d-660285acc372
md"""
## Reconstruction: Looks good out to $\sim$ Mpc

This is for $\rho \sim 1/r^2(1 + r^2)$ essentially. So SIS core, then $1/r^4$ fall off (reason: that's what Mathematica could do analytically)

Asymptotic behavior of $\Delta \Sigma$ (and thus of $Gf$) is $1/R^2$ for this profile. So at large radii, our $1/R^2$ extrapolation should give an M close to the real M. And indeed it does!

We're asssuming 10% fake errors on $G$ measurements, just so we can test the error calculation a bit.
"""

# ╔═╡ da111d64-a4f5-4637-986c-3b26027c058b
@plutoonly function test_reconstruction_SIS_quartic_fall_off(;
		logRMpc_bin_width, interpolate=InterpolateR(1)
)
	# Test: Do we actually reconstruct the correct M?

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
	M_real(R) = 4*(100*r0*u"Msun/pc^2")*r0*atan(R/r0) |> u"Msun"

	# Remove the exponential to test the constant f case
	f(R) = (1/(2u"Msun/pc^2"))*(1 + exp(-R/200u"kpc"))
	G(R) = ΔΣ_real(R)/(1 - f(R)*Σ_real(R))
	# Make up fake 10% measurement errors on G
	σ_G(R) = 0.1 * G(R)

	pGf = plot(R -> G(R)*f(R)*(R/u"Mpc")^2, .1u"Mpc", 3u"Mpc", ylabel="G * f * R^2")

	Rbins = 10 .^ (log10(.15):logRMpc_bin_width:log10(3.5)) .* u"Mpc"
	p = plot(
		RMpc -> M_real(RMpc*u"Mpc"),
		.15, 3.5, label="M real",
		xscale=:log10,
		xlabel="R in Mpc",
		ylabel="M in Msun",
	)
	p_M_ratio = plot(
		RMpc -> 1.0,
		.15, 3.5, label="",
		color=:black,
		xscale=:log10,
		xlabel="R in Mpc",
		ylabel="M reconstructed / M real",
	)
	for n in [1/2, 1, 2, 4]
		M_reconstructed = calculate_M_and_covariance_in_bins(
			R=Rbins,
			G=G.(Rbins),
			G_covariance=diagm(σ_G.(Rbins) .^ 2),
			f=f.(Rbins),
			extrapolate=ExtrapolatePowerDecay(n),
			interpolate=interpolate,
		)
		plot!(p,
			Rbins ./ u"Mpc",
			M_reconstructed.M,
			yerror=M_reconstructed.M_stat_err,
			label="M reconstructed, 1/R^$(n) extrapolate",
			marker=:diamond,
		)
		plot!(p_M_ratio,
			Rbins ./ u"Mpc",
			M_reconstructed.M ./ M_real.(Rbins),
			yerror=M_reconstructed.M_stat_err ./ M_real.(Rbins),
			label="1/R^$(n) extrapolate",
			marker=:diamond,
		)
	end

	plot(p, p_M_ratio, pGf, size=(600, 400*3), layout=(3,1), left_margin=(15, :mm))
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
	# Test: Do we actually reconstruct the correct M?

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
	M_real(R) = 4*R*(100*r0*u"Msun/pc^2") |> u"Msun"

	# Remove the exponential to test the constant f case
	f(R) = (1/(50u"Msun/pc^2"))*(1 + exp(-R/200u"kpc"))
	G(R) = ΔΣ_real(R)/(1 - f(R)*Σ_real(R))

	pGf = plot(R -> G(R)*f(R)*(R/u"Mpc")^2, .1u"Mpc", 3u"Mpc", ylabel="G * f * R^2")

	Rbins = 10 .^ (log10(.15):logRMpc_bin_width:log10(3.5)) .* u"Mpc"
	p = plot(
		RMpc -> M_real(RMpc*u"Mpc"),
		.15, 3.5, label="M real",
		xscale=:log10,
		xlabel="R in Mpc",
		ylabel="M in Msun",
	)
	p_M_ratio = plot(
		RMpc -> 1.0,
		.15, 3.5, label="",
		color=:black,
		xscale=:log10,
		xlabel="R in Mpc",
		ylabel="M reconstructed / M real",
	)
	for n in [1/2, 1, 2, 4]
		M_reconstructed = calculate_M(
			R=Rbins,
			G=G.(Rbins),
			f=f.(Rbins),
			extrapolate=ExtrapolatePowerDecay(n),
			interpolate=interpolate
		)
		plot!(p,
			Rbins ./ u"Mpc",
			M_reconstructed.(Rbins ./ u"Mpc"),
			label="M reconstructed, 1/R^$(n) extrapolate",
			marker=:diamond,
		)
		plot!(p_M_ratio,
			Rbins ./ u"Mpc",
			M_reconstructed.(Rbins ./ u"Mpc")  ./ M_real.(Rbins),
			label="1/R^$(n) extrapolate",
			marker=:diamond,
		)
	end

	plot(p, p_M_ratio, pGf, size=(600, 400*3), layout=(3,1), left_margin=(15, :mm))
end

# ╔═╡ dd9afde0-ea99-41c5-8b86-da1c91a09fc4
@plutoonly test_reconstruction_SIS(logRMpc_bin_width=.12, interpolate=InterpolateR(2))

# ╔═╡ 044926fb-bdf1-4221-905e-de2c04946709
md"""
## NFW extrapolation
"""

# ╔═╡ 81c1fde2-c7fa-456b-b720-52f0358ffa27
@plutoonly let
	# Test: NFW reconstruction

	cosmo = Cosmology.cosmology(h=.7, OmegaM=.3)
	zl = .3
	H = Cosmology.H(cosmo, zl)
	ρcrit = 3*H^2/(8π*u"G") |> u"Msun/Mpc^3"
	nfw = let
		cm = CMRelationMaccio2008(ρcrit, cosmo.h) 
		ExtrapolateNFW(cm)
	end
	
	# Choose ρ0 / rs that matches mass-concentration relation below
	M200 = 1e15u"Msun"
	# According to Maccio et al that's (Li2020 formulas with sign fixed)
	c200 = 10^(.83 - .098*log10(M200*cosmo.h/(1e12u"Msun")) )
	r200 = ( M200 / ( (4π/3) * 200 * ρcrit))^(1/3)
	rs = r200/c200
	ρ0 = M200/(4π*rs^3)/(log(1+c200) - c200/(1+c200))
	
	p = __demo.ProfileNFW(; ρ0, rs, x0=.0u"Mpc", y0=.0u"Mpc")
	Σcrit=2500u"Msun/pc^2"
	Gt = R -> __demo.calculate_azimuthally_averaged_gt(R, p; Σcritinv=1/Σcrit) * Σcrit

	# Mass we reconstruct from shear
	R = (.2:.04:10 |> collect) .* u"Mpc"
	G = Gt.(R)
	f = 1/Σcrit
	M_reconstructed = extrapolate -> calculate_M(;
		G, f, R,
		interpolate=InterpolateLnR(2),
		extrapolate=extrapolate
	).(R ./ u"Mpc")

	# Actual mass
	MNFW(r) = 4π*p.ρ0*p.rs^3*(log(1 + r/p.rs)  - 1/(1 + p.rs/r))

	power1 = ExtrapolatePowerDecay(1)
	power2 = ExtrapolatePowerDecay(2)
	plot(R,
		 M_reconstructed(power1) ./ MNFW.(R),
		 c=:gray, ls=:dash, label="reconstructed 1/R extrapolate")
	plot!(R,
		  M_reconstructed(power2) ./ MNFW.(R),
		  c=:gray, ls=:dash, label="reconstruted 1/R² extrapolate")

	# Find the actual underlying M200
	let
		check_r200 = Roots.find_zero(
			r200-> MNFW(r200)-(4π/3)*r200^3*200*ρcrit,
			(.1u"Mpc", 3u"Mpc")
		)
		check_M200 = MNFW(check_r200)
		check_c200 = check_r200/p.rs

		# Check that the MNFW profile we did actually _has_ the M200 we think it has
		# (doesn't check the mass reconstruction, just self-consistency)
		@assert abs(M200/check_M200 - 1) < 1e-15
		@assert abs(r200/check_r200 - 1) < 1e-15
		@assert abs(c200/check_c200 - 1) < 1e-15

		# Check the (rs, ρs) parameters. They _should_ be pretty good given we
		# make NFW without noise here.
		@info "underlying params" p.rs p.ρ0 M200 c200
		(check_rs, check_ρs) = NFW_find_rs_ρs_from_last_Gf(
			nfw.cm; GfTail=G[end]*f, Rtail=maximum(R),
			Gf_NFW_func=(R; rs, ρs) -> Gf_NFW(R; rs, ρs, f∞=f)
		)
		@info "matched params from last Gf data point" (check_rs, check_ρs)
		@assert abs(check_rs/p.rs - 1) < 1e-15
	end

	let
		# We should be good at "small" radii with SIS extrapolation
		sel = R .< 3u"Mpc" # Regime where extrapolation plays no role
		@assert all(abs.(M_reconstructed(power1) ./ MNFW.(R) .- 1)[sel] .< 5e-3)
		
		# We should be perfect at *all* radii with NFW extrapolation
		@assert all(abs.(M_reconstructed(nfw) ./ MNFW.(R) .- 1) .< 1e-3)

		# non-const f version (with J integral) should give ~same result
		M_nonconst_f = calculate_M(;
			G, f=ones(length(R)) .* f, R,
			interpolate=InterpolateLnR(2),
			extrapolate=nfw
		).(R ./ u"Mpc")
		@assert all(abs.(M_nonconst_f ./ M_reconstructed(nfw) .- 1) .< 1e-4)
	end
	
	plot!(R,
		  M_reconstructed(nfw) ./ MNFW.(R),
		  c=1, ls=:dash, label="reconstruted NFW extrapolate")
	
	plot!(xscale=:log10, ylabel="M_reconstructed/M_true")
end

# ╔═╡ 165969aa-7978-453c-b497-ecc763aed249
@plutoonly let
	# Test: NFW reconstruction with last data point fluctuated negative
	#       (for ease of checking: fluctuated to be exactly -1 of the NFW value)

	cosmo = Cosmology.cosmology(h=.7, OmegaM=.3)
	zl = .3
	H = Cosmology.H(cosmo, zl)
	ρcrit = 3*H^2/(8π*u"G") |> u"Msun/Mpc^3"
	nfw = let
		cm = CMRelationMaccio2008(ρcrit, cosmo.h) 
		ExtrapolateNFW(cm)
	end
	
	# Choose ρ0 / rs that matches mass-concentration relation below
	M200 = 1e15u"Msun"
	# According to Maccio et al that's (Li2020 formulas with sign fixed)
	c200 = 10^(.83 - .098*log10(M200*cosmo.h/(1e12u"Msun")) )
	r200 = ( M200 / ( (4π/3) * 200 * ρcrit))^(1/3)
	rs = r200/c200
	ρ0 = M200/(4π*rs^3)/(log(1+c200) - c200/(1+c200))
	
	p = __demo.ProfileNFW(; ρ0, rs, x0=.0u"Mpc", y0=.0u"Mpc")
	Σcrit=2500u"Msun/pc^2"
	Gt = R -> __demo.calculate_azimuthally_averaged_gt(R, p; Σcritinv=1/Σcrit) * Σcrit

	# Mass we reconstruct from shear
	R = (.2:.04:10 |> collect) .* u"Mpc"
	G = Gt.(R)
	
	# THIS IS WHERE THE NEGATIVE THING ENTERS: Give the last data point a minus
	G[end] *= -1
	
	f = 1/Σcrit
	M_reconstructed = extrapolate -> calculate_M(;
		G, f, R,
		interpolate=InterpolateLnR(2),
		extrapolate=extrapolate
	).(R ./ u"Mpc")

	# Actual mass
	MNFW(r) = 4π*p.ρ0*p.rs^3*(log(1 + r/p.rs)  - 1/(1 + p.rs/r))

	power1 = ExtrapolatePowerDecay(1)
	power2 = ExtrapolatePowerDecay(2)
	plot(R,
		 M_reconstructed(power1) ./ MNFW.(R),
		 c=:gray, ls=:dash, label="reconstructed 1/R extrapolate",
		 title="NFW reconstruction with negative last G data point"
		)
	plot!(R,
		  M_reconstructed(power2) ./ MNFW.(R),
		  c=:gray, ls=:dash, label="reconstruted 1/R² extrapolate")

	let
		# Check the (rs, ρs) parameters. They _should_ be pretty good given we
		# make NFW without noise here.
		@info "underlying params" p.rs p.ρ0 M200 c200
		(check_rs, check_ρs) = NFW_find_rs_ρs_from_last_Gf(
			nfw.cm; GfTail=G[end]*f, Rtail=maximum(R),
			Gf_NFW_func=(R; rs, ρs) -> Gf_NFW(R; rs, ρs, f∞=f)
		)
		@info "matched params from last Gf data point" (check_rs, check_ρs)
		@assert abs(check_rs/p.rs - 1) < 1e-15
		@assert abs(check_ρs/p.ρ0 - (-1)) < 1e-15
	end

	let
		# We should be good at "small" radii with SIS extrapolation
		sel = R .< 2u"Mpc" # Regime where extrapolation plays no role
		@assert all(abs.(M_reconstructed(power1) ./ MNFW.(R) .- 1)[sel] .< 1.1e-2)
		
		# We should be similarly good with NFW extarpolation (not perfect
		# b/c I made last data point negative!)
		@assert all(abs.(M_reconstructed(nfw) ./ MNFW.(R) .- 1)[sel] .< 1.1e-2)

		# _Last_ data point should be very close to minus the actual mass
		@assert all(abs.(M_reconstructed(nfw) ./ MNFW.(R) .- (-1))[end] .< 1e-11)
	end
	
	plot!(R,
		  M_reconstructed(nfw) ./ MNFW.(R),
		  c=1, ls=:dash, label="reconstructed NFW extrapolate")
	
	plot!(xscale=:log10, ylabel="M_reconstructed/M_true")
end

# ╔═╡ bf00d853-5a7e-4509-aa2b-4318dde040e1
@plutoonly let
	# Test: When reconstructing ΔΣ from G, different extrapolation choices
	#       differ by a radius-independent factor!
	# 
	# At least in the f_c = const case.
	# 
	# That's because
	# (ΔΣ|_extrapolate 1)/(ΔΣ_extrapolate) = exp(- ∫_(R_max)^∞ dR'/R' 2 [
	# 	Gf/(1-Gf)|_extrapolate1 - Gf/(1-Gf)|_extrapolate2
	# ])
	#
	# Note that only "Rmax" occurs here. Everything else *cancels exactly*!
	#
	# Because the extrapolation (the only thing that differs) enters only in the
	# exp(-∫ Gf/...) factor and nowhere else (for R < Rmax).

	# Also test: If we assume a *slower* decay for the extrapolation, that makes
	#            G *larger* at large radii (trivially).
	#            But: Relative to _other_ extrapolations, it makes the resulting
	#                 ΔΣ *smaller*. That's because of the exponential exp(-∫ Gf/...)
	#                 factor. (That applies only for ΔΣ(R) with R < Rmax. At 
	#                 larger R, the prefactor of the exponent is also extrapolated
	#                 and usually wins out).

	cosmo = Cosmology.cosmology(h=.7, OmegaM=.3)
	zl = .3
	H = Cosmology.H(cosmo, zl)
	ρcrit = 3*H^2/(8π*u"G") |> u"Msun/Mpc^3"
	nfw = let
		cm = CMRelationMaccio2008(ρcrit, cosmo.h) 
		ExtrapolateNFW(cm)
	end
	
	# Choose ρ0 / rs that matches mass-concentration relation below
	M200 = 1e15u"Msun"
	# According to Maccio et al that's (Li2020 formulas with sign fixed)
	c200 = 10^(.83 - .098*log10(M200*cosmo.h/(1e12u"Msun")) )
	r200 = ( M200 / ( (4π/3) * 200 * ρcrit))^(1/3)
	rs = r200/c200
	ρ0 = M200/(4π*rs^3)/(log(1+c200) - c200/(1+c200))
	
	p = __demo.ProfileNFW(; ρ0, rs, x0=.0u"Mpc", y0=.0u"Mpc")
	Σcrit=2500u"Msun/pc^2"
	Gt = R -> __demo.calculate_azimuthally_averaged_gt(R, p; Σcritinv=1/Σcrit) * Σcrit

	# Mass we reconstruct from shear
	R = (.2:.04:10 |> collect) .* u"Mpc"
	G = Gt.(R)
	f = 1/Σcrit
	M_reconstructed = extrapolate -> calculate_M(;
		G, f, R,
		interpolate=InterpolateLnR(2),
		extrapolate=extrapolate
	).(R ./ u"Mpc")

	# Actual mass
	MNFW(r) = 4π*p.ρ0*p.rs^3*(log(1 + r/p.rs)  - 1/(1 + p.rs/r))

	powerhalf = ExtrapolatePowerDecay(1/2)
	power1 = ExtrapolatePowerDecay(1)
	power2 = ExtrapolatePowerDecay(2)
	p1 = plot(R,
		 M_reconstructed(power1) ./ MNFW.(R),
		 c=:gray, ls=:dash, label="reconstructed 1/R extrapolate")
	plot!(R,
		  M_reconstructed(power2) ./ MNFW.(R),
		  c=:green, ls=:dash, label="reconstruted 1/R² extrapolate")
	plot!(R,
		  M_reconstructed(powerhalf) ./ MNFW.(R),
		  c=:red, ls=:dash, label="reconstruted 1/√R extrapolate")
	plot!(R,
		  M_reconstructed(nfw) ./ MNFW.(R),
		  c=:black, ls=:dash, label="reconstruted NFW extrapolate")
	
	plot!(xscale=:log10, legend_title="reconstructed mass (relative to true, NFW mass)", ylabel="M_reconstructed/M_true", ylim=(.99, 1.01))


	let
		ΔΣ_reconstructed = extrapolate -> calculate_from_ΔΣ(
			_just_get_ΔΣ;
			G, f, R, interpolate=InterpolateLnR(2), extrapolate, miscenter_correct=MiscenterCorrectNone(),
			neglect_kappa=NoNeglectKappa(),
		)
	
		function _just_get_ΔΣ(
			extrapolate,
			interpolate,
			miscenter_correct,
			neglect_kappa,
			pre;
			ΔΣ, rMpc, f∞, Ĝvalues,
		)
			ΔΣ
		end

		ΔΣ_power1 = ΔΣ_reconstructed(power1).(R ./ u"Mpc")
		ΔΣ_power2 = ΔΣ_reconstructed(power2).(R ./ u"Mpc")
		ΔΣ_powerhalf = ΔΣ_reconstructed(powerhalf).(R ./ u"Mpc")
		ΔΣ_nfw = ΔΣ_reconstructed(nfw).(R ./ u"Mpc")

		# Test: all ratios of these are the same at *all* radii
		@assert all(abs.((ΔΣ_power1 ./ ΔΣ_nfw) ./ (ΔΣ_power1[end] / ΔΣ_nfw[end]) .- 1) .< 1e-5)
		@assert all(abs.((ΔΣ_power2 ./ ΔΣ_nfw) ./ (ΔΣ_power2[end] / ΔΣ_nfw[end]) .- 1) .< 4e-5)
		@assert all(abs.((ΔΣ_powerhalf ./ ΔΣ_nfw) ./ (ΔΣ_powerhalf[end] / ΔΣ_nfw[end]) .- 1) .< 1e-5)

		# Test: *slower* decay, aka *larger* Gf leads to *smaller* ΔΣ @ R < Rmax
		# (because of the exp(-∫...) factor).
		@assert all(ΔΣ_powerhalf .< ΔΣ_power1)
		@assert all(ΔΣ_powerhalf .< ΔΣ_power2)
		@assert all(ΔΣ_power1 .< ΔΣ_power2)
		@assert all(ΔΣ_power1 .< ΔΣ_nfw) # NFW = 1/(R ln(R)) or so
		
		p2 = plot(R, ΔΣ_power1 ./ ΔΣ_nfw, label="1/R extrapolation", c=:gray)
		plot!(R, ΔΣ_power2 ./ ΔΣ_nfw, label="1/R^2 extrapolation", c=:green)
		plot!(R, ΔΣ_powerhalf ./ ΔΣ_nfw, label="1/sqrt(R) extrapolation", c=:red)
		plot!(
			legend_title="reconstructed ΔΣ relative to reconstr. ΔΣ with NFW extrap.",
			leg=(.1, .53)
		)

		plot(p1, p2, layout=(2,1), size=(600, 800))
	end
end


# ╔═╡ 072cf58f-902f-4ca8-aec2-be425f3ad547
@plutoonly let
	# Super-simple smoke test to check that ForwardDiff works with ExtrapolateNFW
	cosmo = Cosmology.cosmology(h=.7, OmegaM=.3)
	zl = .3
	H = Cosmology.H(cosmo, zl)
	ρcrit = 3*H^2/(8π*u"G") |> u"Msun/Mpc^3"
	nfw = let
		cm = CMRelationMaccio2008(ρcrit, cosmo.h) 
		ExtrapolateNFW(cm)
	end
	res = calculate_M_and_covariance_in_bins(
		R=[.2, .5, .7] .* u"Mpc",
		G=[.3, .2, .1] .* u"Msun/pc^2",
		# 10% uncertainty on G
		G_covariance=diagm((.1 .* [.3, .2, .1] .* u"Msun/pc^2") .^ 2),
		f=.9 ./ u"Msun/pc^2",
		extrapolate=nfw,
		interpolate=InterpolateLnR(1),
	)

	# Should give (very roughly) 10% uncertainty on M
	@info "Relative uncertainty" res.M_stat_err ./ res.M
	@assert abs(res.M_stat_err[end] / res.M[end] - 0.1) < 1e-2
end

# ╔═╡ f1d226a2-4bc0-4b31-a2e8-92540a9e53d5
@plutoonly function do_NFW_miscentering_test(;Σcritfactor, interpolate, do_asserts)
	# Test: Miscentering correction correctly corrects

	cosmo = Cosmology.cosmology(h=.7, OmegaM=.3)
	zl = .3
	H = Cosmology.H(cosmo, zl)
	ρcrit = 3*H^2/(8π*u"G") |> u"Msun/Mpc^3"
	nfw = let
		cm = CMRelationMaccio2008(ρcrit, cosmo.h) 
		ExtrapolateNFW(cm)
	end
	
	# Choose ρ0 / rs that matches mass-concentration relation below
	M200 = 1e15u"Msun"
	# According to Maccio et al that's (Li2020 formulas with sign fixed)
	c200 = 10^(.83 - .098*log10(M200*cosmo.h/(1e12u"Msun")) )
	r200 = ( M200 / ( (4π/3) * 200 * ρcrit))^(1/3)
	rs = r200/c200
	ρ0 = M200/(4π*rs^3)/(log(1+c200) - c200/(1+c200))
		
	p_original = __demo.ProfileNFW(
		ρ0=ρ0,
		rs=rs,
		x0=0u"Mpc", # Centered
		y0=0u"Mpc",
	)
	p_miscentered = __demo.ProfileNFW(
		ρ0=ρ0,
		rs=rs,
		x0=.16u"Mpc", # Not centered
		y0=0u"Mpc",
	)
	Σcrit = Σcritfactor*3000u"Msun/pc^2"

	R = collect(.4:.01:3.0) .* u"Mpc"

	calc_M = (p, miscenter_correct) -> let 
		gt = __demo.calculate_azimuthally_averaged_gt.(R, Ref(p); Σcritinv=1/Σcrit)
		f = 1/Σcrit
		G = gt*Σcrit
		res = calculate_M_and_covariance_in_bins(
			R=R, G=G, f=f,
			G_covariance=zeros(length(R), length(R)) .* u"(Msun/pc^2)^2",
			interpolate=interpolate,
			extrapolate=nfw,
			miscenter_correct=miscenter_correct
		)

		(res.M, res.M_stat_err)
	end

	(M_centered, _) = calc_M(p_original, MiscenterCorrectNone())
	(M_uncorrected, _) = calc_M(p_miscentered, MiscenterCorrectNone())
	(M_corrected1, M_corrected1_stat_err) = calc_M(
		p_miscentered,
		MiscenterCorrectSmallRmcPreprocessG(
			# Correct by the actual Rmc
			Rmc²=(.16u"Mpc")^2,
			# For later: use (uncertainty of Rmc^2) = Rmc^2
			σ_Rmc²=(.16u"Mpc")^2
		)
	)
	(M_corrected2, M_corrected2_stat_err) = calc_M(
		p_miscentered,
		MiscenterCorrectSmallRmc(
			# Correct by the actual Rmc
			Rmc²=(.16u"Mpc")^2,
			# For later: use (uncertainty of Rmc^2) = Rmc^2
			σ_Rmc²=(.16u"Mpc")^2
		)
	)

	let
		# Check the reconstructed rs/ρs parameters:
		# - _should_ be off when not applying miscentering corrections
		# - should _not_ be off using miscentering correction
		
		myR = R[R .< .7u"Mpc"] # Do it at Rmax=.7Mpc to exaggerate the effects
		p = p_miscentered
		@info "underlying params" p.rs p.ρ0 M200 c200
		gt = __demo.calculate_azimuthally_averaged_gt.(myR, Ref(p); Σcritinv=1/Σcrit)
		f = 1/Σcrit
		G = gt*Σcrit

		# Check: parameters are off when not applying miscentering correction
		(check_rs, check_ρs) = NFW_find_rs_ρs_from_last_Gf(
			nfw.cm; GfTail=G[end]*f, Rtail=maximum(myR),
			Gf_NFW_func=(R; rs, ρs) -> Gf_NFW(R; rs, ρs, f∞=f)
		)
		@info "matched params from last Gf data point w/o miscentering correction" (check_rs, check_ρs)
		@assert abs(check_rs/p.rs - 1) > .02 "_should_ be off w/o correction!"
		@assert abs(check_ρs/p.ρ0 - 1) > .01 "_should_ be off w/o correction!"

		# Check: parameters are _not_ off when applying miscentering correction
		(check_rs, check_ρs) = NFW_find_rs_ρs_from_last_Gf(
			nfw.cm; GfTail=G[end]*f, Rtail=maximum(myR),
			Gf_NFW_func=(R; rs, ρs) -> Gf_NFW_approx_miscentering_applied(
				R; rs, ρs, f∞=f, Rmc²=p_miscentered.x0^2
			)
		)
		@info "matched params from last Gf data point w/ miscentering correction" (check_rs, check_ρs)
		@assert abs(check_rs/p.rs - 1) < 1e-3 "should _not_ be off w correction!"
		@assert abs(check_ρs/p.ρ0 - 1) < 1e-3 "should _not_ be off w correction!"
	end	
	
	MNFW(r) = let
		p = p_original
		4π*p.ρ0*p.rs^3*(log(1 + r/p.rs)  - 1/(1 + p.rs/r))
	end
	M_true = MNFW.(R)

	do_asserts(;
		R,
		M_centered,
		M_true,
		M_uncorrected,
		M_corrected1,
		M_corrected2,
		M_corrected1_stat_err,
		M_corrected2_stat_err,
	)

	# Plots that show this visually
	p1 = plot(
		R, M_centered ./ M_true,
		label="M reconst. original / M true", color=:black,
		ylim=(:auto, 1.05)
	)
	plot!(p1, R,
		M_corrected1 ./ M_true,
		ribbon=M_corrected1_stat_err ./ M_true,
		label="M reconst. miscentered, corrected (Preprocess G) / M true", color=3, marker=:diamond, ms=2,
	)
	plot!(p1, R,
		M_corrected2 ./ M_true,
		# ribbon=M_corrected2_stat_err ./ M_true,
		label="M reconst. miscentered, corrected / M true", color=4, marker=:diamond, ms=2,
	)
	plot!(p1,
		R, M_uncorrected ./ M_true,
		label="M reconst. miscentered / M true", color=1, marker=:diamond, ms=2,
	)
	p2 = plot(
		R, M_centered ./ M_true,
		label="M reconst. original / M true", color=:black,
		ylim=(:auto, 1.05)
	)
	plot!(p2, R,
		M_corrected2 ./ M_true,
		ribbon=M_corrected2_stat_err ./ M_true,
		label="M reconst. miscentered, corrected / M true", color=4, marker=:diamond, ms=2,
	)
	plot!(p2, R,
		M_corrected1 ./ M_true,
		# ribbon=M_corrected1_stat_err ./ M_true,
		label="M reconst. miscentered, corrected (Preprocess G) / M true", color=3, marker=:diamond, ms=2,
	)
	plot!(p2,
		R, M_uncorrected ./ M_true,
		label="M reconst. miscentered / M true", color=1, marker=:diamond, ms=2,
	)
	plot(p1, p2, layout=(2,1), size=(600, 400*2))
end

# ╔═╡ 6945f330-a832-4a74-91df-a027206d536b
@plutoonly do_NFW_miscentering_test(
	Σcritfactor=1, interpolate=InterpolateLnR(2),
	do_asserts = (; R, M_centered, M_true, M_uncorrected, M_corrected1, M_corrected2, M_corrected1_stat_err, M_corrected2_stat_err) -> let
		
		# Assert some stuff
		# 1) reconstruction works for correctly centered profile
		@assert all(abs.(M_centered ./ M_true .- 1) .< 1e-6)
		# 2a) reconstruction _doesn't_ work for miscentered profile at small radii (>5%)
		sel = R .< .5u"Mpc"
		@assert all(abs.(M_uncorrected[sel] ./ M_true[sel] .- 1) .> .05)
		# 2b) at large radii it slowly gets better (naturally) (<1.5% here)
		sel = R .> 1.0u"Mpc"
		@assert all(abs.(M_uncorrected[sel] ./ M_true[sel] .- 1) .< .015)
		# 3a) Miscentering correction helps at small radii! (btter than 1.1%)
		sel = R .< .5u"Mpc"
		@assert all(abs.(M_corrected1[sel] ./ M_true[sel] .- 1) .< .011)
		@assert all(abs.(M_corrected2[sel] ./ M_true[sel] .- 1) .< .011)
		# 3b) Miscentering correction also helps at large radii (now permill!)
		sel = R .> 1.0u"Mpc"
		@assert all(abs.(M_corrected1[sel] ./ M_true[sel] .- 1) .< .001)
		@assert all(abs.(M_corrected2[sel] ./ M_true[sel] .- 1) .< .001)
	
		# 4) Linearity in Rmc^2 means: Uncertainty in M induced by Rmc^2 = M[Rmc^2-Rmc^2] - M[Rmc^2=0]
		@assert all(abs.(abs.(M_uncorrected .- M_corrected1) ./M_corrected1_stat_err .- 1) .< .01)
		@assert all(abs.(abs.(M_uncorrected .- M_corrected2) ./M_corrected2_stat_err .- 1) .< .01)

		# 5) Comparing `MiscenterCorrectSmallRmc` and `...PreprocessG` may
		# give slightly different results b/c they are equivalent only up to terms of
		# order κ(Rmc/R)^2 which can be permill stuff here
		@assert all(abs.(M_corrected1 .- M_corrected2) ./ abs.(M_corrected1) .< 8e-3)
	end
)

# ╔═╡ e3401c57-3fe6-4526-a128-387672b33863
md"""
## Miscentering correction
"""

# ╔═╡ bfd8b4e9-4b43-4720-bcc2-9263ac2d2362
@plutoonly let
	# Test: R and ln R interpolation agree with small bins

	R = (.4:.003:3.0) .* u"Mpc" |> collect
	Gfunc = R -> 300u"Msun/pc^2" / (R/u"Mpc") |> u"Msun/pc^2"
	G = Gfunc.(R)

	mc = MiscenterCorrectSmallRmcPreprocessG(1.0u"Mpc^2", 0.5u"Mpc^2")
	G_corr_R =  miscenter_correct_G(mc, InterpolateR(1), R=R, G=G)
	G_corr_lnR =  miscenter_correct_G(mc, InterpolateLnR(1), R=R, G=G)

	@assert all(abs.(G_corr_R ./ G_corr_lnR .- 1) .< .02) "R and ln R interpolations should agree well for small bins"

	Σcrit = 3000u"Msun/pc^2"
	f = 1/Σcrit
	res_interpR = calculate_M(
		R=R, G=G, f=f,
		interpolate=InterpolateR(1),
		extrapolate=ExtrapolatePowerDecay(1),
		miscenter_correct=MiscenterCorrectSmallRmc(
			Rmc²=(.16u"Mpc")^2,
			σ_Rmc²=(.16u"Mpc")^2
		)
	).(R./u"Mpc")
	res_interpLnR = calculate_M(
		R=R, G=G, f=f,
		interpolate=InterpolateLnR(1),
		extrapolate=ExtrapolatePowerDecay(1),
		miscenter_correct=MiscenterCorrectSmallRmc(
			Rmc²=(.16u"Mpc")^2,
			σ_Rmc²=(.16u"Mpc")^2
		)
	).(R./u"Mpc")

	@assert all(abs.(res_interpR ./ res_interpLnR .- 1) .< .0042) "R and ln R interpolations should agree well for small bins"
end

# ╔═╡ 5c9e62b4-f18c-4038-9e77-312ba057f057
@plutoonly let
	R = (.4:.2:3.0) .* u"Mpc" |> collect
	Gfunc = R -> 300u"Msun/pc^2" / (R/u"Mpc") |> u"Msun/pc^2"
	G = Gfunc.(R)
	Σcrit = 3000u"Msun/pc^2"
	f = 1/Σcrit
	
	# Check that the M_stat_err variant without σ_Rmc² contributions works
	res_with_cov = calculate_M_and_covariance_in_bins(;
		R, G, f,
		G_covariance=LinearAlgebra.diagm((.1 .* G).^2),
		interpolate=InterpolateR(1),
		extrapolate=ExtrapolatePowerDecay(1),
		miscenter_correct=MiscenterCorrectSmallRmc(
			Rmc²=(.16u"Mpc")^2,
			σ_Rmc²=(.16u"Mpc")^2
		)
	)
	res_with_cov_no_σ_Rmc² = calculate_M_and_covariance_in_bins(;
		R, G, f,
		G_covariance=LinearAlgebra.diagm((.1 .* G).^2),
		interpolate=InterpolateR(1),
		extrapolate=ExtrapolatePowerDecay(1),
		miscenter_correct=MiscenterCorrectSmallRmc(
			Rmc²=(.16u"Mpc")^2,
			σ_Rmc²=(.0u"Mpc")^2
		)
	)

	# Super stupid smoke checks
	@assert all(res_with_cov.M_stat_err .> 0u"Msun")
	@assert all(res_with_cov.M_stat_err_without_σ_Rmc² .> 0u"Msun")
	
	# Another smoke test: σ_Rmc² must increase the normal error
	@assert all(res_with_cov.M_stat_err .> res_with_cov_no_σ_Rmc².M_stat_err)
	
	# That's the main test: calculating with σ_Rmc² gives an M_stat_err that matches
	# the "M_stat_err_without_σ_Rmc²" of the calculation with non-zero σ_Rmc²
	@assert all(
		res_with_cov.M_stat_err_without_σ_Rmc² .== res_with_cov_no_σ_Rmc².M_stat_err
	)
end

# ╔═╡ 3a7ca7f1-39c6-4570-9a61-d69a6657b0c7
@plutoonly function do_miscentering_test(;Σcritfactor, interpolate, do_asserts)
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
	Σcrit = Σcritfactor*3000u"Msun/pc^2"

	R = collect(.4:.01:3.0) .* u"Mpc"

	calc_M = (p, miscenter_correct) -> let 
		gt = __demo.calculate_azimuthally_averaged_gt.(R, Ref(p); Σcritinv=1/Σcrit)
		f = 1/Σcrit
		G = gt*Σcrit
		res = calculate_M_and_covariance_in_bins(
			R=R, G=G, f=f,
			G_covariance=zeros(length(R), length(R)) .* u"(Msun/pc^2)^2",
			interpolate=interpolate,
			extrapolate=ExtrapolatePowerDecay(1), # SIS (not exact here!)
			miscenter_correct=miscenter_correct
		)

		(res.M, res.M_stat_err)
	end

	(M_centered, _) = calc_M(p_original, MiscenterCorrectNone())
	(M_uncorrected, _) = calc_M(p_miscentered, MiscenterCorrectNone())
	(M_corrected1, M_corrected1_stat_err) = calc_M(
		p_miscentered,
		MiscenterCorrectSmallRmcPreprocessG(
			# Correct by the actual Rmc
			Rmc²=(.16u"Mpc")^2,
			# For later: use (uncertainty of Rmc^2) = Rmc^2
			σ_Rmc²=(.16u"Mpc")^2
		)
	)
	(M_corrected2, M_corrected2_stat_err) = calc_M(
		p_miscentered,
		MiscenterCorrectSmallRmc(
			# Correct by the actual Rmc
			Rmc²=(.16u"Mpc")^2,
			# For later: use (uncertainty of Rmc^2) = Rmc^2
			σ_Rmc²=(.16u"Mpc")^2
		)
	)
	M_true = (R -> R*4e14u"Msun"/1u"Mpc" |> u"Msun").(R)

	do_asserts(;
		R,
		M_centered,
		M_true,
		M_uncorrected,
		M_corrected1,
		M_corrected2,
		M_corrected1_stat_err,
		M_corrected2_stat_err,
	)

	# Plots that show this visually
	p1 = plot(
		R, M_centered ./ M_true,
		label="M reconst. original / M true", color=:black,
		ylim=(:auto, 1.05)
	)
	plot!(p1, R,
		M_corrected1 ./ M_true,
		ribbon=M_corrected1_stat_err ./ M_true,
		label="M reconst. miscentered, corrected (Preprocess G) / M true", color=3, marker=:diamond, ms=2,
	)
	plot!(p1, R,
		M_corrected2 ./ M_true,
		# ribbon=M_corrected2_stat_err ./ M_true,
		label="M reconst. miscentered, corrected / M true", color=4, marker=:diamond, ms=2,
	)
	plot!(p1,
		R, M_uncorrected ./ M_true,
		label="M reconst. miscentered / M true", color=1, marker=:diamond, ms=2,
	)
	p2 = plot(
		R, M_centered ./ M_true,
		label="M reconst. original / M true", color=:black,
		ylim=(:auto, 1.05)
	)
	plot!(p2, R,
		M_corrected2 ./ M_true,
		ribbon=M_corrected2_stat_err ./ M_true,
		label="M reconst. miscentered, corrected / M true", color=4, marker=:diamond, ms=2,
	)
	plot!(p2, R,
		M_corrected1 ./ M_true,
		# ribbon=M_corrected1_stat_err ./ M_true,
		label="M reconst. miscentered, corrected (Preprocess G) / M true", color=3, marker=:diamond, ms=2,
	)
	plot!(p2,
		R, M_uncorrected ./ M_true,
		label="M reconst. miscentered / M true", color=1, marker=:diamond, ms=2,
	)
	plot(p1, p2, layout=(2,1), size=(600, 400*2))
end

# ╔═╡ 8f2f297f-49b9-4dba-bb52-3868019aa1ea
@plutoonly do_miscentering_test(
	Σcritfactor=1, interpolate=InterpolateR(2),
	do_asserts = (; R, M_centered, M_true, M_uncorrected, M_corrected1, M_corrected2, M_corrected1_stat_err, M_corrected2_stat_err) -> let
		# Assert some stuff
		# 1) reconstruction works for correctly centered profile (permill)
		@assert all(abs.(M_centered ./ M_true .- 1) .< .003)
		# 2a) reconstruction _doesn't_ work for miscentered profile at small radii (>5%)
		sel = R .< .5u"Mpc"
		@assert all(abs.(M_uncorrected[sel] ./ M_true[sel] .- 1) .> .05)
		# 2b) at large radii it slowly gets better (naturally) (<1.5% here)
		sel = R .> 1.0u"Mpc"
		@assert all(abs.(M_uncorrected[sel] ./ M_true[sel] .- 1) .< .015)
		# 3a) Miscentering correction helps at small radii! (btter than 1.1%)
		sel = R .< .5u"Mpc"
		@assert all(abs.(M_corrected1[sel] ./ M_true[sel] .- 1) .< .011)
		@assert all(abs.(M_corrected2[sel] ./ M_true[sel] .- 1) .< .015) # slightly worse?
		# 3b) Miscentering correction also helps at large radii (now permill!)
		sel = R .> 1.0u"Mpc"
		@assert all(abs.(M_corrected1[sel] ./ M_true[sel] .- 1) .< .003)
		@assert all(abs.(M_corrected2[sel] ./ M_true[sel] .- 1) .< .003)
	
		# 4) Linearity in Rmc^2 means: Uncertainty in M induced by Rmc^2 = M[Rmc^2-Rmc^2] - M[Rmc^2=0]
		@assert all(abs.(abs.(M_uncorrected .- M_corrected1) ./M_corrected1_stat_err .- 1) .< .01)
		@assert all(abs.(abs.(M_uncorrected .- M_corrected2) ./M_corrected2_stat_err .- 1) .< .01)

		# 5) Comparing `MiscenterCorrectSmallRmc` and `...PreprocessG` may
		# give slightly different results b/c they are equivalent only up to terms of
		# order κ(Rmc/R)^2 which can be permill stuff here
		@assert all(abs.(M_corrected1 .- M_corrected2) ./ abs.(M_corrected1) .< 8e-3)
	end
)

# ╔═╡ 5e945854-6856-4674-a41e-35672d1db672
@plutoonly do_miscentering_test(
	# This makes κ very small.
	Σcritfactor=1_000, interpolate=InterpolateR(2),
	do_asserts = (; R, M_centered, M_true, M_uncorrected, M_corrected1, M_corrected2, M_corrected1_stat_err, M_corrected2_stat_err) -> let
		# In this case, `MiscenterCorrectSmallRmc` and `...PreprocessG` 
		# should be identical (they are mathematically equivalent in this case).
		@assert all(abs.(M_corrected1 .- M_corrected2) ./ abs.(M_corrected1) .< 4e-5)
	end
)

# ╔═╡ 1e328dce-cc54-43cd-afb4-c814b4366fa5
@plutoonly do_miscentering_test(
	# Linear interpolation: Check that both are ok.
	# That's _non-trivial_ for the `MiscenterCorrectSmallRmc` b/c the divergence
	# cancellation is a little tricky with the discontinuous first derivatives!
	# 
	# NB: `MiscenterCorrectSmallRmcPreprocessG` always internally uses 
	#     qudaratic interpolation (b/c it has to calculate a 2nd order derivative).
	#     So, even for super-small κ there would probably be a small differnce
	#     between both methods. Try e.g. Σcritfactor=1000
	#
	# NB: This is quite slow. ForwardDiff.jl is super slow with the combination of
	#     many data points & linear interpolation! Presumably b/c with linear
	#     interpolation there are points where the derivatives that ForwardDiff.jl
	#     calculates are discontinuous? Not sure. But I tried a lot of things and
	#     nothing else had any effect.
	Σcritfactor=1, interpolate=InterpolateR(1),
	do_asserts = (; R, M_centered, M_true, M_uncorrected, M_corrected1, M_corrected2, M_corrected1_stat_err, M_corrected2_stat_err) -> let
		@assert all(abs.(M_corrected1 ./ M_true .- 1) .< .02)
		@assert all(abs.(M_corrected2 ./ M_true .- 1) .< .02)
	end
)

# ╔═╡ fad5d0ca-d5a8-412c-82cb-30e221ae3c38
md"""
## Neglecting kappa
"""

# ╔═╡ 0da5fb0b-d3ac-42fc-9424-0d3426b8a1ed
@plutoonly let
    calc_with_f(f, neglect_kappa) = calculate_M(;
		R=[.2, .5, .7] .* u"Mpc",
		G=[.3, .2, .1] .* u"Msun/pc^2",
		f, neglect_kappa,
		extrapolate=ExtrapolatePowerDecay(1),
		interpolate=InterpolateR(1),
	).([.2, .5, .7])

	function do_check(f)
		let
			# Test: Neglecting kappa is significant for "large" f (>20% here)
			M1 = calc_with_f(f, NoNeglectKappa())
			M2 = calc_with_f(f, NeglectKappa())
			@assert abs.(M1 ./ M2 .- 1)[begin] > .2
			# Last data point should *not* match exactly
			# (tail formulas are also different when neglecting κ)
			@assert abs.(M1 ./ M2 .- 1)[end] > .05
		end
	
		let
			# Test: Neglecting kappa is ok for "small" f
			M1 = calc_with_f(f .* 1e-10, NoNeglectKappa())
			M2 = calc_with_f(f .* 1e-10, NeglectKappa())
			@assert all(abs.(M1 ./ M2 .- 1) .< 1e-10)
		end
	end

	# Do the tests for fconst and fgeneral
	do_check(.9 ./ u"Msun/pc^2")
	do_check(.9 ./ u"Msun/pc^2" .* [1, 1, 1])
end

# ╔═╡ dfd47416-7e8c-4d7e-8646-2a6df0c7050a
md"""
## Quadgk stress tests

Checking if it can handle cancellations with integrals like

$[f(r/\sin \theta) - f(r)]/\cos^2 \theta$

which are relevant for miscentering correction (and $\rho$ reconstruction)
"""

# ╔═╡ 1659336c-d204-4372-a38a-63265a86330d
@plutoonly let
	f(R) = 1.0/R

	numeric = QuadGK.quadgk(
		th -> ( f(1.0/sin(th)) - f(1.0) ) / (cos(th)^2),
		0, π/2
	)[1]

	# Should be -1
	abs(numeric / (-1) - 1) < 1e-13 || throw("wrong result")
	numeric/(-1)
end

# ╔═╡ 14c9cf73-484a-4baa-ba31-605c0f79a0d8
@plutoonly let
	f(R) = 1.0/R
	df(R) = -1.0/R^2

	numeric = QuadGK.quadgk(
		th -> 6 * ( f(1.0/sin(th)) - f(1.0) ) / (cos(th)^4) - 1.0*df(1.0)*3/cos(th)^2,
		0, π/2
	)[1]

	# Should be -2
	abs(numeric / (-2) - 1) < 1e-8 || throw("wrong result")
	numeric/(-2)
end

# ╔═╡ 62b753d4-dd39-47a0-9c89-71162487610f
@plutoonly let
	f(R) = 1.0/R
	df(R) = -1.0/R^2

	w(θ) = -2(cos(2θ) + (-2 + cos(2θ))*1/cos(θ)^4)
	# @info w(1.32)
	numeric = QuadGK.quadgk(
		th -> (
			( f(1.0/sin(th)) - f(1.0) ) * w(th) - 1.0*df(1.0)*3/cos(th)^2
		),
		0, π/2
	)[1]

	# Should be 8/3
	abs(numeric / (8/3) - 1) < 1e-8 || throw("wrong result")
	numeric/(8/3)
end

# ╔═╡ 7da276c6-29db-423b-9306-20b7dc578261
@plutoonly let
	f(R) = 1.0/R
	df(R) = -1.0/R^2

	w(θ) = -(3/2)*(cos(2θ)-1/cos(θ)^4)
	integrand = th -> (
		( f(1.0/sin(th)) - f(1.0) ) * w(th) - 1.0*df(1.0)*(3/4)/cos(th)^2
	)
	numeric = QuadGK.quadgk(
		integrand, 0, π/2;
		atol=1e-5 # This helps since the integrand is zero
	)[1]

	# Should be 0 identically!
	abs(numeric) < 1e-9 || throw("wrong result")
	numeric
end

# ╔═╡ 1b22b358-6568-490d-9825-082740ac6361
@plutoonly let
	Rmc²Mpc = 1.0
	f(R) = 1.0/R
	df(R) = -1.0/R^2

	# Loosely inspired by actual miscentering correction
	integrand = θ -> let
		rMpc = 1.0
		RMpc = rMpc/sin(θ)
		ΔΣ̂val = f(RMpc)
		
		corr_0d = (Rmc²Mpc/RMpc^2)*ΔΣ̂val
		corr_1d_and_2d = (3/8) * (Rmc²Mpc/rMpc^2) * (
			(cos(2θ)-1/cos(θ)^4) * (ΔΣ̂val-f(rMpc))
			+ (1/2) * (1/cos(θ)^2) * rMpc*df(rMpc)
		)

		ΔΣ̂val + corr_0d + corr_1d_and_2d
	end

	numeric = QuadGK.quadgk(integrand, 0, π/2)[1]

	# It's supposed to be 5/3
	abs(numeric / (5/3) - 1) < 1e-9 || throw("wrong result")
	numeric / (5/3)
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
# ╟─fa01d0c3-f793-44a8-a406-776b77786aa9
# ╠═49397343-2023-4627-89e6-74170976c890
# ╟─42855db1-3956-429e-afe7-46d385e5148c
# ╠═2c7ad8b1-4d4b-4117-82b2-79220746b769
# ╠═2e3d91f1-6b0f-4f5e-9761-e6a359585653
# ╟─6bfbe740-2993-4ae1-ad30-54ea923e0e1c
# ╟─dfe40541-396b-485b-bcb6-d70730a24867
# ╠═c86ab391-86c3-44f8-b0b9-20fb70c4dc87
# ╠═5788ae15-34ca-4230-8d6f-52b2585123ce
# ╟─fa506a97-1c00-488d-a4d1-18b878bc3640
# ╠═6f593629-bd08-44ad-8941-54c95f131908
# ╠═0134ff7b-b627-4016-9a4b-d686207111b3
# ╠═8193ca3f-749d-4734-b2c9-9db46a0458c0
# ╠═ae4b04aa-f4a2-4060-89a6-211eb40a1808
# ╠═f42c2a3a-ac7f-45cd-84dc-8eccd147ccab
# ╟─ea9fc39e-ba29-4502-927f-d2ca77e3b4e7
# ╠═c449a9c8-1739-481f-87d5-982532c2955c
# ╟─861b3ac9-14df-462a-9aa8-40ef9a521b81
# ╠═6399685a-1e4d-41fc-a3cd-01c61bbf56cf
# ╠═df868364-b8c4-47f8-8f8f-860698b448b3
# ╠═18dccd90-f99f-11ee-1bf6-f1ca60e4fcd0
# ╟─f4311bdf-db19-4886-93f2-51143e6845bc
# ╟─f14ddc03-eb68-4029-a828-c78827482ead
# ╠═9dd1c7c4-a44c-4b5c-a810-b6b171ac2569
# ╟─2dbc3c0b-8050-448b-b836-aafc21a7f189
# ╠═2754de10-f637-46a4-ae6c-5e897206233a
# ╟─1ae70636-b3ce-4ac7-b827-e8ec615bde29
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
# ╟─044926fb-bdf1-4221-905e-de2c04946709
# ╠═81c1fde2-c7fa-456b-b720-52f0358ffa27
# ╟─165969aa-7978-453c-b497-ecc763aed249
# ╠═bf00d853-5a7e-4509-aa2b-4318dde040e1
# ╠═072cf58f-902f-4ca8-aec2-be425f3ad547
# ╠═6945f330-a832-4a74-91df-a027206d536b
# ╟─f1d226a2-4bc0-4b31-a2e8-92540a9e53d5
# ╟─e3401c57-3fe6-4526-a128-387672b33863
# ╠═bfd8b4e9-4b43-4720-bcc2-9263ac2d2362
# ╠═5c9e62b4-f18c-4038-9e77-312ba057f057
# ╠═8f2f297f-49b9-4dba-bb52-3868019aa1ea
# ╠═5e945854-6856-4674-a41e-35672d1db672
# ╠═1e328dce-cc54-43cd-afb4-c814b4366fa5
# ╠═3a7ca7f1-39c6-4570-9a61-d69a6657b0c7
# ╟─3f004698-b952-462f-8824-5c78ab1e08ad
# ╟─fad5d0ca-d5a8-412c-82cb-30e221ae3c38
# ╠═0da5fb0b-d3ac-42fc-9424-0d3426b8a1ed
# ╟─dfd47416-7e8c-4d7e-8646-2a6df0c7050a
# ╠═1659336c-d204-4372-a38a-63265a86330d
# ╠═14c9cf73-484a-4baa-ba31-605c0f79a0d8
# ╠═62b753d4-dd39-47a0-9c89-71162487610f
# ╠═7da276c6-29db-423b-9306-20b7dc578261
# ╠═1b22b358-6568-490d-9825-082740ac6361
