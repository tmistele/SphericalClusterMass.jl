### A Pluto.jl notebook ###
# v0.20.3

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
	copy_with_other_Rmc²(::MiscenterCorrectNone, Rmc²) = MiscenterCorrectNone()
	
	# Correct up to (including) O((Rmc/R)^2)
	# (but not including O(κ (Rmc/R)^2))
	# This applies the correction when calculating gobs from ΔΣ.
	# Advantage: Numerically stable with noisy data. Disadvantage: Slightly slower
	struct MiscenterCorrectSmallRmc{R} <: AbstractMiscenterCorrect 
		Rmc²::R # Not typeof(1.0u"Mpc^2") b/c of ForwardDiff
		# Note: That's the uncertainty on (Rmc^2)
		σ_Rmc²::typeof(1.0u"Mpc^2")
		ϵ::Float64
	end
	MiscenterCorrectSmallRmc(; Rmc²::typeof(1.0u"Mpc^2"), σ_Rmc², ϵ=.001) = MiscenterCorrectSmallRmc(
		Rmc²,
		σ_Rmc²,
		ϵ
	)
	copy_with_other_Rmc²(mc::MiscenterCorrectSmallRmc, Rmc²) = MiscenterCorrectSmallRmc(
		Rmc²,
		mc.σ_Rmc²,
		mc.ϵ
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
	
		A_αi(α, i) = Δθ_αi(α, i) + (1/4)*(Rmc²/R[α]^2) * b_αi(α, i)
		B_αi(α, i) = (
			- Δθ_αi(α, i)*R[i]
			+ a_αi(α, i)*R[α]
			- (1/4)*(Rmc²/R[α]^2) * b_αi(α, i) * R[i]
			+ (1/4)*(Rmc²/R[α]^2) * c_minus_dαi(α, i) * R[α]
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
	function get_interpolation_RMpc_dlog(
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
		dfunc = BSplineKit.Derivative(1) * func
		maxRMpc = maximum(RMpc)
		RMpc -> let
			@assert RMpc <= maxRMpc "Extrapolation of derivatives not implemented yet"
			RMpc*dfunc(RMpc)
		end
	end
	function get_interpolation_RMpc_dlog(
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
		dlogfunc = BSplineKit.Derivative(1) * logfunc
		maxRMpc = maximum(RMpc)
		RMpc -> let
			@assert RMpc <= maxRMpc "Extrapolation of derivatives not implemented yet"
			dlogfunc(log(RMpc))
		end
	end
end

# ╔═╡ 6bfbe740-2993-4ae1-ad30-54ea923e0e1c
md"""
## $g_{\mathrm{obs}}$ from $\Delta \Sigma$ incl. miscentering correction
"""

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
	extrapolate::ExtrapolatePowerDecay; θlim, f∞, GfTail, rMpc, rMpcTail
)
	n = extrapolate.n
	rMpcMax = rMpcTail
	Gfmax = GfTail

	if n == 1
		(4*u"G"/f∞)*Gfmax*(rMpcMax/rMpc)*(
			1
			- (1/2)*Gfmax*(rMpcMax/rMpc)*θlim
			- cos(θlim)
			+ (1/2)*Gfmax*(rMpcMax/rMpc)*cos(θlim)*sin(θlim)
		) |> u"m/s^2"
	elseif n == 2
		(4*u"G"/f∞)*(1/2)*Gfmax*(rMpcMax/rMpc)^2*(
			θlim - cos(θlim)*sin(θlim)
		) |> u"m/s^2"
	else
		# We could use that integral for the other n as well. It works well.
		# But I've implemented them already and they're faster of course, so
		# let's keep them for now.
		ΔΣtail(rMpc) = (1/f∞)*Gfmax*(rMpcMax/rMpc)^n*(1 - Gfmax*(rMpcMax/rMpc)^n)^(2/n-1)

		quadgk_result = 4*u"G*Msun/pc^2"*quadgk(
			θ -> ΔΣtail(rMpc/sin(θ))/u"Msun/pc^2" |> NoUnits,
			0, θlim
		)[1] |> u"m/s^2"
	end
end

# ╔═╡ c449a9c8-1739-481f-87d5-982532c2955c
begin
	# The tail needs to be calculated analytically. Reason: The tail goes to R -> ∞.
	# That's ok for Gf(R) and f(R) because those we just extrapolate. But it's not ok
	# for I(R) and J(R) which also enter ΔΣ(R), because those we solved numerically
	# only up to R=Rmax (and the `ODESolution` extrapolation beyond last data point
	# is often completely off).
	
	function calculate_gobs_from_ΔΣ(
		extrapolate::E,
		interpolate::I,
		::Union{MiscenterCorrectNone, MiscenterCorrectSmallRmcPreprocessG};
		ΔΣ, rMpc, f∞, Gf
	) where {E<:AbstractExtrapolate, I<:AbstractInterpolate}
		rMpcTail = maximum(rMpc)
		GfTail = Gf(rMpcTail)
	
		gobs(rMpc) = if rMpc < rMpcTail
			θlim = asin(rMpc/rMpcTail)
			numeric_integral = 4*u"G*Msun/pc^2"*quadgk(
				θ -> ΔΣ(rMpc/sin(θ))/u"Msun/pc^2" |> NoUnits,
				θlim, π/2
			)[1] |> u"m/s^2"
			analytical_tail = calculate_gobs_tail(
				extrapolate;
				θlim=θlim, f∞=f∞, GfTail=GfTail, rMpc=rMpc, rMpcTail=rMpcTail
			)
			numeric_integral + analytical_tail
		else
			calculate_gobs_tail(
				extrapolate;
				θlim=π/2, f∞=f∞, GfTail=GfTail, rMpc=rMpc, rMpcTail=rMpcTail
			)
		end
	end
	
	function calculate_gobs_from_ΔΣ(
		extrapolate::ExtrapolatePowerDecay,
		interpolate::I,
		miscenter_correct::MiscenterCorrectSmallRmc;
		ΔΣ, rMpc, f∞, Gf
	) where I <: AbstractInterpolate
		ϵ = miscenter_correct.ϵ
		@assert 0 < ϵ <= .1 "ϵ must be small"
		Rmc²Mpc = miscenter_correct.Rmc² / u"Mpc^2" |> NoUnits
		ΔΣ̂(RMpc) = ΔΣ(RMpc)/u"Msun/pc^2" |> NoUnits
		
		rMpcTail = maximum(rMpc)
		GfTail = Gf(rMpcTail)

		# Below we need the 1st derivative of ΔΣ̂.
		dlog_ΔΣ̂ = get_interpolation_RMpc_dlog(
			extrapolate, interpolate;
			RMpc=rMpc, values=ΔΣ̂.(rMpc)
		)
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
			@assert all(abs.(dlog_ΔΣ̂.(myrMpc)./myrMpc .- dlog_ΔΣ̂.(myrMpcLarger)./myrMpcLarger) .< 1e-12)
			@assert !any(abs.(dlog_ΔΣ̂.(myrMpc)./myrMpc .- dlog_ΔΣ̂.(myrMpcSmaller)./myrMpcSmaller) .< 1e-12)
		end
		__makesure_interpolation_ok(::InterpolateLnR) = let
			myrMpc = rMpc[begin:end-1]
			myrMpcLarger = 1.0001 .* rMpc[begin:end-1]
			myrMpcSmaller = .9999 .* rMpc[begin:end-1]
			@assert all(abs.(dlog_ΔΣ̂.(myrMpc) .- dlog_ΔΣ̂.(myrMpcLarger)) .< 1e-12)
			@assert !any(abs.(dlog_ΔΣ̂.(myrMpc) .- dlog_ΔΣ̂.(myrMpcSmaller)) .< 1e-12)
		end
		interpolate.order == 1 && __makesure_interpolation_ok(interpolate)
		
		# For the tail, it's easiest to do the miscentering correction directly
		# on Gf(Rmax) in the naive (no fancy integration by parts or whatever) way.
		# That's not a problem b/c the tail is analytical and so we know 1st/2nd
		# order derivatives exactly, no numerics issues.
		#
		# Note: The assumption of 1/R^n decay applies to the _corrected_ G+. But
		#       this formula is derived by assuming the _uncorrected_ G+ is 1/R^n.
		#       But: This is actually okay b/c G_+^0 = G_+ + ϵ^2 (G_+ + ...) which
		#       means that -- to order ϵ^2 -- I can also just do G_+^0 = G_+ + ϵ^2
		#       (G_+^0 + ...) (where ϵ = Rmc/R). Aka the difference between using
		#       G_+ and G_+^0 for the miscentering correction formula is of higher
		#       order in ϵ
		GfTail = GfTail + (1/4)*(Rmc²Mpc/rMpcTail^2)*GfTail*(4-extrapolate.n^2)

		ΔΣ̂_integrand = (rMpc, θ) -> let
			RMpc = rMpc/sin(θ)
			ΔΣ̂val = ΔΣ̂(RMpc)
			# We need: r*f'(r/sin θ).
			# The dlog_ΔΣ̂ thing is: (r/sin θ)*f'(r/sin θ)
			# So: r*f'(r/ sin θ) = "dlog_ΔΣ̂" * sin θ
			r_dΔΣ̂_of_r = sin(θ)*dlog_ΔΣ̂(RMpc)
	
			# Correction from (Rmc/R)^2*4*ΔΣ(R)
			corr_0d = (Rmc²Mpc/RMpc^2)*4*ΔΣ̂val
	
			# Correction from (Rmc/R)^2*R*ΔΣ'(R)
			corr_1d = (Rmc²Mpc/rMpc^2)*(tan(θ)^2+2*sin(θ)^2)*ΔΣ̂val
	
			# Correction from (Rmc/R)^2*R^2*ΔΣ''(R)
			corr_2d = (Rmc²Mpc/rMpc^2)*(sin(θ)+tan(θ)/cos(θ))*r_dΔΣ̂_of_r
			
			corr = (1/4)*(
				corr_0d
				- corr_1d
				- corr_2d
			)
	
			ΔΣ̂val+corr
		end
		
		gobs(rMpc) = if rMpc < rMpcTail
			θlim = asin(rMpc/rMpcTail)
			numeric_integral_before_subtraction = quadgk(
				θ -> ΔΣ̂_integrand(rMpc, θ),
				θlim, π/2-ϵ,
				# This is a performance optimization. We need precision better than
				# ϵ -- but not really more than that
				rtol=min(ϵ/10, 1e-4)
			)[1]
			# The "π/2-ϵ" is only necessary for the divergent miscentering correction
			# parts. For the original, purely ΔΣ̂(r/sin(θ)) part it's trivial to add
			# this back approximately. Not very important in any case since ϵ must
			# anyway be small.
			numeric_integral_before_subtraction += ϵ*ΔΣ̂(rMpc)

			numeric_integral_divergences_subtraction = let
				# Divergence from (Rmc/R)^2*4*ΔΣ(R) integrand
				subtract_d0 = 0.0
		
				# Divergence from (Rmc/R)^2*R*ΔΣ'(R) integrand
				# corr_1d = (Rmc²/r^2)*(tan(θ)^2+2*sin(θ)^2)*Ĝval
				subtract_d1 = -(Rmc²Mpc/rMpc^2)*ΔΣ̂(rMpc)/ϵ
		
				# Divergence from (Rmc/R)^2*R^2*ΔΣ''(R) integrand
				# corr_2d = (Rmc²Mpc/rMpc^2)*(tan(θ)^2+2*sin(θ)^2)*r_dĜ_of_R
				r_dΔΣ̂_of_r = dlog_ΔΣ̂(rMpc) # (not factor of sin θ!)
				subtract_d2 = -(Rmc²Mpc/rMpc^2)*r_dΔΣ̂_of_r/ϵ
				
				(1/4)*(
					subtract_d0
					- subtract_d1
					- subtract_d2
				)
			end

			numeric_integral = 4*u"G*Msun/pc^2"*(
				numeric_integral_before_subtraction +
				numeric_integral_divergences_subtraction
			) |> u"m/s^2"
			
			analytical_tail = calculate_gobs_tail(
				extrapolate;
				θlim=θlim, f∞=f∞, GfTail=GfTail, rMpc=rMpc, rMpcTail=rMpcTail
			)
			numeric_integral + analytical_tail
		else
			calculate_gobs_tail(
				extrapolate;
				θlim=π/2, f∞=f∞, GfTail=GfTail, rMpc=rMpc, rMpcTail=rMpcTail
			)
		end
	end
end

# ╔═╡ 861b3ac9-14df-462a-9aa8-40ef9a521b81
md"""
## $g_{\mathrm{obs}}$ from $G_+$
"""

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

		# Optionally, do miscenter correction as a preprocessing step before running
		# the actual deprojection, at the level of the input G
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

		calculate_gobs_from_ΔΣ(
			extrapolate, interpolate, miscenter_correct;
			ΔΣ=ΔΣ, rMpc=RMpc, f∞=f[end], Gf=Gf,
		)
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
		
		# Optionally, do miscenter correction as a preprocessing step before running
		# the actual deprojection, at the level of the input G
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

		calculate_gobs_from_ΔΣ(
			extrapolate, interpolate, miscenter_correct;
			ΔΣ=ΔΣ, rMpc=RMpc, f∞=f, Gf=Gf,
		)
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
	input_cov[end] = __get_σ_Rmc²(miscenter_correct)^2 / u"(Mpc^2)^2"

	RMpc = R ./ u"Mpc" .|> NoUnits

	__get_gobs(
		f::typeof([1.0/u"Msun/pc^2"]),
		new_miscenter_correct;
		G_no_units
	) = calculate_gobs_fgeneral(;
		G=G_no_units .* u"Msun/pc^2",
		f=f,
		R=R,
		interpolate=interpolate,
		extrapolate=extrapolate,
		miscenter_correct=new_miscenter_correct, # _not_ the original one!
	)
	__get_gobs(
		f::typeof(1.0/u"Msun/pc^2"),
		new_miscenter_correct;
		G_no_units
	) = calculate_gobs_fconst(;
		G=G_no_units .* u"Msun/pc^2",
		f=f,
		R=R,
		interpolate=interpolate,
		extrapolate=extrapolate,
		miscenter_correct=new_miscenter_correct, # _not_ the original one!
	)

	# Forward-diff
	# - requires a single argument as input
	# - no units as input or output
	gobs_func = input -> let
		
		# The value of this `new_...` should be identical to `miscenter_correct`.
		# This is just to make it clear to `ForwardDiff.jl` where `Rmc²` is used.
		new_miscenter_correct = copy_with_other_Rmc²(
			miscenter_correct, input[end] .* u"Mpc^2"
		)
		
		gobs = __get_gobs(f, new_miscenter_correct; G_no_units=input[1:end-1])
		gobs.(RMpc) ./ u"m/s^2"
	end

	# We could just `DiffResults` to avoid calculating `value` ourselves. That is
	# also done during the jacobian calculation anyway.
	# But: I tried that and in some cases the `value` was then off. Only by <1% but
	# still. Don't like that it's off at all. So let's just call gobs_fun(...)
	# once ourselves and lose a little perf :)
	value = gobs_func(input)
	jac = ForwardDiff.jacobian(gobs_func, input)

	# `gobs_func` doesn't have units. So we have to put them back ourselves.
	gobs = value .* u"m/s^2"
	# See here https://juliadiff.org/ForwardDiff.jl/stable/user/api/
	# jac[α, i] = ∂gobs(α)/∂x[i]
	# So the correct thing is jac * Cov * transpose(jac)
	# (and _not_ transpose(jac) * Cov * jac)
	gobs_stat_cov = jac * input_cov * jac' .* u"(m/s^2)^2"
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

	__get_old_Rmc²(mc::MiscenterCorrectNone) = 0.0u"Mpc^2"
	__get_old_Rmc²(mc::MiscenterCorrectSmallRmc) = mc.Rmc²
	__get_old_σ_Rmc²(mc::MiscenterCorrectNone) = 0.0u"Mpc^2"
	__get_old_σ_Rmc²(mc::MiscenterCorrectSmallRmc) = mc.σ_Rmc²

	function do_test(; f,  extrapolate, allowed_difference_factor=1.0, allowed_difference_factor_err=1.0, miscenter_correct=MiscenterCorrectNone())
		@info "Testing with" nameof(typeof(f)) extrapolate nameof(typeof(miscenter_correct))
		new = calculate_gobs_and_covariance_in_bins(;
			R=R, f=f, G=G, G_covariance=G_covariance,
			interpolate=InterpolateR(1),
			extrapolate, miscenter_correct
		)

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
			σ_Rmc²=(0.16u"Mpc")^2,
			ϵ=1e-4
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

	R = (.4:.003:3.0) .* u"Mpc" |> collect
	Gfunc = R -> 300u"Msun/pc^2" / (R/u"Mpc") |> u"Msun/pc^2"
	G = Gfunc.(R)

	mc = MiscenterCorrectSmallRmcPreprocessG(1.0u"Mpc^2", 0.5u"Mpc^2")
	G_corr_R =  miscenter_correct_G(mc, InterpolateR(1), R=R, G=G)
	G_corr_lnR =  miscenter_correct_G(mc, InterpolateLnR(1), R=R, G=G)

	@assert all(abs.(G_corr_R ./ G_corr_lnR .- 1) .< .02) "R and ln R interpolations should agree well for small bins"

	Σcrit = 3000u"Msun/pc^2"
	f = 1/Σcrit
	res_interpR = calculate_gobs_fconst(
		R=R, G=G, f=f,
		interpolate=InterpolateR(1),
		extrapolate=ExtrapolatePowerDecay(1),
		miscenter_correct=MiscenterCorrectSmallRmc(
			Rmc²=(.16u"Mpc")^2,
			σ_Rmc²=(.16u"Mpc")^2
		)
	).(R./u"Mpc")
	res_interpLnR = calculate_gobs_fconst(
		R=R, G=G, f=f,
		interpolate=InterpolateLnR(1),
		extrapolate=ExtrapolatePowerDecay(1),
		miscenter_correct=MiscenterCorrectSmallRmc(
			Rmc²=(.16u"Mpc")^2,
			σ_Rmc²=(.16u"Mpc")^2
		)
	).(R./u"Mpc")

	@assert all(abs.(res_interpR ./ res_interpLnR .- 1) .< .002) "R and ln R interpolations should agree well for small bins"
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

	calc_gobs = (p, miscenter_correct) -> let 
		gt = __demo.calculate_azimuthally_averaged_gt.(R, Ref(p); Σcritinv=1/Σcrit)
		f = 1/Σcrit
		G = gt*Σcrit
		res = calculate_gobs_and_covariance_in_bins(
			R=R, G=G, f=f,
			G_covariance=zeros(length(R), length(R)) .* u"(Msun/pc^2)^2",
			interpolate=interpolate,
			extrapolate=ExtrapolatePowerDecay(1), # SIS (not exact here!)
			miscenter_correct=miscenter_correct
		)

		(res.gobs, res.gobs_stat_err)
	end

	(gobs_centered, _) = calc_gobs(p_original, MiscenterCorrectNone())
	(gobs_uncorrected, _) = calc_gobs(p_miscentered, MiscenterCorrectNone())
	(gobs_corrected1, gobs_corrected1_stat_err) = calc_gobs(
		p_miscentered,
		MiscenterCorrectSmallRmcPreprocessG(
			# Correct by the actual Rmc
			Rmc²=(.16u"Mpc")^2,
			# For later: use (uncertainty of Rmc^2) = Rmc^2
			σ_Rmc²=(.16u"Mpc")^2
		)
	)
	(gobs_corrected2, gobs_corrected2_stat_err) = calc_gobs(
		p_miscentered,
		MiscenterCorrectSmallRmc(
			# Correct by the actual Rmc
			Rmc²=(.16u"Mpc")^2,
			# For later: use (uncertainty of Rmc^2) = Rmc^2
			σ_Rmc²=(.16u"Mpc")^2
		)
	)
	gobs_true = (R -> (u"G"/R^2)*R*4e14u"Msun"/1u"Mpc" |> u"m/s^2").(R)

	do_asserts(;
		R=R,
		gobs_centered=gobs_centered,
		gobs_true=gobs_true,
		gobs_uncorrected=gobs_uncorrected,
		gobs_corrected1=gobs_corrected1,
		gobs_corrected2=gobs_corrected2,
		gobs_corrected1_stat_err=gobs_corrected1_stat_err,
		gobs_corrected2_stat_err=gobs_corrected2_stat_err,
	)

	# Plots that show this visually
	p1 = plot(
		R, gobs_centered ./ gobs_true,
		label="M reconst. original / M true", color=:black,
		ylim=(:auto, 1.05)
	)
	plot!(p1, R,
		gobs_corrected1 ./ gobs_true,
		ribbon=gobs_corrected1_stat_err ./ gobs_true,
		label="M reconst. miscentered, corrected (Preprocess G) / M true", color=3, marker=:diamond, ms=2,
	)
	plot!(p1, R,
		gobs_corrected2 ./ gobs_true,
		# ribbon=gobs_corrected2_stat_err ./ gobs_true,
		label="M reconst. miscentered, corrected / M true", color=4, marker=:diamond, ms=2,
	)
	plot!(p1,
		R, gobs_uncorrected ./ gobs_true,
		label="M reconst. miscentered / M true", color=1, marker=:diamond, ms=2,
	)
	p2 = plot(
		R, gobs_centered ./ gobs_true,
		label="M reconst. original / M true", color=:black,
		ylim=(:auto, 1.05)
	)
	plot!(p2, R,
		gobs_corrected2 ./ gobs_true,
		ribbon=gobs_corrected2_stat_err ./ gobs_true,
		label="M reconst. miscentered, corrected / M true", color=4, marker=:diamond, ms=2,
	)
	plot!(p2, R,
		gobs_corrected1 ./ gobs_true,
		# ribbon=gobs_corrected1_stat_err ./ gobs_true,
		label="M reconst. miscentered, corrected (Preprocess G) / M true", color=3, marker=:diamond, ms=2,
	)
	plot!(p2,
		R, gobs_uncorrected ./ gobs_true,
		label="M reconst. miscentered / M true", color=1, marker=:diamond, ms=2,
	)
	plot(p1, p2, layout=(2,1), size=(600, 400*2))
end

# ╔═╡ 8f2f297f-49b9-4dba-bb52-3868019aa1ea
@plutoonly do_miscentering_test(
	Σcritfactor=1, interpolate=InterpolateR(2),
	do_asserts = (; R, gobs_centered, gobs_true, gobs_uncorrected, gobs_corrected1, gobs_corrected2, gobs_corrected1_stat_err, gobs_corrected2_stat_err) -> let

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
		@assert all(abs.(gobs_corrected1[sel] ./ gobs_true[sel] .- 1) .< .011)
		@assert all(abs.(gobs_corrected2[sel] ./ gobs_true[sel] .- 1) .< .015) # slightly worse?
		# 3b) Miscentering correction also helps at large radii (now permill!)
		sel = R .> 1.0u"Mpc"
		@assert all(abs.(gobs_corrected1[sel] ./ gobs_true[sel] .- 1) .< .003)
		@assert all(abs.(gobs_corrected2[sel] ./ gobs_true[sel] .- 1) .< .003)
	
		# 4) Linearity in Rmc^2 means: Uncertainty in gobs induced by Rmc^2 = gobs[Rmc^2-Rmc^2] - gobs[Rmc^2=0]
		@assert all(abs.(abs.(gobs_uncorrected .- gobs_corrected1) ./gobs_corrected1_stat_err .- 1) .< .01)
		@assert all(abs.(abs.(gobs_uncorrected .- gobs_corrected2) ./gobs_corrected2_stat_err .- 1) .< .01)

		# 5) Comparing `MiscenterCorrectSmallRmcPreprocessG` and `...PreprocessG` may
		# give slightly different results b/c they are equivalent only up to terms of
		# order κ(Rmc/R)^2 which can be permill stuff here
		@assert all(abs.(gobs_corrected1 .- gobs_corrected2) ./ abs.(gobs_corrected1) .< 4e-3)
	end
)

# ╔═╡ 5e945854-6856-4674-a41e-35672d1db672
@plutoonly do_miscentering_test(
	# This makes κ very small.
	Σcritfactor=1_000, interpolate=InterpolateR(2),
	do_asserts = (; R, gobs_centered, gobs_true, gobs_uncorrected, gobs_corrected1, gobs_corrected2, gobs_corrected1_stat_err, gobs_corrected2_stat_err) -> let
		# In this case, `MiscenterCorrectSmallRmcPreprocessG` and `...PreprocessG` 
		# should be identical (they are mathematically equivalent in this case).
		@assert all(abs.(gobs_corrected1 .- gobs_corrected2) ./ abs.(gobs_corrected1) .< 2e-4)
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
	do_asserts = (; R, gobs_centered, gobs_true, gobs_uncorrected, gobs_corrected1, gobs_corrected2, gobs_corrected1_stat_err, gobs_corrected2_stat_err) -> let
		@assert all(abs.(gobs_corrected1 ./ gobs_true .- 1) .< .02)
		@assert all(abs.(gobs_corrected2 ./ gobs_true .- 1) .< .02)
	end
)

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
# ╟─6bfbe740-2993-4ae1-ad30-54ea923e0e1c
# ╟─dfe40541-396b-485b-bcb6-d70730a24867
# ╠═c86ab391-86c3-44f8-b0b9-20fb70c4dc87
# ╠═c449a9c8-1739-481f-87d5-982532c2955c
# ╟─861b3ac9-14df-462a-9aa8-40ef9a521b81
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
# ╠═8f2f297f-49b9-4dba-bb52-3868019aa1ea
# ╠═5e945854-6856-4674-a41e-35672d1db672
# ╠═1e328dce-cc54-43cd-afb4-c814b4366fa5
# ╠═3a7ca7f1-39c6-4570-9a61-d69a6657b0c7
# ╟─3f004698-b952-462f-8824-5c78ab1e08ad
