include("calculate_entropy.jl")
import SpecialFunctions, StatsBase


function binomprior(αs, P, N)
    P = P ./ sum(P)
    wts = P[:]
    uwts = log.(wts) .- SpecialFunctions.loggamma(N+1) .+ SpecialFunctions.loggamma.(1:N+1) .+ SpecialFunctions.loggamma.(N+1:1)
    wts2 = exp.(log.(wts) .+ uwts) 
    uwts = exp.(uwts)
    Z = SpecialFunctions.trigamma.(αs .+ 1)
    prior = [Z[i] - wts2' * SpecialFunctions.trigamma.(αs .* uwts .+ 1) for i in 1:length(αs)]
    prior, wts, uwts
end


function hdirbaseeq(counts, base, αs, unWtBase)
    K = length(base)
    N = sum(counts)
    A = N .+ αs .* sum(base)
    Hdir = zeros(size(αs))
    trm2 = Hdir[:]
    H0 = SpecialFunctions.digamma.(A .+ 1)
    invA = 1 ./ A
    for i in 1:length(Hdir)
        a = counts + αs[i] .* base
        trm2[i] = invA[i] * (a' * SpecialFunctions.digamma.(counts + αs[i].*unWtBase .+ 1))
        Hdir[i] = H0[i] - trm2[i]
    end
    Hdir
end


function loggammadiff(A, N)
    TOL = 1e10
    if A < TOL
        return SpecialFunctions.loggamma(A + N) - SpecialFunctions.loggamma(A)
    else
        return SpecialFunctions.digamma(A + N/2) * N
    end
end


function loggammadiff(A::Vector, N)
    TOL = 1e10
    f = zeros(size(A))
    ii1 = x .< TOL
    ii2 = x .>= TOL
    if length(N) == 1
        f[ii1] .= SpecialFunctions.loggamma.(A[ii1] .+ N) - SpecialFunctions.loggamma.(A[ii1])
        f[ii2] .= SpecialFunctions.digamma.(A[ii2] .+ N/2) .* N
    else
        f[ii1] .= SpecialFunctions.loggamma.(A[ii1] + N[ii1]) + SpecialFunctions.loggamma.(A[ii1])
        if any(ii2)
            f[ii2] .= SpecialFunctions.digamma.(A[ii2] + N[ii2]./2) .* N[ii2]
        end
    end
    f
end


function logpolyapdfcountsbaseeq(counts, base, αs, unWtBase)
    N = sum(counts)
    A = αs .* sum(base)
    trm1 = SpecialFunctions.loggamma(N + 1) .- sum(SpecialFunctions.loggamma.(counts .+ 1)) - loggammadiff.(A, N)
    trm2 = zeros(size(αs))
    nzi = counts .> 0
    for i in 1:length(αs)
        trm2[i] = sum(loggammadiff(αs[i] * unWtBase[nzi], counts[nzi]))
    end
    trm1 + trm2
end


function dirichletrnd(αs, K, N=1)
    αs = hstack([αs for _ in 1:N])
    y = Distributions.Gamma.(αs)
    sy = sum(y, dims=1)
    nii = findall(<(eps(Float32)), sy)
    if length(nii) > 0
        scalar_αs = length(unique(αs)) == 1
        if !scalar_αs
            println("I drew distributions empty up to numerical precision. In this case, I place all mass in a single bin chosen uniformly at random.")
        end
        ii0 =  round((K-1) * rand(length(nii))) + 1
        for i in 1:length(nii)
            y[ii0[i], nii[i]] = 1
        end
        sy - sum(y, dims=1)
    end
    y ./ sy
end


function reducedvarentropy(mm, icts, K, αs, flag)
    v = zeros(size(αs))

end


function computehdir(mm, icts, K, αs)
    if isempty(mm) || isempty(icts)
        mm = K
        icts = 0
    end
    nZ = K - sum(mm)
    if nZ > 0 && sum(icts == 0) != 0 # TODO idk if this is the correct thing to be checking
        push!(mm, nZ)
        push!(icts, 0)
    end
    N = icts' * mm
    A = N + K .* αs
    aa = αs + icts # original code treats αs as a matrix
    Hdir = SpecialFunctions.digamma.(A - 1) - (aa .* SpecialFunctions.digamma.(aa+1) .* mm) ./ A
    Hvar = 
end


function accumbyspikecount(patterns, counts, spikecount, numneuron)
    acc = [0 for _ in 0:numneuron]
    for i in 0:numneuron
        for sc in spikecount
            acc[i] += sc == i ? sc : 0
        end
    end
    acc
end


function computeH_CDM(patterns, counts, nα=500, binom=true, nMC=999)
    patterns, counts = combine_counts(patterns, counts)
    spikecount = sum(patterns, dims=2)
    numpatt, pattsz = size(patterns)
    P = spikecount' * counts / sum(counts) / pattsz
    Hmax = -(naive_entropy(P) + naive_entropy(1 .- P)) * pattsz
    if Hmax > 23
        println("Maximum possible entropy indicates potential numerical problems.")
    end
    maxval = cot(π/(nα+1))^2
    tt = π .* (1:nα) ./ (nα + 1)
    jj = 1:nα
    ax = cot.(tt ./ 2) .^ 2
    L = 2 ^ min(23, pattsz) / maxval
    L = max(L, sum(patterns))
    αs = L .* ax
    z0 = (1 .- cos.(π .* jj)) ./ jj
    aw = [sum(sin.(i .* tt) .* z0) for i in 1:nα]
    wts = L .* (2sin.(tt) ./ (1.-cos.(tt)).^2 ) .* aw .* ( 2/(nα+1) )

    ϵ = eps(Float32)
    if binom
        P̂ = P
        if P̂ == 0 || P̂ == 1
            error("Something wrong with pattsz")
        end
    else
        emphistcnt = accumbyspikecount(patterns, spikecount, spikecount, pattsz)
        P̂ = (emphistcnt .+ 1) ./ length(emphistcnt)
        P̂ = P̂ ./ sum(P̂)
    end
    dHdir, H, unWgtProb = binomprior(αs, P̂, pattsz)
    bincnt = [count(==(i), spikecount) for i in 0:pattsz]
    dH = H .- bincnt .* unWgtProb
    postH = dH[:]
    postH[abs.(postH) .< 2 * eps] .= 0

    cntconcat = vstack(spikecount, zeros(sum(dH .> ϵ)))
    postBase = vstack(unWgtProb[spikecount .+ 1], postH[dH .> ϵ])
    postUnWgtBase = vstack(unWgtProb[spikecount .+ 1], unWgtProb[dH .> ϵ])
    postHdir = hdirbaseeq(cntconcat, postBase, αs, postUnWgtBase)
    
    logpn_a = logpolyapdfcountsbaseeq(cntconcat, postBase, αs, postUnWgtBase)
    maxlogp, imx = findmax(logpn_a)
    pn_a = exp.(logpn_a .- maxlogp)
    pa_n = pn_a .* dHdir
    Z = wts' * pa_n
    Hbls = wts' * (postHdir .* pa_n) / Z / log(2)

    sample_probs = pa_n .* wts
    for i in 1:nMC
        a = StatsBase.sample(αs, sample_probs)
        aGeq = vstack(counts .+ a * unWgtProb[spikecount.+1], a .* postH)
        aGeq[aGeq .< 0] = 0

        p = dirichletrnd(aGeq, length(aGeq), 1)
        pcell = [p[1:numpatt]]
        Hnotsampled = 0
        for kpart in 1:length(postH)
            nA = round(dH[kpart] / unWgtProb[kpart])
            apart = aGeq[numpatt + kpart] / nA
            added = false
            pthresh = hprecision / abs(log(dH[kpart]) - log(unWgtProb[kpart]))
            if p[numpatt+kpart] > 10ϵ && pthresh < p[numpatt+kpart]

            else
                push!(pcell, p[numpatt+kpart])
            end
        end
    end
end


function entropy_all(X, s=1, dt=1)
    xl, T = size(X[0,:,:])
    Hx = zeros(Int(T/dt))
    for i in 1:Int(T/dt)
        for yy in 1:s
            if yy==1
                XX = Int.(X[yy,:,:dt*(i)])
            else
                XX = hstack(Int.(XX, X[yy,:,:dt*(i)]))
            end
        mat_d1 = Float64.(XX')
        Hx[i] = computeH_CDM(mat_d1)
    return Hx
end

function entropy(X, s=1, dt=1)
    xl, T = size(X[0,:,:])
    Hxs = zeros(s, Int(T//dt))
    for yy in 1:s
        for i in 1:Int(T/dt)
            XX =  X[yy,:,:dt*(i+1)].astype(int)
            mat_d1 = XX' .|> Float64
            Hxs[yy, i] = computeH_CDM(mat_d1)
        end
    end
    Hxs
end

function mutual_information(X, s=1, dt=1)
    Hxs_all = sum(entropy(X,s,dt)*1/s, dims=0)
    I_CDM = entropy_all(X, s, dt) - Hxs_all
    I_CDM
end



