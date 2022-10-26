import Pkg; Pkg.add("JLD");Pkg.add("LinearAlgebra"), Pkg.add("DelimitedFiles"), Pkg.add("IterativeSolvers"), Pkg.add("Roots")
using JLD,LinearAlgebra, DelimitedFiles, IterativeSolvers, Roots
#SIMULATED DATA 
using JLD
function simulate_data(n,vrank,m, sigma1)

    #X = hcat(zeros(n), gender)
    X = zeros(n,m)
    Z = Array{Matrix{Float64}}(undef, m-1)
    Zall = zeros(n,vrank*(m-1))
    # two variance components: additive and environment
    V2 = Array{Matrix{Float64}}(undef, m)

    for i = 2:m
         #r = randn(n,vrank)
         #r =  r/(norm(r))^(1/4)
         #Z[i-1] = r
         #mat = r*r'
         #V2[i] =  mat
         r = randn(n,vrank)
         V2[i] =  r*r'
         scale = norm(V2[i],2)
         V2[i] =  V2[i]/scale
         Z[i-1] = r/sqrt(scale)
    end

    Zall = reduce(hcat, Z)
    V2[1] = I(n)
    V = V2
    β = randn(m,1) # coef of sex effect


    σ2 = (randn(m,1).+1).^2  # variance of kinsip(additive), delta7(dominance),
        # H(householder) and error(environment)

    σ2[1] = sigma1

    Ω = zeros(n, n)
    nmethod = 4
    nsim = 50
    vcratio = [0 0.05 0.1 1 10 20]
    nratio = length(vcratio)
    mendelresult = zeros(6, nmethod, nratio)
    mse_β = zeros(2, nmethod, m, nratio)
    mse_σ2 = zeros(2, nmethod, m, nratio)
    obj = zeros(nsim, nmethod)
    t = zeros(nsim, nmethod)
    iter = ones(Int, nsim, nmethod)
    σ2sqerr = zeros(nmethod, m, nsim)
    βsqerr = zeros(nmethod, m, nsim)


    Ω = zeros(n, n)
    for i = 1:m
        Ω += σ2[i] * V[i]
    end

    y = X * β + cholesky(Symmetric(Ω)).L * randn(n)
    y = vec(y)
    σ2 = vec(σ2)

    p    = length(Z) # no. non-identity variance components  
    qi   = [size(Z[i], 2) for i in 1:p] #rank of Z's
    qidx = [[1:qi[1]]; [(sum(qi[1:i-1])+1):sum(qi[1:i]) for i in 2:p]]
    return p,qi,qidx, X, Z, Zall,V,β ,σ2,Ω,y
end

function vc(
    y::AbstractVector{T}, 
    X::AbstractMatrix{T},
    V::Vector{<:AbstractMatrix{T}};
    verbose::Bool = false, 
    maxiter::Integer = 5000, 
    funtol::Number = 1e-8,
    stop_comp::Bool = true,
    out_comp::Number = 1,
    β::Vector{T} = X \ y,
    σ2::Vector{T} = ones(T, length(V)),
    algo::Symbol = :MM
    ) where T <: LinearAlgebra.BlasFloat

    n, p = size(X)
    m    = length(V)   # no. variance components
    # pre-allocate working arrays
    storagen  = zeros(T, n)
    storagenp = zeros(T, n, p)
    storagepp = zeros(T, p, p)
    storagep  = zeros(T, p)
    if algo == :EM
        zrk = [rank(V[i]) for i in 1:m]
    elseif algo == :FS
        z     = log.(σ2) # z is log transformed variables
        znew  = similar(z)
        σ2new = similar(σ2)
        gradz = zeros(T, m)
        infoz = zeros(T, m, m)
        Δz    = zeros(T, m)    
    end
    # initialize algorithm
    Ω = zeros(T, n, n)
    update_Ω!(Ω, σ2, V)
    Ωchol = cholesky(Ω)
    Ωinv  = inv(Ωchol)
    res   = y - X * β
    copy!(storagen, res)
    ldiv!(Ωchol.L, storagen)
    loglConst = - n * log(2π)
    logl = (loglConst - logdet(Ωchol) - norm(storagen)^2) / 2
    verbose && println("iter=", 0," logl=", logl)
    verbose && println("β=", β," σ2=", σ2)
    niter = maxiter
    # MM/EM/FS loop
    for iter in 1:maxiter
        # update variance components
        ldiv!(transpose(Ωchol.L), storagen) # stroagen is inv(Ω) * residual
        if algo == :MM || algo == :EM
            for j in 1:m
                qf = dot(V[j] * storagen, storagen)
                tr = dot(Ωinv, V[j])
                if algo == :MM
                    σ2[j] *= sqrt(qf / tr)
                elseif algo == :EM
                    σ2[j] += (qf - tr) / zrk[j] * σ2[j]^2
                end
            end
            # update Ω
            Ω = zeros(n, n)
            for i = 1:m
                Ω += σ2[i] * V[i]
            end
            Ωchol = cholesky(Ω)
        elseif algo == :FS
            for j in 1:m
                gradz[j] = σ2[j] * (dot(V[j] * storagen, storagen) - dot(Ωinv, V[j])) / 2
                for k in j:m
                    infoz[k, j] = σ2[j] * σ2[k] * tr(Ωinv * V[j] * Ωinv * V[k]) / 2
                    infoz[j, k] = infoz[k, j]
                end
            end
            # make sure the information matrix is positive definite
            if eigmin(infoz) ≤ 1e-3; infoz += T(1e-3)I; end
            # line search
            Δz = cholesky(infoz)\gradz
            #A_ldiv_B!(Δz, cholesky(infoz), gradz) # Newton direction
            stepsize = T(1)
            for btiter in 1:50
                znew  .= z .+ stepsize .* Δz
                σ2new .= exp.(znew)
                update_Ω!(Ω, σ2new, V)
                if eigmin(Ω) > 0
                    Ωchol = cholesky(Ω)
                else
                    stepsize /= 2
                    continue
                end
                storagen = Ωchol\res
                #A_ldiv_B!(storagen, Ωchol, res)
                loglnew = (loglConst - logdet(Ωchol) - dot(res, storagen)) / 2
                if loglnew > logl
                    break
                elseif btiter == 50
                    warn("line search failed")
                else
                    stepsize /= 2
                end                
            end  # end of line search
            copy!(z, znew)
            copy!(σ2, σ2new)    
        end
        Ωinv = inv(Ωchol)
        # update fixed effects
        mul!(storagenp, Ωinv, X)
        mul!(storagepp, transpose(X), storagenp)
        mul!(storagen, Ωinv, y)
        mul!(storagep, transpose(X), storagen)
        copy!(β, storagep)
        if !isposdef(storagepp)
            storagepp += T(1e-3)I
        end
        ldiv!(cholesky(Symmetric(storagepp)), β)
        mul!(res, X, β)
        res .= y .- res
        copy!(storagen, res)
        ldiv!(Ωchol.L, storagen)
        # check convergence
        loglold = logl
        logl = (loglConst - logdet(Ωchol) - norm(storagen)^2) / 2
        verbose && println("iter=", iter," logl=", logl)
        #verbose && println("β=", β," σ2=", σ2)
        if (abs(logl - loglold) < funtol * (abs(logl) + 1)) & stop_comp
            niter = iter
            break
        end
        if (logl - out_comp > 0) &(abs(logl - out_comp) < 10^-1) & (~stop_comp)
            niter = iter
            break
        end
    end
    # output
    return β, σ2, niter, logl
end

function update_Ω!(
    Ω::AbstractMatrix{T},
    σ2::AbstractVector{T},
    V::Vector{<:AbstractMatrix{T}}
    ) where T <: AbstractFloat
    fill!(Ω, 0)
    for j in 1:length(σ2)
        Ω .+= σ2[j] .* V[j]
    end
    Ω
end

#function update_Ω!(
#    Ω::AbstractMatrix{T},
#    σ2::AbstractVector{T},
#    V::Vector{<:AbstractMatrix{T}}
#    ) where T <: LinearAlgebra.BlasFloat
#    fill!(Ω, 0)
#    for j in 1:length(σ2)
#        BLAS.axpy!(σ2[j], V[j], Ω)
#    end
#    Ω
#end


function lmm_woodbury(
    y::AbstractVector{T}, 
    X::AbstractMatrix{T},
    Z::Vector{<:AbstractMatrix{T}};
    verbose::Bool = false, 
    maxiter::Integer = 5000, 
    funtol::Number = 1e-8,
    stop_comp::Bool = true,
    out_comp::Number = 1,
    β::Vector{T} = X \ y,
    σ2::Vector{T} = ones(T, length(V)),
    algo::Symbol = :MM
    ) where T <: LinearAlgebra.BlasFloat
    
    # dimensions
    n, p = size(X)
    m    = length(Z) # no. non-identity variance components  
    qi   = [size(Z[i], 2) for i in 1:m] #rank of Z's
    qidx = [[1:qi[1]]; [(sum(qi[1:i-1])+1):sum(qi[1:i]) for i in 2:m]] # arrays of rank indexs
    q    = sum(qi) # sum of all ranks 
    # pre-compute fixed quantities
    ztx = [Z[i]' * X    for i in 1:m]           # ztx[i] = Z[i]' * X
    zty = [Z[i]' * y    for i in 1:m]
    ztz = [Z[i]' * Z[j] for i in 1:m, j in 1:m] # ztz[i, j] = Z[i]'Z[j]
    zi2 = [norm(Z[i])^2 for i in 1:m]
    xtx = X'X
    xty = X'y
    # pre-allocate working arrays
    if algo == :EM
        zrk = [rank(Z[i]) for i in 1:m]
    elseif algo == :FS
        z     = log.(σ2) # z is log transformed variables
        znew  = similar(z)
        Δz    = similar(z)
        ztoiz = deepcopy(ztz)
        σnew  = similar(σ2)
        gradz = zeros(T, m + 1)
        infoz = zeros(T, m + 1, m + 1)
        storagez = deepcopy(Z)
    end
    storagen  = zeros(T, n)    # n-by-1 vector
    σztr      = zeros(T, q)    # q-by-1 vector
    storageq  = zeros(T, q)    # q-by-1 vector
    storagep  = zeros(T, p)    # p-by-1 vector
    M         = zeros(T, q, q) # q-by-q matrix
    storageqq = zeros(T, q, q) # q-by-q matrix
    storageqp = zeros(T, q, p) # q-by-p matrix
    storagepp = zeros(T, p, p) # p-by-p matrix
    # pre-allocate views/pointers
    storageqi   = [view(storageq , qidx[i])    for i in 1:m]
    storageqpi  = [view(storageqp, qidx[i], :) for i in 1:m]
    storageqqi  = [view(storageqq, qidx[i], :) for i in 1:m]
    storageqqj  = [view(storageqq, :, qidx[j]) for j in 1:m]
    storageqqij = [view(storageqq, qidx[i], qidx[j]) for i in 1:m, j in 1:m]
    # calculate initial objective value
    σ = sqrt.(σ2)
    res = y - X * β
    update_σztr!(σztr, Z, res, σ, qidx)
    update_woodbury!(storageqp, storageq, storageqq, M, ztx, zty, ztz, σ, qidx)
    Mchol = cholesky(M)
    ldiv!(Mchol.L, σztr)
    loglconst = - n * log(2π)
    logl = (loglconst - - (n - q) * log(σ2[end]) - logdet(Mchol) - 
        (norm(res)^2 - norm(σztr)^2) / σ2[end]) / 2
    verbose && println("iter=", 0, " logl=", logl)
    verbose && println("β=", β," σ2=", σ2)
    # MM/EM/FS loop
    niter = maxiter
    for iter in 1:maxiter
        # storagen = Ω^{-1} * res
        copy!(storageq, σztr)
        ldiv!(transpose(Mchol.L), storageq) # storageq = inv(L) * σztr
        copy!(storagen, res)
        for i in 1:m
            BLAS.gemv!('N', -σ[i], Z[i], storageqi[i], T(1), storagen)
        end
        # storageqq = inv(L) * diag(σ) * Z' * Z
        ldiv!(Mchol.L, storageqq)
        if algo == :MM || algo == :EM
            # update variance components σ1,..., σm
            for i in 1:m
                mul!(storageqi[i], transpose(Z[i]), storagen)
                qf = (norm(storageqi[i]) / σ2[end])^2 # quadratic form
                tr = (zi2[i] - norm(storageqqj[i])^2) / σ2[end] # trace term
                if algo == :MM
                    σ2[i] *= sqrt(qf / tr)
                elseif algo == :EM
                    σ2[i] += (qf - tr) / zrk[i] * σ2[i]^2
                end
            end
            # update σ0
            qf = (norm(storagen) / σ2[end])^2
            ldiv!(transpose(Mchol.L), storageqq) # storageqq is inv(M) * diag(σ) * Z' * Z
            tr = T(n)
            for i in 1:m
                for j in qidx[i]
                    tr -= σ[i] * storageqq[j, j]
                end
            end
            tr /= σ2[end]
            if algo == :MM
                σ2[end] *= sqrt(qf / tr)
            elseif algo == :EM
                σ2[end] += (qf - tr) / n * σ2[end]^2
            end
            σ .= sqrt.(σ2)
            # update M and other quantities
            update_woodbury!(storageqp, storageq, storageqq, M, ztx, zty, ztz, σ, qidx)
            Mchol = cholesky(M)
##############################################################
        elseif algo == :FS
            # gradz and infoz
            for j in 1:m
                mul!(storageqi[j], transpose(Z[j]), storagen)
                qf = (norm(storageqi[j]) / σ2[end])^2 # quadratic form
                tr = (zi2[j] - norm(storageqqj[j])^2) / σ2[end] # trace term
                gradz[j] = σ2[j] * (qf - tr) / 2
                for i in j:m
                    copy!(ztoiz[i, j], ztz[i, j])
                    BLAS.gemm!('T', 'N', T(-1), storageqqj[i], storageqqj[j], T(1), ztoiz[i, j])
                    infoz[i, j] = σ2[i] * σ2[j] * norm(ztoiz[i, j])^2 / 2 / σ2[end]^2
                    infoz[j, i] = infoz[i, j]
                end
            end
            # last entry of gradz
            qf = (norm(storagen) / σ2[end])^2
            ldiv!(transpose(Mchol.L), storageqq) # storageqq is inv(M) * diag(σ) * Z' * Z
            tr = T(n)
            for i in 1:m, j in qidx[i]
                tr -= σ[i] * storageqq[j, j]
            end
            tr /= σ2[end]
            gradz[m + 1] = σ2[end] * (qf - tr) / 2
            # last column/row of infoz
            for j in 1:m
                copy!(storagez[j], Z[j])
                for i in 1:m
                    BLAS.gemm!('N', 'N', -σ[i], Z[i], storageqqij[i, j], T(1), storagez[j])
                end
                infoz[m + 1, j] = σ2[j] * norm(storagez[j])^2 / σ2[end] / 2
                infoz[j, m + 1] = infoz[m + 1, j]
            end
            infoz[m + 1, m + 1] = T(n)
            for j in 1:m, i in qidx[j]
                infoz[m + 1, m + 1] -= 2 * σ[j] * storageqq[i, i]
                for k in 1:m, l in qidx[k]
                    infoz[m + 1, m + 1] += σ[j] * σ[k] * storageqq[i, l] * storageqq[l, i]
                end
            end
            infoz[m + 1, m + 1] /= 2
            # make sure the information matrix is positive definite
            if eigmin(infoz) ≤ 1e-3; infoz += T(1e-3)I; end
            # line search
            ldiv!(Δz, cholesky!(infoz), gradz) # Newton direction
            stepsize = T(1)
            for btiter in 1:50
                znew .= z .+ stepsize .* Δz
                σnew .= exp.(znew ./ 2)
                update_woodbury!(storageqp, storageq, storageqq, M, ztx, zty, ztz, σnew, qidx)
                if eigmin(M) > 0
                    Mchol = cholesky!(M)
                else
                    stepsize /= 2
                    continue
                end
                update_σztr!(σztr, Z, res, σnew, qidx)
                ldiv!(Mchol.L, σztr)
                loglnew = (loglconst - 2(n - q) * log(σnew[end]) - logdet(Mchol) - 
                    (norm(res)^2 - norm(σztr)^2) / σnew[end]^2) / 2
                if loglnew > logl
                    break
                elseif btiter == 50
                    warn("line search failed")
                else
                    stepsize /= 2
                end
            end  # end of line search
            copy!(z, znew)
            copy!(σ, σnew)
            σ2 .= σ .* σ
        end
##############################################################
        # update fixed effects
        ldiv!(Mchol.L, storageqp) # storageqp is inv(L) * σztx 
        mul!(storagepp, transpose(storageqp), storageqp) 
        storagepp .= xtx .- storagepp # Gram matrix
        if !isposdef(storagepp)
            storagepp += 1e-3I
        end
        ldiv!(Mchol.L, storageq) # storageq is inv(L) * σzty 
        copy!(β, xty)
        BLAS.gemv!('T', T(-1), storageqp, storageq, T(1), β) # rhs
        ldiv!(cholesky!(Symmetric(storagepp)), β)
        # update residuals
        copy!(res, y)
        BLAS.gemv!('N', T(-1), X, β, T(1), res)
        update_σztr!(σztr, Z, res, σ, qidx)
        ldiv!(Mchol.L, σztr)
        # check convergence
        loglold = logl
        logl = (loglconst - (n - q) * log(σ2[end]) - logdet(Mchol) - 
            (norm(res)^2 - norm(σztr)^2) / σ2[end]) / 2
        verbose && println("iter=", iter, " logl=", logl)
        #verbose && println("β=", β," σ2=", σ2)
        if (abs(logl - loglold) < funtol * (abs(logl) + 1)) & stop_comp
            niter = iter
            break
        end
        if (logl - out_comp > 0) && (abs(logl - out_comp) < 10^-1) && (~stop_comp)
            niter = iter
            break
        end
    end
    # output
    return β, σ2, niter, logl
end

"""
    update_woodbury!()

Update quantities in Woodbury formula.
"""
function update_woodbury!(
    σztx::Matrix{T},
    σzty::Vector{T},
    σztz::Matrix{T},
    M::Matrix{T},
    ztx::Vector{<:AbstractMatrix{T}},
    zty::Vector{<:AbstractVector{T}},
    ztz::Matrix{<:AbstractMatrix{T}},
    σ::Vector{T},
    qidx::Vector{<:UnitRange}
    ) where T <: AbstractFloat
    m = length(qidx)
    for j in 1:m
        σztx[qidx[j], :] = σ[j] * ztx[j]
        σzty[qidx[j]] = σ[j] * zty[j]
        for i in 1:m
            σztz[qidx[i], qidx[j]] = σ[i] * ztz[i, j]
            M[qidx[i], qidx[j]]    = σ[i] * σ[j] * ztz[i, j]
        end
    end
    σ02 = σ[end] * σ[end]
    for i in 1:size(M, 1)
        M[i, i] += σ02
    end
    σztx, σzty, σztz, M
end

function update_σztr!(
    σztr::Vector{T},
    Z::Vector{<:AbstractMatrix{T}},
    r::Vector{T},
    σ::Vector{T},
    qidx::Vector{<:UnitRange}
    ) where T <: AbstractFloat
    m = length(qidx)
    for i in 1:m
        @views mul!(σztr[qidx[i]], transpose(Z[i]), r)
        σztr[qidx[i]] .*= σ[i]
    end
    σztr
end


function vc_cd(
    y::AbstractVector{T}, 
    X::AbstractMatrix{T},
    V::Vector{<:AbstractMatrix{T}};
    verbose::Bool = false, 
    maxiter::Integer = 5000, 
    funtol::Number = 1e-8,
    β::Vector{T} = X \ y,
    σ2::Vector{T} = ones(T, length(V)),
    algo::Symbol = :MM,
    low_rank::Bool = false,
    immediate_update::Bool = false
    ) where T <: LinearAlgebra.BlasFloat

    n, p = size(X)
    m    = length(V)   # no. variance components
    # pre-allocate working arrays
    storagen  = zeros(T, n)
    storagenp = zeros(T, n, p)
    storagepp = zeros(T, p, p)
    storagep  = zeros(T, p)
    if algo == :EM
        zrk = [rank(V[i]) for i in 1:m]
    elseif algo == :FS
        z     = log.(σ2) # z is log transformed variables
        znew  = similar(z)
        σ2new = similar(σ2)
        gradz = zeros(T, m)
        infoz = zeros(T, m, m)
        Δz    = zeros(T, m)    
    end
    # initialize algorithm
    Ω = zeros(T, n, n)
    update_Ω!(Ω, σ2, V)
    Ωchol = cholesky(Ω)
    Ωinv  = inv(Ωchol)
    res   = y - X * β
    copy!(storagen, res)
    ldiv!(Ωchol.L, storagen)
    loglConst = - n * log(2π)
    logl = (loglConst - logdet(Ωchol) - norm(storagen)^2) / 2
    verbose && println("iter=", 0," logl=", logl)
    verbose && println("β=", β," σ2=", σ2)
    niter = maxiter
    # MM/EM/FS loop
    for iter in 1:maxiter
        # update variance components
        ldiv!(transpose(Ωchol.L), storagen) # stroagen is inv(Ω) * residual
        for j in 1:m
            cons = dot(Ωinv, V[j])
            #print(cons)
            #print("\n")
            Ωj_ = Ω - σ2[j]*V[j]
            f(x) = fsd(x,Ωj_,V[j],y, cons)  
            if j == 1 & low_rank
                (σ2jnew, niter) = newtonm2(f, 10^-4,σ2[j])
            else 
                (σ2jnew, niter) = newtonm(f, 10^-4,σ2[j])
            end
            Ω = Ω +  (σ2jnew- σ2[j])*V[j]
            σ2[j] = σ2jnew
            if immediate_update
                Ωchol = cholesky(Ω)
                Ωinv = inv(Ωchol)
            end
        end
        # update Ω
        Ωchol = cholesky(Ω)
        Ωinv = inv(Ωchol)
        # update fixed effects
        mul!(storagenp, Ωinv, X)
        mul!(storagepp, transpose(X), storagenp)
        mul!(storagen, Ωinv, y)
        mul!(storagep, transpose(X), storagen)
        copy!(β, storagep)
        if !isposdef(storagepp)
            storagepp += T(1e-3)I
        end
        ldiv!(cholesky(Symmetric(storagepp)), β)
        mul!(res, X, β)
        res .= y .- res
        copy!(storagen, res)
        ldiv!(Ωchol.L, storagen)
        # check convergence
        loglold = logl
        logl = (loglConst - logdet(Ωchol) - norm(storagen)^2) / 2
        verbose && println("iter=", iter," logl=", logl)
        verbose && println("β=", β," σ2=", σ2)
        if abs(logl - loglold) < funtol * (abs(logl) + 1) 
            niter = iter
            break
        end

    end
    # output
    return β, σ2, niter, logl
end


function fsd(x,Gi_,vi,y, cons)    
    L = Gi_+x*vi
    Zy = cg(L,y)
    temp = vi*Zy
    fd = -Zy'*temp+cons
    temp2 = cg(L,temp)
    sd = 2*temp'*temp2
    return fd, sd     
end


function fsd1(x,Gi_,vi,y, cons)    
    L = Gi_+x*vi
    Zy = cg(L,y)
    temp = vi*Zy
    fd = -Zy'*temp+cons
    return fd   
end

function fsd2(x,Gi_,vi,y, cons)    
    L = Gi_+x*vi
    Zy = L\y
    temp = vi*Zy
    fd = -Zy'*temp+cons
    return fd   
end

function fsd3(x,Gi_,vi,y, cons)    
    L = Gi_+x*vi
    Zy = L\y
    temp = vi*Zy
    fd = -Zy'*temp+cons
    temp2 = L\temp
    sd = 2*temp'*temp2
    return fd, sd     
end

function fsd4(x,Gi_,vi,y, cons)    

    return y'*inv(Gi_+x*vi)*y 
end



function newtonm(fun,tol,x_start)
    (fd,sd) = fun(0);
    x = 0;
    if fd>0
        niter = 0;
        xfinal = x;
        return xfinal, niter
    else
       x = x_start
       (fd,sd) = fun(x);
       max_iter = 100;
       niter = 0;
       for i=1:max_iter
          xnew = x-fd/sd;
          while xnew<0
              sd = sd*2;
              xnew = x-fd/sd;
          end
          (fd,sd) = fun(xnew);
          ctol = abs(xnew-x);
          x = xnew;
          niter = niter+1; 
            
          if ctol<tol
              xfinal= x;
              return xfinal, niter
          end              
       end
       xfinal= x;
       return xfinal, niter
    end
end


function newtonm2(fun,tol,x_start)
   x = x_start;
   (fd,sd) = fun(x);
   max_iter = 100;
   niter = 0;
   for i=1:max_iter
      xnew = x-fd/sd;
      while xnew<0
          sd = sd*2;
          xnew = x-fd/sd;
      end
      (fd,sd) = fun(xnew);
      ctol = abs(xnew-x);
      x = xnew;
      niter = niter+1; 
      if ctol<tol
          xfinal= x;
          return xfinal, niter
      end              
   end
   xfinal= x;
   return xfinal, niter
end

function vc_cd_wb(
    y::AbstractVector{T}, 
    X::AbstractMatrix{T},
    Z::Vector{<:AbstractMatrix{T}},
    V::Vector{<:AbstractMatrix{T}};
    verbose::Bool = false, 
    maxiter::Integer = 5000, 
    funtol::Number = 1e-8,
    β::Vector{T} = X \ y,
    σ2::Vector{T} = ones(T, length(V)),
    algo::Symbol = :MM,
    low_rank::Bool = false,
    ) where T <: LinearAlgebra.BlasFloat

    n, p = size(X)
    m    = length(V)   # no. variance components
    # pre-allocate working arrays
    storagen  = zeros(T, n)
    storagenp = zeros(T, n, p)
    storagepp = zeros(T, p, p)
    storagep  = zeros(T, p)
    # initialize algorithm
    
    Ω = zeros(T, n, n)
    for j in 1:length(σ2)
        Ω .+= σ2[j] * V[j]
    end
    
    Ωchol = cholesky(Ω)
    local Ωinv  = inv(Ωchol)
    res   = y - X * β
    copy!(storagen, res)
    ldiv!(Ωchol.L, storagen)
    loglConst = - n * log(2π)
    logl = (loglConst - logdet(Ωchol) - norm(storagen)^2) / 2
    verbose && println("iter=", 0," logl=", logl)
    verbose && println("β=", β," σ2=", σ2)
    niter = maxiter
    # MM/EM/FS loop
    for iter in 1:maxiter
        # update variance components
        ldiv!(transpose(Ωchol.L), storagen) # stroagen is inv(Ω) * residual
        for j in 1:m
            if j == 1 & low_rank
                Vn = I(n)
                cons = dot(Ωinv, Vn)
                Ωj_ = Ω - σ2[j]*Vn
                f(x) = fsd(x,Ωj_,Vn,y, cons)  
                (σ2jnew, niter) = newtonm2(f, 10^-4,σ2[j])
                Ω = Ω +  (σ2jnew - σ2[j])*Vn
                σ2[j] = σ2jnew
                Ωchol = cholesky(Ω)
                Ωinv = inv(Ωchol)
            else
                Vj = V[j]
                iz = j-1
                Zi = Z[iz]
                B = Zi'*Ωinv*Zi
                yt = Zi'*(Ωinv*y)
                σt = σ2[j]
                #cons = dot(Ωinv, Vj)
                cons = tr(B)
                g(x) = fsdwb(x,B,yt,σt,cons)  
                (σ2jnew, niter) = newtonm(g, 10^-4, σt)
                alph = (σ2jnew-σt)
                Ω .+= alph*Vj
                np, pi = size(Zi)
                BI  = B*alph+I(pi)
                GZ = Zi'*Ωinv
                Ωinv = Ωinv-alph*GZ'*(BI\GZ)
                σ2[j] = σ2jnew
            end
        end
        # update Ω
        Ωchol = cholesky(Ω)
        Ωinv = inv(Ωchol)
        # update fixed effects
        mul!(storagenp, Ωinv, X)
        mul!(storagepp, transpose(X), storagenp)
        mul!(storagen, Ωinv, y)
        mul!(storagep, transpose(X), storagen)
        copy!(β, storagep)
        if !isposdef(storagepp)
            storagepp += T(1e-3)I
        end
        ldiv!(cholesky(Symmetric(storagepp)), β)
        mul!(res, X, β)
        res .= y .- res
        copy!(storagen, res)
        ldiv!(Ωchol.L, storagen)
        # check convergence
        loglold = logl
        logl = (loglConst - logdet(Ωchol) - norm(storagen)^2) / 2
        verbose && println("iter=", iter," logl=", logl)
        #verbose && println("β=", β," σ2=", σ2)
        if abs(logl - loglold) < funtol * (abs(logl) + 1)
            niter = iter
            break
        end
    end
    # output
    return β, σ2, niter, logl
end

function fsd(x,Gi_,vi,y, cons)    
    L = Gi_+x*vi
    Zy = cg(L,y)
    temp = vi*Zy
    fd = -Zy'*temp+cons
    temp2 = cg(L,temp)
    sd = 2*temp'*temp2
    return fd, sd     
end

function fsd1(x,Gi_,vi,y, cons)    
    L = Gi_+x*vi
    Zy = cg(L,y)
    temp = vi*Zy
    fd = -Zy'*temp+cons
    return fd   
end

function fsd2(x,Gi_,vi,y, cons)    
    L = Gi_+x*vi
    Zy = L\y
    temp = vi*Zy
    fd = -Zy'*temp+cons
    return fd   
end

function fsd3(x,Gi_,vi,y, cons)    
    L = Gi_+x*vi
    Zy = L\y
    temp = vi*Zy
    fd = -Zy'*temp+cons
    temp2 = L\temp
    sd = 2*temp'*temp2
    return fd, sd     
end

function fsd4(x,Gi_,vi,y, cons)    

    return y'*inv(Gi_+x*vi)*y 
end

function fsdwb(x,B,yt,σt,cons)   
    x = x-σt
    M = I(length(yt))+x*B
    My = cg(M,yt)
    term = My'*B*My
    fd = -yt'*My+x*term+cons
    temp = B*My
    MBMy = cg(M,temp)
    sd = 2*term+x*temp'*MBMy
    return fd, sd     
end

function fsdwb2(x,B,yt,σt,cons)   
    x = x-σt
    M = I(length(yt))+x*B
    My = M\yt
    term = My'*B*My
    fd = -yt'*My+x*term+cons
    temp = B*My
    MBMy = M\temp
    sd = 2*term+x*temp'*MBMy
    return fd, sd     
end

n = 1000
ms = [20, 50 , 75, 100]
vranks = [20, 50 , 75, 100, 125, 150]
nms = length(ms)
nrs = length(vranks)
sigma1s = [0.1 1 10]
nsim = 10 
global objs= zeros(nsim, nmethod)
global t = zeros(nsim, nmethod)
global iters = ones(Int, nsim, nmethod) 

for l in 1= 1:sigma1s
    print("sigma1:   ")
    print(sigma1s[l])
    print("\n")
    for i=1: nrs
        for j =1:nms
            print("\nRank: ")
            print(vranks[i])
            print("\n")
            print("m: ")
            print(ms[j])        
            print("\n")
            for k in 1:nsim 
                print(k)
                p,qi,qidx, X, Z, Zall,V,β ,σ2,Ω,y = simulate_data(n,vranks[i],ms[j], sigma1s[l]);
                t[k, 1] = @elapsed βhat, σ2hat, iters[k,1], obj = vc_cd_wb(y, X, Z, V; algo = :MM, verbose = false, funtol = funtolall, low_rank = true)
                objs[k,1] = obj
                t[k, 2] = @elapsed βhat, σ2hat, iters[k,2], objs[k,2] = vc(y, X, V; algo = :MM, verbose = false,  funtol = funtolall, maxiter = 5*10^3, stop_comp = false, out_comp  = obj);
            end
            print("Arrays:\n")
            print("time:\n")
            print(t)
            print("Iters:\n")
            print(iters)
            print("Objs:\n")
            print(objs)
        end
    end
end
