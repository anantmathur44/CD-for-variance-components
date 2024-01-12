"""
cdvc(y, X, V)

Fitting a variance component model with zero-mean responses `y`, and 
positive semi-definite matrices `I, V[2]`, ..., `V[end]`, and covariance `σ2[1]*I+σ2[2]*V[2]+ ...+σ2[end]*V[end]`. 

# Arguments
- `y`: n-by-1 response vector.  
- `V`: `V[i]` is the i-th psd matrix.  

# Keyword Arguments
- `verbose`: logical, verbose display.
- `maxiter`: maximum number of iterations.
- `funtol`: covergence tolerance for objective value.
- `γ`: staring value of variance components.

# Output
- `γ`: fitted variance components.
- `nter`: number of iterations.
- `logl`: fitted log-likelihood.
"""

import Pkg; Pkg.add("JLD"); Pkg.add("LinearAlgebra"), Pkg.add("DelimitedFiles"), Pkg.add("IterativeSolvers"), Pkg.add("Roots"), Pkg.add("CSV"), Pkg.add("Tables")
using JLD,LinearAlgebra, DelimitedFiles, IterativeSolvers, Roots, CSV, Tables, Statistics

function cdvc(
    y::AbstractVector{T}, 
    X::AbstractMatrix{T},
    V::Vector{<:AbstractMatrix{T}};
    verbose::Bool = false, 
    maxiter::Integer = 5000, 
    funtol::Number = 1e-8,
    γ::Vector{T} = ones(T, length(V)),
    cdi::Bool = false
    ) where T <: LinearAlgebra.BlasFloat

    n, p = size(X)
    m    = length(V)   # no. variance components
    
    # pre-allocate working arrays
    storagen  = zeros(T, n)
    # initialize algorithm
    Ω = zeros(T, n, n)
    update_Ω!(Ω, γ, V)
    Ωchol = cholesky(Ω)
    Ωinv  = inv(Ωchol)
    res   = y
    copy!(storagen, res)
    ldiv!(Ωchol.L, storagen)
    loglConst = - n * log(2π)
    logl = (loglConst - logdet(Ωchol) - norm(storagen)^2) / 2
    verbose && println("iter=", 0," logl=", logl)
    verbose && println( "γ=", γ)
    niter = maxiter
    
    # CD loop
    for iter in 1:maxiter
        # update variance components
        ldiv!(transpose(Ωchol.L), storagen) # stroagen is inv(Ω) * residual
        for j in 1:m
            cons = dot(Ωinv, V[j])
            #print(cons)
            #print("\n")
            Ωj_ = Ω - γ[j]*V[j]
            f(x) = fsd(x,Ωj_,V[j],y, cons)  
            (γjnew, niter) = newtonm(f, 10^-4,γ[j])
            Ω = Ω +  (γjnew- γ[j])*V[j]
            γ[j] = γjnew
            if cdi
                Ωchol = cholesky(Ω)
                Ωinv = inv(Ωchol)
            end
        end
        
        # update Ω
        Ωchol = cholesky(Ω)
        Ωinv = inv(Ωchol)
        copy!(storagen, res)
        ldiv!(Ωchol.L, storagen)
        # check convergence
        loglold = logl
        logl = (loglConst - logdet(Ωchol) - norm(storagen)^2) / 2
        verbose && println("iter=", iter," logl=", logl)
        #verbose && println("β=", β," γ=", γ)
        if (abs(logl - loglold) < funtol * (abs(logl) + 1))
            niter = iter
            break
        end

    end
    # output
    return γ, niter, logl
end

# Evaluate first and second derivatives of CD function
function fsd(x,Gi_,vi,y, cons)    
    L = Gi_+x*vi
    Zy = cg(L,y)
    temp = vi*Zy
    fd = -Zy'*temp+cons
    temp2 = cg(L,temp)
    sd = 2*temp'*temp2
    return fd, sd     
end

#Implement Newton's method
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


function update_Ω!(
    Ω::AbstractMatrix{T},
    γ::AbstractVector{T},
    V::Vector{<:AbstractMatrix{T}}
    ) where T <: AbstractFloat
    fill!(Ω, 0)
    for j in 1:length(γ)
        Ω .+= γ[j] .* V[j]
    end
    Ω
end
