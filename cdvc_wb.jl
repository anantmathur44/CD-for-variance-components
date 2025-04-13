
"""
# Coordinate Descent for Variance Components Implementation using Woodbury Identity

# This code implements the algorithm from Mathur et al. (2022) and code base builds upon the MM approach from Zhou et al. (2016).

# References:
# - Mathur et al. (2022). *Coordinate Descent for Variance Components*. 
# - Zhou et al. (2016). *MM Algorithm for Variance Components*. 

cdvc_wb(y, Z, V)

# Fitting a variance component model with zero-mean responses `y`, and 
# positive semi-definite matrices `I, V[2]`, ..., `V[end]`, and covariance 
# `γ[1]*I + γ[2]*V[2] + ... + γ[end]*V[end]`.
# Suitable when has Z[i] has columns much less than n.

# Arguments
# - `y`: n-by-1 response vector.
# - `Z`: `Z[i]` is the i-th random effects matriz.
# - `V`: `Z[i]*Z[i].T` is the i-th psd variance component matrix.


# Keyword Arguments
# - `verbose`: logical, verbose display.
# - `maxiter`: maximum number of iterations.
# - `funtol`: convergence tolerance for objective value.
# - `γ`: starting value of variance components.

# Output
# - `γ`: fitted variance components.
# - `nter`: number of iterations.
# - `logl`: fitted log-likelihood.
"""

import Pkg; Pkg.add("JLD"); Pkg.add("LinearAlgebra"), Pkg.add("DelimitedFiles"), Pkg.add("IterativeSolvers"), Pkg.add("Roots"), Pkg.add("CSV"), Pkg.add("Tables")
using JLD,LinearAlgebra, DelimitedFiles, IterativeSolvers, Roots, CSV, Tables, Statistics


function cdvc_wb(
    y::AbstractVector{T}, 
    Z::Vector{<:AbstractMatrix{T}},
    V::Vector{<:AbstractMatrix{T}};
    verbose::Bool = false, 
    maxiter::Integer = 5000, 
    funtol::Number = 1e-8,
    γ::Vector{T} = ones(T, length(V)),
    algo::Symbol = :MM,
    low_rank::Bool = false,
    ) where T <: LinearAlgebra.BlasFloat

    n = length(y)
    m    = length(V)   # no. variance components

    # pre-allocate working arrays
    storagen  = zeros(T, n)
    # initialize algorithm    
    Ω = zeros(T, n, n)
    for j in 1:length(γ)
        Ω .+= γ[j] * V[j]
    end
    
    Ωchol = cholesky(Ω)
    local Ωinv  = inv(Ωchol)
    res   = y
    copy!(storagen, res)
    ldiv!(Ωchol.L, storagen)
    loglConst = - n * log(2π)
    logl = (loglConst - logdet(Ωchol) - norm(storagen)^2) / 2
    verbose && println("iter=", 0," logl=", logl)
    verbose && println(" γ=", γ)
    niter = maxiter
    # MM/EM/FS loop
    logs = []
    for iter in 1:maxiter
        # update variance components
        ldiv!(transpose(Ωchol.L), storagen) # stroagen is inv(Ω) * residual
        for j in 1:m
            if j == 1
                Vn = I(n)
                cons = dot(Ωinv, Vn)
                Ωj_ = Ω - γ[j]*Vn
                f(x) = fsd(x,Ωj_,Vn,y, cons) 
                if low_rank
                    (γjnew, niter) = newtonm2(f, 10^-6,γ[j])
                else
                    (γjnew, niter) = newtonm(f, 10^-6,γ[j])
                end
                Ω = Ω +  (γjnew - γ[j])*Vn
                γ[j] = γjnew
                Ωchol = cholesky(Ω)
                Ωinv = inv(Ωchol)
            else
                Vj = V[j]
                iz = j-1
                Zi = Z[iz]
                B = Zi'*Ωinv*Zi
                yt = Zi'*(Ωinv*y)
                σt = γ[j]
                cons = tr(B)
                g(x) = fsdwb(x,B,yt,σt,cons)  
                (γjnew, niter) = newtonm(g, 10^-6, σt)
                alph = (γjnew-σt)
                Ω .+= alph*Vj
                np, pi = size(Zi)
                BI  = B*alph+I(pi)
                GZ = Zi'*Ωinv
                Ωinv = Ωinv-alph*GZ'*(BI\GZ)
                γ[j] = γjnew
            end
        end
        # update Ω
        Ωchol = cholesky(Ω)
        Ωinv = inv(Ωchol)
        # update fixed effects
        mul!(storagen, Ωinv, y)
        if !isposdef(storagepp)
            storagepp += T(1e-3)I
        end
        copy!(storagen, res)
        ldiv!(Ωchol.L, storagen)
        # check convergence
        loglold = logl
        logl = (loglConst - logdet(Ωchol) - norm(storagen)^2) / 2
        verbose && println("iter=", iter," logl=", logl)
        #verbose && println(" γ=", γ)
        logs = [ logs; logl]
        if abs(logl - loglold) < funtol * (abs(logl) + 1)
            niter = iter
            break
        end
    end
    # output
    return  γ, niter, logl, logs
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


function fsd(x,Gi_,vi,y, cons)    
    L = Gi_+x*vi
    Zy = cg(L,y)
    temp = vi*Zy
    fd = -Zy'*temp+cons
    temp2 = cg(L,temp)
    sd = 2*temp'*temp2
    return fd, sd     
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
