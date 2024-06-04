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
