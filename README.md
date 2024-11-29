# RenewalInference

[![Build Status](https://github.com/SanteriVtj/RenewalInference.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/SanteriVtj/RenewalInference.jl/actions/workflows/CI.yml?query=branch%3Amain)


Model for doing patent value evaluation from renewal data. Model is partly based on <a href="https://doi.org/10.1111/1467-937X.00064" title="Patent Protection in the Shadow of Infringement: Simulation Estimations of Patent Value">Lanjouw 1998</a>.

## Running a simulation


In order to run a simulation with the software one needs to create a ModelData instance
```julia
md = ModelData(
    survival,
    cost,
    X,
    Σ,
    renewals
)
```
Where survival is the observed number of periods that each patent have survived, cost defines the patenting cost scheme, X is the data-set that defines the intial value distribution, and Σ is the dataset defining learning distribution.

Then the simulation draw instance can be created

```julia
sim = Sim(T,S)
```

where `T` is the maximum number of periods that patent can be enforced and `S` number of number of simulation draws.

After this the loss function has to be defined:
```julia
f(x) = fval2(x,md,sim)
```
and finally passed to the optimizer:
```julia
optimize_n(
    f,
    [Uniform(),Uniform(),...],
    n,
    [0,0,-Inf,...]
    [1,1,Inf,...],
    options=Optim.Options(...)
)
```
The first argument for the optimizer is the loss function, second takes vector of distributions from which the intial values are drawn, `n` is the number of optimizations that are run, next two vectors hold box constraints for the parameter vector and last takes ´Optim.Options´ instance, if some additional options are needed for the optimization.