# NeuralMaxEnt

## Data
I have used this data for initial analysis [pvc3 data](https://crcns.org/data-sets/vc/pvc-3/about)

## Experiments and analysis
To run experiments on the above dataset, run the `blanch_expt.jl` file with the following command:
```julia blanche_expt.jl <binsz> <maxiter> <numsplt> <dev> <batchsize>```

For these experiments, `binsz` ranged from 500 to 5000 (in microseconds), `maxiter` was set to 100, `numsplit` was set to 10, `dev` is just the device ID for different runs and `batchsize` was set to 10000.


This code base is using the [Julia Language](https://julialang.org/) and
[DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> NeuralMaxEnt

To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box, including correctly finding local paths.

You may notice that most scripts start with the commands:
```julia
using DrWatson
@quickactivate "NeuralMaxEnt"
```
which auto-activate the project and enable local path handling from DrWatson.

