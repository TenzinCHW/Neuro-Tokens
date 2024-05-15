# NeuralMaxEnt

## DrWatson install instructions
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

## Estimating entropy with CDM in octave
First install [octave](https://octave.org/download).
Then, start octave and innstall `statistics`, `struct` and `parallel` with
```
pkg install -forge statistics
pkg install -forge struct
pkg install -forge parallel
```

## Data
I have used these two datasets for my experiments: [pvc3](https://crcns.org/data-sets/vc/pvc-3/about)
and [this one from Lamberti](https://datadryad.org/stash/dataset/doi:10.5061/dryad.p5hqbzkqj).

## Experiments and analysis
To run experiments on the above datasets, place their files into a `data/exp_raw` directory and
run the `blanch_expt.jl` file with the following command:
```julia blanche_expt.jl <binsz> <maxiter> <numsplit> <dev> <batchsize>```

### Heatmaps
For these experiments, `binsz` ranged from 500 to 6000 (in microseconds), `maxiter` was set to 100,
`numsplit` was set to 1, `dev` is the device ID and `batchsize` was set to 10000.
If GPU runs out of memory, please lower the `batchsize` variable or use smaller window sizes.

### Entropy vs window size
We chose a fixed `binsz` and varied `winsz` from 1 to some large value (typically up to
a timescale that we're interested in based on the chosen bin size).

## Citations
Blanche, Tim (2009): Multi-neuron recordings in primary visual cortex. CRCNS.org.
http://dx.doi.org/10.6080/K0MW2F2J
Lamberti, M., Hess, M., Dias, I. et al. Maximum entropy models provide functional connectivity estimates in neural networks. Sci Rep 12, 9656 (2022). https://doi.org/10.1038/s41598-022-13674-4

