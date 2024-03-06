import Pkg
Pkg.activate("..")
import MEFK, NPZ, Flux, CUDA

function uniquecount(data)
    unique_array = unique(data, dims=1)
    counts = Dict(unique_array[i, :] => 0 for i in 1:size(unique_array)[1])
    for i in 1:size(data)[1]
      counts[data[i, :]] += 1
    end
    reduce(hcat, keys(counts))' |> Array, values(counts) |> collect .|> Float32
end

function window(data, win_sz::Int)
    n_sample = size(data)[1] - win_sz + 1
    out = [data[i:i+win_sz-1, :][:] for i in 1:n_sample]
    reduce(hcat, out)' |> Array
end

if abspath(PROGRAM_FILE) == @__FILE__
    data = NPZ.npzread("../data/exp_raw/blanche_140000_area18.npz")["spikes_arr"][1, :, :]' .|> Int8
    win_sz = 10
    array_cast = CUDA.cu
    data = window(data, win_sz)
    data, counts = uniquecount(data)
    for i in 1:size(data)[1]
        if sum(data[i, :]) == 0
            println(i)
            counts[i] = 1000
            break
        end
    end
    n = size(data)[2]
    # model = MEFK.MEFMPNK(n, 2)
    model = MEFK.MEF3T(n; array_cast=array_cast)
    optim = Flux.setup(Flux.Adam(0.1), model)
    max_iter = 100

    for i in 1:max_iter
        # loss, grads = model(data, counts, i==1)
        loss, grads = model(data, counts)
        println(loss)
        Flux.update!(optim, model, grads)
    end
end