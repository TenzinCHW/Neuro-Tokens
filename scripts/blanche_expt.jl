import DrWatson
DrWatson.@quickactivate
import MEFK, NPZ, Flux, CUDA
include("extract_blanche.jl")


function model_to_cpu(model, params)
    if isa(model, MEFK.MEF3T)
        cpumodel = MEFK.MEF3T(model, Array)
        #cpumodel = MEFK.MEF3T(model.n,
        #                  model.W1 |> Array,
        #                  model.W2 |> Array,
        #                  model.W3 |> Array,
        #                  model.W2_mask |> Array,
        #                  model.W3_mask |> Array,
        #                  [g |> Array for g in model.gradients],
        #                  Array
        #                 )
    elseif isa(model, MEFK.MEF2T)
        cpumodel = MEFK.MEF2T(model, Array)
        #cpumodel = MEFK.MEF2T(model.n,
        #                  model.W1 |> Array,
        #                  model.W2 |> Array,
        #                  model.W2_mask |> Array,
        #                  [g |> Array for g in model.gradients],
        #                  Array
        #                 )
    else
        cpumodel = MEFK.MEFMPNK(model, Array)
        #cpumodel = MEFK.MEFMPNK(model.n,
        #                    model.K,
        #                    model.W .|> Array,
        #                    [g |> Array for g in model.grad],
        #                    [inds .|> Array for inds in model.indices],
        #                    [winds .|> Array for winds in model.windices],
        #                    Array
        #                   )
    end
    cpumodel
end


function train_on_data(data, max_iter, winsz, batchsize, array_cast)
    println("windowing")
    data, counts = window(data, winsz)
    n = size(data)[2]
    println(size(data))
    # Prep dataloader
    println("prep loader")
    counts = reshape(counts, (length(counts), 1))
    loader = Flux.Data.DataLoader((data', counts'); batchsize=batchsize, partial=true)
    model = MEFK.MEF2T(n; array_cast=array_cast)
    optim = Flux.setup(Flux.Adam(0.1), model)
    for i in 1:max_iter
        loss = 0
        grads = [0, 0, 0]
        for (d, c) in loader
            d = d' |> Array
            l, _ = model(d, c[:])
            loss += l
        end
        println(loss)
        grads = MEFK.retrieve_reset_gradients!(model; reset_grad=true)
        Flux.update!(optim, model, grads)
    end
    model
end


function recording_split_train(data, num_split::Int)
    split_sz = size(data)[1] / num_split
    [data[ceilint(i*split_sz)+1:ceilint((i+1)*split_sz), :] for i in 0:num_split-1]
end


function converge_dynamics(model, data)
    out_ = MEFK.dynamics(model, data)
    out = MEFK.dynamics(model, out_)
    while !all(out .== out_)
        out_ .= out
        out = MEFK.dynamics(model, out)
    end
    out
end


if abspath(PROGRAM_FILE) == @__FILE__
    dir = DrWatson.datadir("exp_raw", "pvc3", "crcns_pvc3_cat_recordings", "spont_activity", "spike_data_area18")
    params = Dict("binsz"=>parse(Int, ARGS[1]), "maxiter"=>parse(Int, ARGS[2]))
    winszs = [5i for i in 2:10]
    max_iter = params["maxiter"]
    binsz = params["binsz"]
    num_split = parse(Int, ARGS[3])

    data = extract_bin_spikes(dir, binsz)
    data = trim_recording(data, 0.15)#094)
    println(size(data))
    data_split = recording_split_train(data, num_split)
    println("splits $num_split total $(size(data))")

    dev = parse(Int, ARGS[4])
    CUDA.device!(dev)
    array_cast = CUDA.cu
    batchsize = parse(Int, ARGS[5])

    #k = 2
    #params["k"] = k
    #model = MEFK.MEFMPNK(n, k; array_cast=array_cast)
    #model = MEFK.MEF3T(n; array_cast=array_cast)
    println("starting")
    for winsz in winszs
        params["winsz"] = winsz
        # Only doing matrix for now
        save_loc = DrWatson.datadir("exp_pro", "matrix", "split", "models", "$(DrWatson.savename(params)).jld2")
        if isfile(save_loc)
            println("$save_loc exists, skipping")
            continue
        end

        save_data = Dict()
        for i in 1:num_split
            model = train_on_data(data_split[i], max_iter, winsz, batchsize, array_cast)
            # TODO converge on data_split[i]
            input = ordered_window(data_split[i])
            output = converge_dynamics(model, input |> array_cast) |> Array
            cpumodel = model_to_cpu(model, params)
            save_data["$i"] = Dict("input"=>input, "output"=>output, "net"=>cpumodel)
        end

        DrWatson.wsave(save_loc, save_data)
    end
end

