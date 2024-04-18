import DrWatson
DrWatson.@quickactivate
import DataStructures


ceilint(a) = a |> ceil |> Int


function convert_to_Int64(raw_spike::Array{UInt8})
    val::Int64 = 0
    i = 0
    for v in reverse(raw_spike) # little endian
        val <<= 8
        val |= v
    end
    val
end


function convert_to_Int64_arr(raw_spikes::Array{UInt8})
    length(raw_spikes) % 8 == 0 || error("Wrong number of bytes")
    nspike = Int(length(raw_spikes) / 8)
    spikes = [convert_to_Int64(raw_spikes[1+8*(i-1):8*i]) for i in 1:nspike]
    spikes
end


function extract_spikes(fp)
    open(fp) do f
        raw_spikes = read(f)
        return convert_to_Int64_arr(raw_spikes)
    end
end


mm(m, stuff) = m([m(s...) for s in stuff]...)


function bin_spikes(spikes_times, binsz)
    num_bins = max([max(st...) / binsz for st in spikes_times]...) |> ceilint
    num_neurons = length(spikes_times)
    bin_data = zeros(UInt8, (num_bins, num_neurons))
    for (i, neuron_spikes) in enumerate(spikes_times)
        for st in neuron_spikes
            bin_data[st / binsz |> ceilint, i] = 1
        end
    end
    bin_data
end


function extract_bin_spikes(dir::String, binsz) # binsz is in μs
    spike_files = [f for f in readdir(dir) if f[end-2:end] == "spk"]
    spikes_times = extract_spikes.(joinpath.(dir, spike_files))
    bin_spikes(spikes_times, binsz)
end


calc_n_sample(data, win_sz) = size(data)[1] - win_sz + 1


function window(data, win_sz::Int)
    n_sample = calc_n_sample(data, win_sz)
    patt_cnt = DataStructures.DefaultDict{Vector{UInt8}, Float32}(0)
    for i in 1:n_sample
        patt_cnt[data[i:i+win_sz-1, :][:]] += 1
    end
    out = reduce(hcat, patt_cnt |> keys |> collect)' |> Array
    cnt = patt_cnt |> values |> collect
    out, cnt
end


function ordered_window(data, win_sz::Int)
    n_sample = calc_n_sample(data, win_sz)
    out = [data[i:i+win_sz-1, :][:] for i in 1:n_sample]
    reduce(hcat, out)' |> Array
end


function trim_recording(data, pct::AbstractFloat)
    @assert 0 <= pct < 1
    tot_sz = size(data)[1]
    bf_trim = tot_sz * pct / 2 |> ceilint
    data[bf_trim+1:end-bf_trim, :]
end


if abspath(PROGRAM_FILE) == @__FILE__
    dir = "../data/exp_raw/pvc3/crcns_pvc3_cat_recordings/spont_activity/spike_data_area18"
    binszs = [1000i for i in 1:10]
    winszs = [i for i in 1:20]
    windowed = []
    for binsz in binszs
        spikes = extract_bin_spikes(dir, binsz)
        Threads.@threads for winsz in winszs
            # race condition for dict so put into array first
            push!(windowed, ("$binsz $winsz", window(spikes, winsz)))
        end
    end
    data = Dict(windowed)
    DrWatson.wsave(DrWatson.datadir("exp_raw", "blanche_bin_windowed.jld2"), data)
end

