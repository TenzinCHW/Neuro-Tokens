include("blanche_expt.jl")
import Images


if abspath(PROGRAM_FILE) == @__FILE__
    numwin = 3
    numbin = 3
    winszs = [i for i in 1:numwin]
    binszs = [1000i for i in 1:numbin]
    dir = "../data/exp_raw/pvc3/crcns_pvc3_cat_recordings/spont_activity/spike_data_area18"
    maxiter = 50
    ents = zeros(numwin, numbin)
    for (i, winsz) in enumerate(winszs)
        for (j, binsz) in enumerate(binszs)
            winsz = 10
            binsz = 2000
            println("win $winsz bin $binsz")
            params = Dict("winsz"=>winsz, "binsz"=>binsz, "maxiter"=>maxiter)
            data = extract_bin_spikes(dir, binsz)
            data = trim_recording(data, 0.15094)
            println(size(data))
            #println(size(data[12446:end-12445, :]));exit()
            data, counts = window(data, winsz)
            println(size(data))
            img = Images.Gray.(data)
            Images.save(DrWatson.plotsdir("raw_blanche", "$(binsz)_$(winsz).png"), img)
        end
    end
end

