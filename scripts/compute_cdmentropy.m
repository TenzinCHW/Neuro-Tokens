function [Hb, Hv] = parcompute_cdmentropy(filepath)
    load(filepath{1,1}{1,1});
    % for some reason cells is a matrix, so have to extract
    counts = double(counts);
    spike_counts = double(spike_counts);
    cells = double(cells(1));
    [Hb, Hv] = computeH_CDM(counts, spike_counts, cells(1));
end

