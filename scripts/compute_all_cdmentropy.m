addpath("~/Documents/CDMentropy/src")
addpath("~/Documents/CDMentropy/lib/PYMentropy/src")
pkg load statistics
pkg load parallel


basedir = "../data/exp_pro/matrix_old/joost_long/full/cdmentropy";
dirs = readdir(basedir)(3:end);
for i = 1:length(dirs)
    sdp = fullfile(basedir, dirs(i)){1};
    files = readdir(sdp);
    files = files(3:length(files));
    fps = cell(length(files), 1);
    for j = 1:length(files)
        fps{j} = fullfile(sdp, files(j, 1));
    end
    [Hbs, Hvs] = pararrayfun(nproc-1, @compute_cdmentropy, fps);
    savepath = fullfile(basedir, strcat(dirs(i), ".mat")){1};
    save(savepath, "files", "Hbs", "Hvs", "-v6")
end

