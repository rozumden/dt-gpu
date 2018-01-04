%%% Matlab file for plotting the experiments.

imgs = {'man.jpg', 'io.png','stripes.jpg','pattern.jpg'};

for i = 1:numel(imgs)
	im = imread(imgs{i});
	sz(i) = size(im,1) * size(im,2);
end
%%%%%%%%% 3x3 %%%%%%%%%%%
exn = 1;

cpu{exn} =  [0.000133632 0.00221014 0.0931701 0.170447];
gpu1{exn} = [0.0021477   0.00682804 1.77135   3.22811];
gpu2{exn} = [0.00213469  0.0031112  0.0535174 0.120993];

%%%%%%%%% 5x5 %%%%%%%%%%%
exn = 2;

cpu{exn} =  [0.00017314 0.00228281 0.115029  0.231032];
gpu1{exn} = [0.00182596 0.0101653  0.990944  4.9895];
gpu2{exn} = [0.00242452 0.00324057 0.15459   0.242954];

%%%%%%%%% exact %%%%%%%%%%%
exn = 3;

cpu{exn} =  [0.000293392 0.00248415 0.118699  0.196388];
gpu1{exn} = [0.00136573  0.0037048  0.971304  1.497001];
gpu2{exn} = [0.00252771  0.00400622 0.0880544 0.080774];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fc = 1000;
for i = 1:exn
	figure;
	hold on;
	plot(sz, [cpu{i}; gpu1{i}; gpu2{i}]*fc, '-');
	legend('CPU', 'GeForce GTS 450', 'GTX TITAN X');
	xlabel('Problem size');
	ylabel('Time [ms]');
	ylim([0 fc*gpu1{i}(end)/3])
end