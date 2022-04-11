%% Visualize Kinematics Data
%%% change the value of i and j variable for number of trials and tasks
%%% which need to be visualized.
clear all;
dataMatrix = load('S1_A1_E3.mat', 'emg', 'glove');
theta_estimates = load('a13t1_estimates.mat');
trialsActivities = findTrialsActivities();

startActivity = 1;
endActivity = 1;
startTrial = 1;
endTrial = 1;

trial_activity_data = [];

for activity = startActivity:endActivity
    for trial = startTrial:endTrial
        emg_signal = dataMatrix.('glove');
        index = 10*(activity-1) + trial;
        newTimeLength = -trialsActivities.('starts')(index)+trialsActivities.ends(index);

        trial_activity_data = [trial_activity_data; emg_signal(trialsActivities.('starts')(index):trialsActivities.ends(index), :)];
    end
end


 
 
% trialInd =  [1 2 3 4 5 6 7 8 9 10];  
% taskInd = [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23];
ds = 100; % downsampling rate
Y_ds_filt_emg = [];     Z_Kinematics = [];


% for j = 1:1  % task
%    for i = 1:1  %all 5 train trial
%        intr = trialInd(i);
%        ints = taskInd(j);
%        Y_ds_filt_emg    = [Y_ds_filt_emg; dsfilt_emg{intr,j}];
%        Z_Kinematics     = [Z_Kinematics; finger_kinematics{intr,ints}];
%    end
% end

%Z_Kinematics = trial_activity_data(1:end,:);

Z_Kinematics = theta_estimates(1:end, :); 


size(Z_Kinematics)
figure (1);
for i=1:size(Z_Kinematics)
    clf;
    posVal =Z_Kinematics(i,:);
    handMocapVisualiseKINE(posVal);
    pause(0.001);
end
%% Plot EMG Data
%plot(Y_ds_filt_emg(:,1));
%%
function handMocapModifyKINE(handle, posVal)

limb{1} = [20 17;17 18;18 19]; %thumb
limb{2} = [20 1;1 5;5 6;6 7];  %index
limb{3} = [20 2;2 8;8 9;9 10]; %middle
limb{4} = [20 3;3 11;11 12;12 13]; %ring
limb{5} = [20 4;4 14;14 15;15 16]; %little
limb{6} = [21 22;22 23;23 21]; %wrist

% Convert positions for plotting.
    jointMarkerPos = handJointPosExtract(posVal);
    counter = 0;
      linestyle = '-';
      markersize = 20;
      marker = '.';

%disp([num2str(posVal)]);
    set(handle(1), 'Xdata', posVal(:, 1), 'Ydata', posVal(:, 2), 'Zdata', posVal(:, 3));
    set(handle(1), 'markersize', markersize);
    for i = 1:1:length(limb)
       for j = 1:1:size(limb{i},1)
          counter = counter + 1; 
          set(handle(counter+1), 'Xdata', jointMarkerPos(limb{i}(j,:),1), 'Ydata', jointMarkerPos(limb{i}(j,:),2), ...
                                 'Zdata', jointMarkerPos(limb{i}(j,:),3) );
        end  
    end
end
%%
function handle = handMocapVisualiseKINE(posVal)
% 
% limb{1} = [20 17;17 18;18 19]; %thumb
% limb{2} = [20 1;1 5;5 6;6 7];  %index
% limb{3} = [20 2;2 8;8 9;9 10]; %middle
% limb{4} = [20 3;3 11;11 12;12 13]; %ring
% limb{5} = [20 4;4 14;14 15;15 16]; %little
% limb{6} = [21 22;22 23;23 21]; %wrist



limb{1} = [21 1;1 2;2 3]; %thumb
limb{2} = [21 5;5 11;11 6;6 7];  %index
limb{3} = [21 8;8 15;15 9;9 10]; %middle
limb{4} = [21 12;12 19;19 13;13 14]; %ring
limb{5} = [21 20;20 16;16 17;17 18]; %little
limb{6} = [1 22;22 21;21 1]; %wrist

% Convert positions for plotting.
    jointMarkerPos = handJointPosExtract(posVal);
    counter = 0;
      linestyle = '-';
      markersize = 20;
      marker = '.';
      color ='b';

    handle(1) = plot3(posVal(:, 1), posVal(:, 2), posVal(:, 3));
    set(handle(1), 'markersize', markersize,'color',color, 'LineWidth', 3.0);
    axis on;
    hold on
    grid on
    for i = 1:1:length(limb) % 
       for j = 1:1:size(limb{i},1)% 
          counter = counter + 1; 
          handle(counter+1) = line(jointMarkerPos(limb{i}(j,:),1),jointMarkerPos(limb{i}(j,:),2), ...
                             jointMarkerPos(limb{i}(j,:),3), 'LineWidth', 2, 'LineStyle', linestyle, ...
                             'Marker', marker, 'markersize', markersize,'color',color);
       
        end  
    end 
    axis([-100 300 -120 110 -110 120]); % in general
 
end
    %%
function joint = handJointPosExtract(pos)

% XYZANKUR2JOINT Converts data to xyz positions for each joint.
%
%	Description:
%
%	XYZANKUR2JOINT(POS, JOINT) takes in a vector of values and returns a
%	matrix with points in rows and coordinate positions in columns.
%	 Arguments:
%	  POS - the vector of values.
%	  JOINT - the matrix of values with points in rows and x,y,z
%	   positions in columns.
%	


%	Copyright (c) 2008 Carl Henrik Ek and Neil Lawrence
% 	xyzankur2joint.m SVN version 161
% 	last update 2008-12-01T06:39:30.000000Z
% joint(:,1) = pos(1:3:end);
% joint(:,2) = pos(2:3:end);      
% joint(:,3) = pos(3:3:end);

joint(:, 1) = pos;
joint(:, 2) = pos;
joint(:, 3) = pos;

return
end