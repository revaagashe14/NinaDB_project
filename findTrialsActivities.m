function trialActivityArray = findTrialsActivities()
    trialActivityArray = [];
    matrix = load('Data/S1_A1_E3.mat', 'restimulus');
    indices = matrix.restimulus;
    trials = [];
    startIndices = [];
    endIndices = [];
    startFound = 0;
    for i=1:227493
        if (indices(i) == 0 && startFound == 1)
            endIndices = [endIndices, i-1];
            startFound = 0;
            
        else 
            if (startFound == 0 & indices(i) ~= 0)
                startIndices = [startIndices, i];
                startFound = 1;
                trials = [trials, indices(i)];
            end
        end
    end
    trialActivityArray.trials = trials;
    trialActivityArray.starts = startIndices;
    trialActivityArray.ends = endIndices;
end
