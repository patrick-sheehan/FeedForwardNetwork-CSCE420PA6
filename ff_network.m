# Patrick Sheehan
# CSCE420 - PA6
# Feed Forward Network
# 24 April 2014

# load training data
tData = importdata("training_data.txt", " ");
[tRows, tColumns] = size(tData);
tInput = tData(:, 1:end-1);
tOutput = tData(:, end);
tInput = tInput';     # transpose for proper format
tOutput = tOutput';
nTrainSets = tRows;

# load evaluation data
eData = importdata("evaluation_data.txt", " ");
[eRows, eColumns] = size(eData);
eInput = eData(:,1:end-1);
eOutput = eData(:,end);
eInput = eInput';     # transpose for proper format
eOutput = eOutput';
nEvalSets = eRows;


# define the max and min inputs for each row
mMinMaxElements = min_max(tInput); 

# define network parameters
nHiddenNeurons = 2;
nOutputNeurons = 1;

# creat feed-forward network
MLPnet = newff(mMinMaxElements,[nHiddenNeurons nOutputNeurons], {"tansig","purelin"},"trainlm","","mse");

# define training data to pass to train()
trainData.P = tInput;
trainData.T = tOutput;

# define validation data to pass to train()
evalData.P = eInput;
evalData.T = eOutput;


# train the network
[net] = train(MLPnet, tInput, tOutput); #, [], [], trainData);

# test the network and save outputs to a file

# simulate with training set
[trainSimOut] = sim(net, tInput);

# simulate with evaluation set
[evalSimOut] = sim(net, eInput);

# write the estimated outputs to file
save trainSimOut.txt trainSimOut
save evalSimOut.txt evalSimOut
