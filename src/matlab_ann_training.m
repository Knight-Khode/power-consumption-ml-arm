% upload data from csv into MATLAB
data= readtable("testData.csv");

%convert data into array
arrayData= table2array(data);

X= arrayData(:, 1:5)' % size of 5*samples representing the features 
Y= arrayData(:,6)' % this contains a row vector of ground truth where each column corresponds to a sample of features


% Define the number of hidden neurons
hiddenLayerSize = [18,18];

% Create a Pattern Recognition Network
net = patternnet(hiddenLayerSize);

% Set the transfer functions for the layers
net.layers{1}.transferFcn = 'poslin'; %ReLu in Matlab
net.layers{2}.transferFcn = 'poslin';
net.layers{end}.transferFcn = 'logsig';  % Output layer with sigmoid function

% View the network configuration
view(net);

net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;



% Train the network
[net, tr] = train(net, X, Y);

% Test the network
outputs = net(X);
errors = gsubtract(Y, outputs);
performance = perform(net, Y, outputs);

% View the training performance
plotperform(tr);


%extract weightbs and biases 

inputWeights = net.IW{1};  % Weights for the connections from input to first hidden layer
layer1Weights = net.LW{2,1};  % Weights for the connections from first hidden layer to second hidden layer
outputWeights = net.LW{3,2};  % Weights for the connections from second hidden layer to output layer

layer1Biases = net.b{1};  % Biases for the first hidden layer
layer2Biases = net.b{2};  % Biases for the second hidden layer
outputBiases = net.b{3};  % Biases for the output layer

