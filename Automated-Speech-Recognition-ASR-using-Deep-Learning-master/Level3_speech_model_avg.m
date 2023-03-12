load ACS61011projectData.mat

% sample training data with replacement and train model
% size of training data
XTrainSize = size(XTrain); % get dimensions of training data XTrain
N = XTrainSize(4); % N is the number of training data samples

%====1===1====1===1===1===1===1===1===1===1===1===1===1===1===1===1===1===1===%

% generate random samples with replacement according to size of training data
idx = randi([1 N],N,1);
% check this: the percentage of unique samples in idx - usually about 63%
% this indicates we have sampled about 63% of the original data
% if you do this a few different times you get a different 63% each time
uniqueSamples = 100*length(unique(idx))/length(idx);
% create new randomly sampled training data from the indices idx

%First Model
XTrain1 = XTrain(:,:,1,idx);
YTrain1 = YTrain(idx,1);
dropout_value=0.1;
model_1 = [

imageInputLayer([98 50 1])
convolution2dLayer([3 3],8,"Padding","same")
batchNormalizationLayer
reluLayer
dropoutLayer(dropout_value)


maxPooling2dLayer([2 2],"Padding","same","Stride",[2 2])
convolution2dLayer([3 3],16,"Padding","same")
batchNormalizationLayer
reluLayer
dropoutLayer(dropout_value)

maxPooling2dLayer([2 2],"Padding","same","Stride",[2 2])
convolution2dLayer([3 3],32,"Padding","same")
batchNormalizationLayer
reluLayer
dropoutLayer(dropout_value)

maxPooling2dLayer([2 2],"Padding","same","Stride",[2 2])
convolution2dLayer([3 3],64,"Padding","same")
batchNormalizationLayer
reluLayer
dropoutLayer(dropout_value)

fullyConnectedLayer(12)
dropoutLayer(dropout_value)
softmaxLayer

classificationLayer
];

% analyze the network
analyzeNetwork(model_1)

 options_1 = trainingOptions('adam', ...
'MiniBatchSize',128, ...
'MaxEpochs',30, ...
'InitialLearnRate',1e-3, ...
'ValidationData',{XValidation,YValidation},...
'Verbose',true, ...
'Plots','training-progress',...
'Plots','training-progress',...
'ExecutionEnvironment','gpu');
% Train the network using the training data

net_1 = trainNetwork(XTrain1,YTrain1,model_1,options_1);
[YPred1,probs1] = classify(net_1,XValidation);

%===2===2===2===2===2===2===2===2===2===2===2===2===2===2===2===2===2===2===2===%

%==========This Model is not using any dropout and using rms prop ================%

% again generatig random samples with replacement according to size of training data
% generate random samples with replacement according to size of training data
idx = randi([1 N],N,1);
% check this: the percentage of unique samples in idx - usually about 63%
% this indicates we have sampled about 63% of the original data
% if you do this a few different times you get a different 63% each time
uniqueSamples = 100*length(unique(idx))/length(idx);
% create new randomly sampled training data from the indices idx

%Second Model
XTrain2 = XTrain(:,:,1,idx);
YTrain2 = YTrain(idx,1);
%dropout_value=0.1;
model_2 = [

imageInputLayer([98 50 1])
convolution2dLayer([3 3],8,"Padding","same")
batchNormalizationLayer
reluLayer
%dropoutLayer(dropout_value)


maxPooling2dLayer([2 2],"Padding","same","Stride",[2 2])
convolution2dLayer([3 3],16,"Padding","same")
batchNormalizationLayer
reluLayer
%dropoutLayer(dropout_value)

maxPooling2dLayer([2 2],"Padding","same","Stride",[2 2])
convolution2dLayer([3 3],32,"Padding","same")
batchNormalizationLayer
reluLayer
%dropoutLayer(dropout_value)

maxPooling2dLayer([2 2],"Padding","same","Stride",[2 2])
convolution2dLayer([3 3],64,"Padding","same")
batchNormalizationLayer
reluLayer
%dropoutLayer(dropout_value)

fullyConnectedLayer(12)
%dropoutLayer(dropout_value)
softmaxLayer

classificationLayer
];

% analyze the network
analyzeNetwork(model_2)

 options_2 = trainingOptions('rmsprop', ...
'MiniBatchSize',128, ...
'MaxEpochs',30, ...
'InitialLearnRate',1e-3, ...
'ValidationData',{XValidation,YValidation},...
'Verbose',true, ...
'Plots','training-progress',...
'Plots','training-progress',...
'ExecutionEnvironment','gpu');
% Train the network using the training data

net_2 = trainNetwork(XTrain2,YTrain2,model_2,options_2);
[YPred2,probs2] = classify(net_2,XValidation);

%===3===3===3===3===3===3===3===3===3===3===3===3===3===3===3===3===3===3===3===%

%==========This Model uses more convolution layers then the previous 2 models also uses droput ================%

% again generatig random samples with replacement according to size of training data
% generate random samples with replacement according to size of training data
idx = randi([1 N],N,1);
% check this: the percentage of unique samples in idx - usually about 63%
% this indicates we have sampled about 63% of the original data
% if you do this a few different times you get a different 63% each time
uniqueSamples = 100*length(unique(idx))/length(idx);
% create new randomly sampled training data from the indices idx

%Third Model
XTrain3 = XTrain(:,:,1,idx);
YTrain3 = YTrain(idx,1);
dropout_value=0.1;
model_3 = [

imageInputLayer([98 50 1])
convolution2dLayer([3 3],8,"Padding","same")
batchNormalizationLayer
reluLayer
dropoutLayer(dropout_value)


maxPooling2dLayer([2 2],"Padding","same","Stride",[2 2])
convolution2dLayer([3 3],16,"Padding","same")
batchNormalizationLayer
reluLayer
dropoutLayer(dropout_value)

maxPooling2dLayer([2 2],"Padding","same","Stride",[2 2])
convolution2dLayer([3 3],32,"Padding","same")
batchNormalizationLayer
reluLayer
dropoutLayer(dropout_value)

maxPooling2dLayer([2 2],"Padding","same","Stride",[2 2])
convolution2dLayer([3 3],64,"Padding","same")
batchNormalizationLayer
reluLayer
dropoutLayer(dropout_value)

maxPooling2dLayer([2 2],"Padding","same","Stride",[2 2])
convolution2dLayer([3 3],32,"Padding","same")
batchNormalizationLayer
reluLayer
dropoutLayer(dropout_value)

fullyConnectedLayer(12)
dropoutLayer(dropout_value)
softmaxLayer

classificationLayer
];

% analyze the network
analyzeNetwork(model_3)

 options_3 = trainingOptions('adam', ...
'MiniBatchSize',128, ...
'MaxEpochs',30, ...
'InitialLearnRate',1e-3, ...
'ValidationData',{XValidation,YValidation},...
'Verbose',true, ...
'Plots','training-progress',...
'Plots','training-progress',...
'ExecutionEnvironment','gpu');
% Train the network using the training data

net_3 = trainNetwork(XTrain3,YTrain3,model_3,options_3);
[YPred3,probs3] = classify(net_3,XValidation);

Prediction = [YPred1,YPred2,YPred3];
pred = mode(Prediction,2);
accuracy = 100*mean(pred == YValidation); % accuracy
display(['Validation Accuracy: ' num2str(accuracy) '%']);
accuracy_1=100*mean(YPred1 == YValidation);
accuracy_2=100*mean(YPred2 == YValidation);
accuracy_3=100*mean(YPred3 == YValidation);
% plot confusion matrix
plotconfusion(YValidation,pred)



