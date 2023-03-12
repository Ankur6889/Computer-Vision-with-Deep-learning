%% define network in code
% define network layers
clear all;
load mostdata.mat
dropout_value=0.1;
model = [

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
analyzeNetwork(model)
 options = trainingOptions('adam', ...
'MiniBatchSize',128, ...
'MaxEpochs',30, ...
'InitialLearnRate',1e-3, ...
'ValidationData',{XValidation,YValidation},...
'Verbose',true, ...
'Plots','training-progress',...
'Plots','training-progress',...
'ExecutionEnvironment','gpu');
% Train the network using the training data

net = trainNetwork(XTrain,YTrain,model,options);


% Classify the validation images using the trained network
[YPred,probs] = classify(net,XValidation);
accuracy = 100*mean(YPred == YValidation); % accuracy
disp(['Validation Accuracy: ' num2str(accuracy) '%']);

% plot confusion matrix
plotconfusion(YValidation,YPred)
