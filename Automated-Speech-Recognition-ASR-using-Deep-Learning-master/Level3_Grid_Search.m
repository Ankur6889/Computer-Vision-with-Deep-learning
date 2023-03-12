clear all;
load ACS61011projectData.mat

%% dynamic layer graph creation
% dynamically create a deep convolutional network based on the following inputs
nlayers = [2,3,4]; % number of convolutional layers
nfilters = [8,16,32]; % number of filters in each layer
filterSize = 3; % filter size
inputSize = [98 50 1]; % input image size
numClasses = 12;


% Train the network
for i=1:length(nlayers)
    for j= 1:length(nfilters)
        lgraph = createConvNetwork(nlayers(i),nfilters(j),filterSize,inputSize,numClasses);
        analyzeNetwork(lgraph);
        options = trainingOptions('adam', ...
        'MaxEpochs',60, ...
        'MiniBatchSize',128, ...
        'Plots','training-progress', ...
        'Verbose',true, ...
        'ValidationData',{XValidation,YValidation}, ...
        'ExecutionEnvironment','gpu');
        trainedNet = trainNetwork(XTrain,YTrain,lgraph,options);
        % Classify the validation images using the trained network
        [YPred,probs] = classify(trainedNet,XValidation);
        accuracy = 100*mean(YPred == YValidation); % accuracy
        disp(['Validation Accuracy: ' num2str(accuracy) '%']);
        accuracy_matrix(i,j)=accuracy;
        % plot confusion matrix
        plotconfusion(YValidation,YPred)
    end
end
fprintf("Table showing Validation Accuracy for various Networks");
Number_of_Layers=[nlayers(1),nlayers(2),nlayers(3)];
Filters_8=[accuracy_matrix(1,1),accuracy_matrix(2,1),accuracy_matrix(3,1)];
Filters_16=[accuracy_matrix(1,2),accuracy_matrix(2,2),accuracy_matrix(3,2)];
Filters_32=[accuracy_matrix(1,3),accuracy_matrix(2,3),accuracy_matrix(3,3)];
validation_table=table(Number_of_Layers',Filters_8',Filters_16',Filters_32')



