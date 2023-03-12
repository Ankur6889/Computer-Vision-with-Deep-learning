function ObjFcn = makeObjFcn(XTrain,YTrain,XValidation,YValidation)
ObjFcn = @valErrorFun;
    function [valError,cons,fileName] = valErrorFun(optVars)
    filterSize = 3; % filter size
    dropout_value=0.1;
    inputSize = [98 50 1]; % input image size
    numClasses = 12;
lgraph = createConvNetwork(optVars.nlayers,optVars.nfilters,filterSize,inputSize,numClasses);

% analyze the network
analyzeNetwork(lgraph)
 options = trainingOptions('adam', ...
'MiniBatchSize',optVars.MiniBatchSize, ...
'MaxEpochs',30, ...
'InitialLearnRate',optVars.InitialLearnRate, ...
'ValidationData',{XValidation,YValidation},...
'L2Regularization',optVars.L2Regularization, ...
'Verbose',true, ...
'Plots','training-progress',...
'ExecutionEnvironment','gpu');
% Train the network using the training data

 trainedNet= trainNetwork(XTrain,YTrain,lgraph,options);
  YPredicted = classify(trainedNet,XValidation);
  valError = 1 - mean(YPredicted == YValidation);
  fileName = num2str(valError) + ".mat";
  save(fileName,'trainedNet','valError','options')
  cons = [];
        
    end
end