clear all;clc;
%%Forward TCN Feature Extraction
class=['C','H','E'];
classes = cell(3,1);
for c=1:3
    classes{c,1}=class(c);
end
numClasses = numel(classes);
numBlocks = 4;
numFilters = 512;
filterSize = 5;
dropoutFactor = 0.05;
hyperparameters = struct;
hyperparameters.NumBlocks = numBlocks;
hyperparameters.DropoutFactor = dropoutFactor;
miniBatchSize = 2048;
numObservations = numel(XTrain);
flag=0;
if rem((numObservations./miniBatchSize),1)==0
    numIterationsPerEpoch = numObservations./miniBatchSize;
else
    numIterationsPerEpoch = floor(numObservations./miniBatchSize)+1;
    flag=1;
end
len=[];
for i=1:numel(XTrain)
    len(i)=size(YTrain{i,1},2);
end
for20_train={};
train_i=1;
for i = 1:numIterationsPerEpoch

        if i==numIterationsPerEpoch&&flag==1
            idx = (i-1)*miniBatchSize+1:numObservations;
        else
            idx = (i-1)*miniBatchSize+1:i*miniBatchSize;
        end
        str_len=len(idx);
         [X,Y,numTimeSteps] = transformSequences(XTrain(idx),YTrain(idx));
    dlX = dlarray(X);

doTraining = false;
[dlYPred_train,dlYPred_train20] = model(dlX,parameters,hyperparameters,doTraining);

YPred_train = gather(extractdata(dlYPred_train));

for j=1:numel(idx)
for20_train{train_i,1}=extractdata(gather([XTrain{train_i,1};double(reshape(dlYPred_train20(:,j,1:str_len(j)),20,str_len(j)))]));
train_i=train_i+1;
end
end
forXTrain={};
for i=1:11650
    forXTrain{i,1}=for20_train{i,1}(62:81,:);
end
test_i=1;
        XTest=XTest_va;
        YTest=YTest_va;
        len=str_len_v;
numObservationsTest = numel(XTest);
numIterationsPerEpoch = floor(numObservationsTest./miniBatchSize)+1;
for i = 1:numIterationsPerEpoch

        if i==numIterationsPerEpoch
            idx = (i-1)*miniBatchSize+1:numObservationsTest;
        else
            idx = (i-1)*miniBatchSize+1:i*miniBatchSize;
        end
        str_len=len(idx);
        
        [X,Y] = transformSequences(XTest(idx),YTest(idx));
dlXTest = dlarray(X);

doTraining = false;
[dlYPred,dlYPred20] = model(dlXTest,parameters,hyperparameters,doTraining);

YPred = gather(extractdata(dlYPred));

for j=1:numel(idx)
for20_va{test_i,1}=extractdata(gather([XTest{test_i,1};double(reshape(dlYPred20(:,j,1:str_len(j)),20,str_len(j)))]));
test_i=test_i+1;
end
end
forXTest_va={};
for i=1:1456
    forXTest_va{i,1}=for20_va{i,1}(62:81,:);
end
test_i=1;
        XTest=XTest_te;
        YTest=YTest_te;
        len=str_len_t;
numObservationsTest = numel(XTest);
numIterationsPerEpoch = floor(numObservationsTest./miniBatchSize)+1;
for i = 1:numIterationsPerEpoch

        if i==numIterationsPerEpoch
            idx = (i-1)*miniBatchSize+1:numObservationsTest;
        else
            idx = (i-1)*miniBatchSize+1:i*miniBatchSize;
        end
        str_len=len(idx);
        
        [X,Y] = transformSequences(XTest(idx),YTest(idx));
dlXTest = dlarray(X);

doTraining = false;
[dlYPred,dlYPred20] = model(dlXTest,parameters,hyperparameters,doTraining);

YPred = gather(extractdata(dlYPred));

for j=1:numel(idx)
for20_te{test_i,1}=extractdata(gather([XTest{test_i,1};double(reshape(dlYPred20(:,j,1:str_len(j)),20,str_len(j)))]));
test_i=test_i+1;
end
end
forXTest_te={};
for i=1:1456
    forXTest_te{i,1}=for20_te{i,1}(62:81,:);
end

for test_i=1:5
    if test_i==1
        XTest=XTest10;
        YTest=YTest10;
        str_len=str10_len;
    elseif test_i==2
        XTest=XTest11;
        YTest=YTest11;
        str_len=str11_len;
    elseif test_i==3
        XTest=XTest12;
        YTest=YTest12;
        str_len=str12_len;
    elseif test_i==4
        XTest=XTest1313;
        YTest=YTest1313;
        str_len=str1313_len;
    else
        XTest=XTest14;
        YTest=YTest14;
        str_len=str14_len;
    end
    
numObservationsTest = numel(XTest);

[X,Y] = transformSequences(XTest,YTest);
dlXTest = dlarray(X);

doTraining = false;
[dlYPred,dlYPred20] = model(dlXTest,parameters,hyperparameters,doTraining);

for j=1:numel(XTest)
    if test_i==1
        for20_Xtest10{j,1}=extractdata(gather([XTest{j,1};double(reshape(dlYPred20(:,j,1:str_len(j)),20,str_len(j)))]));
    elseif test_i==2
        for20_Xtest11{j,1}=extractdata(gather([XTest{j,1};double(reshape(dlYPred20(:,j,1:str_len(j)),20,str_len(j)))]));
    elseif test_i==3
        for20_Xtest12{j,1}=extractdata(gather([XTest{j,1};double(reshape(dlYPred20(:,j,1:str_len(j)),20,str_len(j)))]));
    elseif test_i==4
        for20_Xtest1313{j,1}=extractdata(gather([XTest{j,1};double(reshape(dlYPred20(:,j,1:str_len(j)),20,str_len(j)))]));
    else
        for20_Xtest14{j,1}=extractdata(gather([XTest{j,1};double(reshape(dlYPred20(:,j,1:str_len(j)),20,str_len(j)))]));
    end
end

end
forXTest10={};
for i=1:99
    forXTest10{i,1}=for20_Xtest10{i,1}(62:81,:);
end
forXTest11={};
for i=1:81
    forXTest11{i,1}=for20_Xtest11{i,1}(62:81,:);
end
forXTest12={};
for i=1:19
    forXTest12{i,1}=for20_Xtest12{i,1}(62:81,:);
end
forXTest1313={};
for i=1:22
    forXTest1313{i,1}=for20_Xtest1313{i,1}(62:81,:);
end
forXTest14={};
for i=1:23
    forXTest14{i,1}=for20_Xtest14{i,1}(62:81,:);
end

test_i=1;
        XTest=XTest13;
        YTest=YTest13;
        len=str13_len;
numObservationsTest = numel(XTest);
numIterationsPerEpoch = floor(numObservationsTest./miniBatchSize)+1;
for i = 1:numIterationsPerEpoch

        if i==numIterationsPerEpoch
            idx = (i-1)*miniBatchSize+1:numObservationsTest;
        else
            idx = (i-1)*miniBatchSize+1:i*miniBatchSize;
        end
        str_len=len(idx);
        
        [X,Y] = transformSequences(XTest(idx),YTest(idx));
dlXTest = dlarray(X);

doTraining = false;
[dlYPred,dlYPred20] = model(dlXTest,parameters,hyperparameters,doTraining);

YPred = gather(extractdata(dlYPred));

for j=1:numel(idx)
for20_cb513{test_i,1}=extractdata(gather([XTest{test_i,1};double(reshape(dlYPred20(:,j,1:str_len(j)),20,str_len(j)))]));
test_i=test_i+1;
end
end
forXTest13={};
for i=1:504
    forXTest13{i,1}=for20_cb513{i,1}(62:81,:);
end

%%Back TCN Feature Extraction
class=['C','H','E'];
classes = cell(3,1);
for c=1:3
    classes{c,1}=class(c);
end
numClasses = numel(classes);
numBlocks = 4;
numFilters = 512;
filterSize = 5;
dropoutFactor = 0.05;
hyperparameters = struct;
hyperparameters.NumBlocks = numBlocks;
hyperparameters.DropoutFactor = dropoutFactor;
miniBatchSize = 2048;
numObservations = numel(XTrain);
flag=0;
if rem((numObservations./miniBatchSize),1)==0
    numIterationsPerEpoch = numObservations./miniBatchSize;
else
    numIterationsPerEpoch = floor(numObservations./miniBatchSize)+1;
    flag=1;
end
len=[];
for i=1:numel(XTrain)
    len(i)=size(YTrain{i,1},2);
end
for20_train={};
train_i=1;
for i = 1:numIterationsPerEpoch

        if i==numIterationsPerEpoch&&flag==1
            idx = (i-1)*miniBatchSize+1:numObservations;
        else
            idx = (i-1)*miniBatchSize+1:i*miniBatchSize;
        end
        str_len=len(idx);
         [X,Y,numTimeSteps] = transformSequences(XTrain(idx),YTrain(idx));
    dlX = dlarray(X);

doTraining = false;
[dlYPred_train,dlYPred_train20] = model(dlX,parameters,hyperparameters,doTraining);

YPred_train = gather(extractdata(dlYPred_train));

for j=1:numel(idx)
back20_train{train_i,1}=extractdata(gather([XTrain{train_i,1};double(reshape(dlYPred_train20(:,j,1:str_len(j)),20,str_len(j)))]));
train_i=train_i+1;
end
end
backXTrain={};
for i=1:11650
    backXTrain{i,1}=back20_train{i,1}(62:81,:);
end
test_i=1;
        XTest=XTest_va;
        YTest=YTest_va;
        len=str_len_v;
numObservationsTest = numel(XTest);
numIterationsPerEpoch = floor(numObservationsTest./miniBatchSize)+1;
for i = 1:numIterationsPerEpoch

        if i==numIterationsPerEpoch
            idx = (i-1)*miniBatchSize+1:numObservationsTest;
        else
            idx = (i-1)*miniBatchSize+1:i*miniBatchSize;
        end
        str_len=len(idx);
        
        [X,Y] = transformSequences(XTest(idx),YTest(idx));
dlXTest = dlarray(X);

doTraining = false;
[dlYPred,dlYPred20] = model(dlXTest,parameters,hyperparameters,doTraining);

YPred = gather(extractdata(dlYPred));

for j=1:numel(idx)
back20_va{test_i,1}=extractdata(gather([XTest{test_i,1};double(reshape(dlYPred20(:,j,1:str_len(j)),20,str_len(j)))]));
test_i=test_i+1;
end
end
backXTest_va={};
for i=1:1456
    backXTest_va{i,1}=back20_va{i,1}(62:81,:);
end
test_i=1;
        XTest=XTest_te;
        YTest=YTest_te;
        len=str_len_t;
numObservationsTest = numel(XTest);
numIterationsPerEpoch = floor(numObservationsTest./miniBatchSize)+1;
for i = 1:numIterationsPerEpoch

        if i==numIterationsPerEpoch
            idx = (i-1)*miniBatchSize+1:numObservationsTest;
        else
            idx = (i-1)*miniBatchSize+1:i*miniBatchSize;
        end
        str_len=len(idx);
        
        [X,Y] = transformSequences(XTest(idx),YTest(idx));
dlXTest = dlarray(X);

doTraining = false;
[dlYPred,dlYPred20] = model(dlXTest,parameters,hyperparameters,doTraining);

YPred = gather(extractdata(dlYPred));

for j=1:numel(idx)
back20_te{test_i,1}=extractdata(gather([XTest{test_i,1};double(reshape(dlYPred20(:,j,1:str_len(j)),20,str_len(j)))]));
test_i=test_i+1;
end
end
backXTest_te={};
for i=1:1456
    backXTest_te{i,1}=back20_te{i,1}(62:81,:);
end


for test_i=1:5
    if test_i==1
        XTest=XTest10;
        YTest=YTest10;
        str_len=str10_len;
    elseif test_i==2
        XTest=XTest11;
        YTest=YTest11;
        str_len=str11_len;
    elseif test_i==3
        XTest=XTest12;
        YTest=YTest12;
        str_len=str12_len;
    elseif test_i==4
        XTest=XTest1313;
        YTest=YTest1313;
        str_len=str1313_len;
    else
        XTest=XTest14;
        YTest=YTest14;
        str_len=str14_len;
    end
    
numObservationsTest = numel(XTest);

[X,Y] = transformSequences(XTest,YTest);
dlXTest = dlarray(X);

doTraining = false;
[dlYPred,dlYPred20] = model(dlXTest,parameters,hyperparameters,doTraining);

for j=1:numel(XTest)
    if test_i==1
        for20_Xtest10{j,1}=extractdata(gather([XTest{j,1};double(reshape(dlYPred20(:,j,1:str_len(j)),20,str_len(j)))]));
    elseif test_i==2
        for20_Xtest11{j,1}=extractdata(gather([XTest{j,1};double(reshape(dlYPred20(:,j,1:str_len(j)),20,str_len(j)))]));
    elseif test_i==3
        for20_Xtest12{j,1}=extractdata(gather([XTest{j,1};double(reshape(dlYPred20(:,j,1:str_len(j)),20,str_len(j)))]));
    elseif test_i==4
        for20_Xtest1313{j,1}=extractdata(gather([XTest{j,1};double(reshape(dlYPred20(:,j,1:str_len(j)),20,str_len(j)))]));
    else
        for20_Xtest14{j,1}=extractdata(gather([XTest{j,1};double(reshape(dlYPred20(:,j,1:str_len(j)),20,str_len(j)))]));
    end
end

end
backXTest10={};
for i=1:99
    backXTest10{i,1}=for20_Xtest10{i,1}(62:81,:);
end
backXTest11={};
for i=1:81
    backXTest11{i,1}=for20_Xtest11{i,1}(62:81,:);
end
backXTest12={};
for i=1:19
    backXTest12{i,1}=for20_Xtest12{i,1}(62:81,:);
end
backXTest1313={};
for i=1:22
    backXTest1313{i,1}=for20_Xtest1313{i,1}(62:81,:);
end
backXTest14={};
for i=1:23
    backXTest14{i,1}=for20_Xtest14{i,1}(62:81,:);
end

test_i=1;
        XTest=XTest13;
        YTest=YTest13;
        len=str13_len;
numObservationsTest = numel(XTest);
numIterationsPerEpoch = floor(numObservationsTest./miniBatchSize)+1;
for i = 1:numIterationsPerEpoch

        if i==numIterationsPerEpoch
            idx = (i-1)*miniBatchSize+1:numObservationsTest;
        else
            idx = (i-1)*miniBatchSize+1:i*miniBatchSize;
        end
        str_len=len(idx);
        
        [X,Y] = transformSequences(XTest(idx),YTest(idx));
dlXTest = dlarray(X);

doTraining = false;
[dlYPred,dlYPred20] = model(dlXTest,parameters,hyperparameters,doTraining);

YPred = gather(extractdata(dlYPred));

for j=1:numel(idx)
for20_cb513{test_i,1}=extractdata(gather([XTest{test_i,1};double(reshape(dlYPred20(:,j,1:str_len(j)),20,str_len(j)))]));
test_i=test_i+1;
end
end
backXTest13={};
for i=1:504
    backXTest13{i,1}=for20_cb513{i,1}(62:81,:);
end

%%Feature fusion
for i=1:numel(forXTrain)
    XTrain{i,1}=forXTrain{i,1}+fliplr(backXTrain{i,1});
end
clear forXTrain;clear backXTrain;
for i=1:numel(forXTest_va)
    XTest_va{i,1}=forXTest_va{i,1}+fliplr(backXTest_va{i,1});
end
clear forXTest_va;clear backXTest_va;
for i=1:numel(forXTest_te)
    XTest_te{i,1}=forXTest_te{i,1}+fliplr(backXTest_te{i,1});
end
clear forXTest_te;clear backXTest_te;
for i=1:numel(forXTest10)
    XTest10{i,1}=forXTest10{i,1}+fliplr(backXTest10{i,1});
end
clear forXTest10;clear backXTest10;
for i=1:numel(forXTest11)
    XTest11{i,1}=forXTest11{i,1}+fliplr(backXTest11{i,1});
end
clear forXTest11;clear backXTest11;
for i=1:numel(forXTest12)
    XTest12{i,1}=forXTest12{i,1}+fliplr(backXTest12{i,1});
end
clear forXTest12;clear backXTest12;
for i=1:numel(forXTest1313)
    XTest1313{i,1}=forXTest1313{i,1}+fliplr(backXTest1313{i,1});
end
clear forXTest1313;clear backXTest1313;
for i=1:numel(forXTest14)
    XTest14{i,1}=forXTest14{i,1}+fliplr(backXTest14{i,1});
end
clear forXTest14;clear backXTest14;
for i=1:numel(forXTest13)
    XTest13{i,1}=forXTest13{i,1}+fliplr(backXTest13{i,1});
end
clear forXTest13;clear backXTest13;


%%Build the TCN model.
numObservations = numel(XTrain);
class=['C','H','E'];
classes = cell(3,1);
for c=1:3
    classes{c,1}=class(c);
end
numClasses = numel(classes);

numBlocks = 1;
numFilters = 512;
filterSize = 5;
dropoutFactor = 0.05;

hyperparameters = struct;
hyperparameters.NumBlocks = numBlocks;
hyperparameters.DropoutFactor = dropoutFactor;

numInputChannels = 20;

parameters = struct;
numChannels = numInputChannels;

for k = 1:numBlocks
    parametersBlock = struct;
    blockName = "Block"+k;
    
    weights = initializeGaussian([filterSize, numChannels, numFilters]);
    bias = zeros(numFilters, 1, 'single');
    parametersBlock.Conv1.Weights = dlarray(weights);
    parametersBlock.Conv1.Bias = dlarray(bias);
    
    weights = initializeGaussian([filterSize, numFilters, numFilters]);
    bias = zeros(numFilters, 1, 'single');
    parametersBlock.Conv2.Weights = dlarray(weights);
    parametersBlock.Conv2.Bias = dlarray(bias);
    
    weights = initializeGaussian([filterSize, numFilters, numFilters]);
    bias = zeros(numFilters, 1, 'single');
    parametersBlock.Conv3.Weights = dlarray(weights);
    parametersBlock.Conv3.Bias = dlarray(bias);
    
    % If the input and output of the block have different numbers of channels, 
    % then add a convolution with filter size 1.
    if numChannels ~= numFilters
        weights = initializeGaussian([1, numChannels, numFilters]);
        bias = zeros(numFilters, 1, 'single');
        parametersBlock.Conv4.Weights = dlarray(weights);
        parametersBlock.Conv4.Bias = dlarray(bias);
    end
    numChannels = numFilters;
    
    parameters.(blockName) = parametersBlock;
end

weights = initializeGaussian([numClasses,20]);
bias = zeros(numClasses,1,'single');
weights20 = initializeGaussian([20,numChannels]);
bias20 = zeros(20,1,'single');

parameters.FC.Weights = dlarray(weights);
parameters.FC.Bias = dlarray(bias);
parameters.FC.Weights20 = dlarray(weights20);
parameters.FC.Bias20 = dlarray(bias20);

maxEpochs = 1;
miniBatchSize = 2048;
initialLearnRate = 0.001;
learnRateDropFactor = 0.1;
learnRateDropPeriod = 10;
gradientThreshold = 1;

executionEnvironment = "gpu";
plots = "training-progress";

%%Train Model
learnRate = initialLearnRate;
trailingAvg = [];
trailingAvgSq = [];

iteration = 0;
flag=0;
if rem((numObservations./miniBatchSize),1)==0
    numIterationsPerEpoch = numObservations./miniBatchSize;
else
    numIterationsPerEpoch = floor(numObservations./miniBatchSize)+1;
    flag=1;
end

start = tic;
aaa=[];
% Loop over epochs.
for epoch = 1:maxEpochs
    
    %Shuffle the data.
    idx = randperm(numObservations);
    XTrain = XTrain(idx);
    YTrain = YTrain(idx);
     
    % Loop over mini-batches.
    for i = 1:numIterationsPerEpoch
        iteration = iteration + 1;
        
        % Read mini-batch of data and apply the transformSequence preprocessing function.
        if i==numIterationsPerEpoch&&flag==1
            idx = (i-1)*miniBatchSize+1:numObservations;
        else
            idx = (i-1)*miniBatchSize+1:i*miniBatchSize;
        end
        
        [X,Y,numTimeSteps] = transformSequences(XTrain(idx),YTrain(idx));
        
        % Convert to dlarray.
        dlX = dlarray(X);
        
        % If training on a GPU, convert data to gpuArray.
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            dlX = gpuArray(dlX);
        end
        
        % Evaluate the model gradients and loss using dlfeval.
        [gradients, loss] = dlfeval(@modelGradients,dlX,Y,parameters,hyperparameters,numTimeSteps);
        
        % Clip the gradients.
        gradients = dlupdate(@(g) thresholdL2Norm(g,gradientThreshold),gradients);
        
        % Update the network parameters using the Adam optimizer.
        [parameters,trailingAvg,trailingAvgSq] = adamupdate(parameters,gradients, ...
            trailingAvg, trailingAvgSq, iteration, learnRate);
        
    end
    
    % Reduce the learning rate after learnRateDropPeriod epochs
    if mod(epoch,learnRateDropPeriod) == 0
        learnRate = learnRate*learnRateDropFactor;
    end
    

    
%%Test Model
for test_i=1:8

    if test_i==1
        XTest=XTest10;
        YTest=YTest10;
        str_len=str10_len;
    elseif test_i==2
        XTest=XTest11;
        YTest=YTest11;
        str_len=str11_len;
    elseif test_i==3
        XTest=XTest12;
        YTest=YTest12;
        str_len=str12_len;
    elseif test_i==4
        XTest=XTest1313;
        YTest=YTest1313;
        str_len=str1313_len;
    elseif test_i==5
        XTest=XTest14;
        YTest=YTest14;
        str_len=str14_len;
    elseif test_i==6
        XTest=XTest13;
        YTest=YTest13;
        str_len=str13_len;
    elseif test_i==7
        XTest=XTest_va;
        YTest=YTest_va;
        str_len=str_len_v;
    else
        XTest=XTest_te;
        YTest=YTest_te;
        str_len=str_len_t;
    end
    
numObservationsTest = numel(XTest);

[X,Y] = transformSequences(XTest,YTest);
dlXTest = dlarray(X);

doTraining = false;
dlYPred = model(dlXTest,parameters,hyperparameters,doTraining);

YPred = gather(extractdata(dlYPred));

end

end