function dlY = model(dlX,parameters,hyperparameters,doTraining)

numBlocks = hyperparameters.NumBlocks;
dropoutFactor = hyperparameters.DropoutFactor;

dlY = dlX;
% Residual blocks.
for k = 1:numBlocks
    dilationFactor = 2^(k-1);
    parametersBlock = parameters.("Block"+k);
    
    dlY = residualBlock(dlY,dilationFactor,dropoutFactor,parametersBlock,doTraining);
    if k==1
        dlY0=dlY;
    else
        dlY0=[dlY;dlY0];
    end

end

dlY=dlY0;

% Fully connect.
weights = parameters.FC.Weights;
bias = parameters.FC.Bias;
weights20 = parameters.FC.Weights20;
bias20 = parameters.FC.Bias20;

dlY = fullyconnect(dlY,weights20,bias20,'DataFormat','CBT');
dlY = fullyconnect(dlY,weights,bias,'DataFormat','CBT');

end