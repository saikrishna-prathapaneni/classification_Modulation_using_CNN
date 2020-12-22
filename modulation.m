t=0:0.01:10;
fc=3000;
fs=8000;
x= 2*cos(2*pi*t)+3*sin(2*0.02*pi*t);
%k= audioread("sample3.mp3");

modulationTypes = {'amssb','am','amssb-tc','fm'};
numModulationTypes=4;
numFramesPerModType=20;
for i=0:numFramesPerModType
fc=fc+20;
y_fm =modulate(x,fc,fs,modulationTypes{2});
 figure,plot(y_fm);
 axis([0 200 -4 4])
destination='C:\Users\balud\Desktop\matlab_ac_AM_modulation\data\fm\fm';
%print([destination,num2str(i),'.png'],'-dpng');
fileName = sprintf('%d.jpg', i); % Create filename from i.
saveas(gca, fileName)
end

%%
% y_am =modulate(x,fc,fs,modulationTypes{2});
% figure,plot(y_am);
 
 %y_ssbt =modulate(k,fc,fs,am_methods{3});
 %subplot(3,1,3);
 %plot(y_ssbt);
 
 y_fm =modulate(x,fc,fs,modulationTypes{1});
 figure,plot(y_fm);
 
 axis([0 200 -4 4])
 %%
 %creating a data object
inputsize_of_cnn=[224 224 3];
file=fullfile('data');
cat={'am','fm','ssb'};

net = googlenet
inputSize = net.Layers(1).InputSize;

output_number=numel(net.Layers(end).ClassNames);
%%

output_number=numel(net.Layers(end).ClassNames);
if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 
learnableLayer = 'loss3-classifier'
 classLayer='output'

data_set=imageDatastore(fullfile(file,cat),'LabelSource','foldernames');
count=countEachLabel(data_set);
min_value=min(count{:,2});

data_set=splitEachLabel(data_set,min_value,'randomize');
countEachLabel(data_set);
 

 %%
 
 
[train_data,test_data]=splitEachLabel(data_set,0.8,'randomize');

%changing the input data size to required size

augumented_train_data=augmentedImageDatastore(inputsize_of_cnn,train_data);
augumented_test_data=augmentedImageDatastore(inputsize_of_cnn,test_data);
 
 
traininglables=train_data.Labels;
testlables=test_data.Labels;

 
 %%
 numClasses = numel(categories(train_data.Labels));

newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    


lgraph = replaceLayer(lgraph,learnableLayer,newLearnableLayer);


newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer,newClassLayer);

figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
plot(lgraph)
ylim([0,10])

layers = lgraph.Layers;
connections = lgraph.Connections;

%layers(1:10) = freezeWeights(layers(1:10));
lgraph = createLgraphUsingConnections(layers,connections);

scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),train_data, ...
    'DataAugmentation',imageAugmenter);

augimdsValidation = augmentedImageDatastore(inputSize(1:2),test_data);

%%
miniBatchSize = 32;
%%valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(augimdsTrain,lgraph,options);

[YPred,probs] = classify(net,augimdsValidation);
accuracy = mean(YPred == testlables)

 %%
 
 
layers = [
    imageInputLayer([224 224 3],'Name','input')...
    
    convolution2dLayer(5,16,'Padding','same','Name','conv_1')...
    batchNormalizationLayer('Name','BN_1')...
    reluLayer('Name','relu_1')...
 
    convolution2dLayer(3,32,'Padding','same','Stride',2,'Name','conv_2')...
    batchNormalizationLayer('Name','BN_2')...
    reluLayer('Name','relu_2')...
    convolution2dLayer(3,32,'Padding','same','Name','conv_3')...
    batchNormalizationLayer('Name','BN_3')...
    reluLayer('Name','relu_3')...
    
    additionLayer(2,'Name','add')...
    
    averagePooling2dLayer(2,'Stride',2,'Name','avpool')...
    fullyConnectedLayer(10,'Name','fc')...
    softmaxLayer('Name','softmax')...
    classificationLayer('Name','classOutput')];


lgraph = layerGraph(layers);
skipConv = convolution2dLayer(1,32,'Stride',2,'Name','skipConv');
lgraph = addLayers(lgraph,skipConv);
lgraph = connectLayers(lgraph,'relu_1','skipConv');
lgraph = connectLayers(lgraph,'skipConv','add/in2');
figure
plot(lgraph);

 options = trainingOptions('sgdm', ...
    'MaxEpochs',8, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{augumented_test_data,testlables}, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');
%%

net_1 = trainNetwork(augumented_train_data,traininglables,lgraph,options);

%plotconfusion(testlables,predict_labels)


%%
layers = [ ...
    imageInputLayer([224 224 3])...
    convolution2dLayer(5,20)...
    reluLayer...
    maxPooling2dLayer(2,'Stride',2)...
    fullyConnectedLayer(10)...
    softmaxLayer...
    classificationLayer];


options = trainingOptions('sgdm', ...
    'MaxEpochs',10,...
    'InitialLearnRate',1e-4, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(augumented_train_data,traininglables,layerGraph(layers),options);

 