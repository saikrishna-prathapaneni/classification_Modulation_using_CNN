inputsize_of_cnn=[224 224 3];
file=fullfile('data');
cat={'am','fm','ssb'};


data_set=imageDatastore(fullfile(file,cat),'LabelSource','foldernames');
count=countEachLabel(data_set);
min_value=min(count{:,2});

data_set=splitEachLabel(data_set,min_value,'randomize');
countEachLabel(data_set);


net = resnet50
%net = resnet50()
%lgraph = resnet50('Weights','none')
%plot(net);
%set(gca,'YLim',[150 170]);

%know the no. of nodes in input layer and number of nodes in the output
%layer
%analyzeNetwork(net)
net.Layers(1);
output_number=numel(net.Layers(end).ClassNames);
inputsize_of_cnn=net.Layers(1).InputSize;

[train_data,test_data]=splitEachLabel(data_set,0.8,'randomize');

%changing the input data size to required size

augumented_train_data=augmentedImageDatastore(inputsize_of_cnn,train_data);
augumented_test_data=augmentedImageDatastore(inputsize_of_cnn,test_data);

%know the weights of the 1st layer 
weight_1=net.Layers(2).Weights;
weight_1=mat2gray(weight_1);
%figure
%montage(weight_1);


%%
try
    nnet.internal.cnngpu.reluForward(1);
catch ME
end


%obtaining features from the one of the layers of neural network
featurelayer = 'fc1000';
training_features=activations(net,augumented_train_data...
    ,featurelayer,'MiniBatchSize',30,'OutputAs','columns');
training_features=transpose(training_features);
%obtain labels from the train_data
traininglables=train_data.Labels;
testlables=test_data.Labels;



%create the classifier for the present training features
classifier=fitcecoc(training_features,traininglables...
    ,'Learner','Linear','Coding','onevsall');

%%
%time for the prediction(how well the data model is performed over the test data)
test_features=activations(net,augumented_test_data...
    ,featurelayer,'MiniBatchSize',32,'OutputAs','columns');     
test_features=transpose(test_features);



predict_labels=predict(classifier,test_features);    

testlables=test_data.Labels;
confmat=confusionmat(testlables,predict_labels);


%know the percentage of the accuracy of the trained data
k=bsxfun(@rdivide,confmat,sum(confmat,2));
mean(diag(k))
%%
img = imread(fullfile('16.jpg'));
 
 test=augmentedImageDatastore(inputsize_of_cnn,img,'ColorPreprocessing','gray2rgb');
 testk=activations(net,test...
     ,featurelayer,'MiniBatchSize',32,'OutputAs','columns');     
 testk=transpose(testk);
 imshow(img)
 label=predict(classifier,testk);
 title(string(label));
 %%
 idx = randperm(numel(test_data.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(test_data,idx(i));
    imshow(I)
    label = predict(classifier,testk);
    title(string(label) + ", " + num2str(100*max(probs(idx(i),:)),3) + "%");
end
