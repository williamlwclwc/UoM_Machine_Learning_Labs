close all;
clear all;
fname = input('Input the filename of the dataset:', 's');
load(fname);

% Put your NB training function below
[probability, p_total] = NBTrain(AttributeSet, LabelSet); % NB training
% Put your NB test function below
[predictLabel, accuracy] = NBTest(probability, p_total, testAttributeSet, validLabel); % NB test

fprintf('********************************************** \n');
fprintf('Overall Accuracy on Dataset %s: %f \n', fname, accuracy);
fprintf('********************************************** \n');