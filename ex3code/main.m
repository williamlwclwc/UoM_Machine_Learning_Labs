close all;
clear all;
fname = input('Enter a filename to load data for training/testing: ','s');
load(fname);

% Put your NB training function below
%[ "Parameter List" ] = NBTrain(AttributeSet, LabelSet); % NB training
% Put your NB test function below
%[predictLabel, accuracy] = NBTest( "Parameter List" , testAttributeSet, validLabel); % NB test

fprintf('********************************************** \n');
fprintf('Overall Accuracy on Dataset %s: %f \n', fname, accuracy);
fprintf('********************************************** \n');