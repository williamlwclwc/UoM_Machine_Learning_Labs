%  Naive Bayes Training model
%
%  Y = NBTrain(AttributeSet, LabelSet)
%
%  Arguments:
%  'AttributeSet' a M*57 matrix where each row represents the feature vector of an training example,
% M is the number of training examples
%
%  'LabelSet' should be a M*1 column vector, where elements represent the labels of corresponding to
% training examples in AttributeSet
%
% Return values: probability of each value of each feature and probability for each categories

function [y1, y2] = NBTrain(AttributeSet, LabelSet)

    % number of training samples
    trainNum = length(LabelSet);
    % number of features
    fNum = length(AttributeSet(1, :));
    % number of each features' possible values
    aNum = max(AttributeSet(:)) + 1;
    % number of categories, 1 for spam, 2 for not spam...
    cNum = max(LabelSet) + 1; 
    total = zeros(cNum, 1); % number of instances of each class
    p_total = zeros(cNum, 1); % probability of each class
    training = zeros(aNum, cNum, fNum); % count for each class, attribute, attribute value
    p_train = zeros(aNum, cNum, fNum); % probability for each class, attribute, attribute value
    if round(AttributeSet) - AttributeSet == 0
        % discrete valued features
        % do the counting stuff
        for i = 1: trainNum
            for c = 1: cNum
                if LabelSet(i) == c - 1
                    total(c) = total(c) + 1;
                    for j = 1: fNum
                        training(AttributeSet(i, j) + 1, c, j) = training(AttributeSet(i, j) + 1, c, j) + 1;
                    end
                end
            end
        end
        % calculate the probability based on counts
        for c = 1: cNum
            p_total(c) = total(c) / sum(total);
            for j = 1: fNum
                for k = 1: aNum
                    p_train(k, c, j) = (1/aNum + training(k, c, j)) / (total(c) + 1);
                end
            end
        end
        y1 = p_train;
    else
        % continuous valued features
        for i = 1: trainNum
            for c = 1: cNum
                if LabelSet(i) == c - 1
                    total(c) = total(c) + 1;
                end
            end
        end
        featureSet = zeros(cNum, trainNum, fNum);
        featureSet(featureSet == 0) = -1;
        index = ones(cNum, 1);
        avr_std = zeros(2, cNum, fNum);
        % divide attributeset into n parts(n is the number of classes)
        for i = 1: trainNum
            featureSet(LabelSet(i) + 1, index(LabelSet(i) + 1), :) = AttributeSet(i, :);
            index(LabelSet(i) + 1) = index(LabelSet(i) + 1) + 1;
        end
        % calculate mean&std for each class, attribute
        for c = 1: cNum
            for j = 1: fNum
                temp = featureSet(c, :, j);
                avr_std(1, c, j) = mean(temp(temp ~= -1));
                avr_std(2, c, j) = std(temp(temp ~= -1));
            end
        end
        for c = 1: cNum
            p_total(c) = total(c) / sum(total);
        end
        y1 = avr_std; 
    end
    y2 = p_total;
    