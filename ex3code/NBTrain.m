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
    total = zeros(cNum, 1);
    p_total = zeros(cNum, 1);
    training = zeros(cNum, fNum, aNum);
    probability = zeros(aNum, cNum, fNum);
    for i = 1: trainNum
        for c = 1: cNum
            if LabelSet(i) == c - 1
                total(c) = total(c) + 1;
                for j = 1: fNum
                    for k = 1: aNum
                        if AttributeSet(i, j) == k - 1
                            training(c, j, k) = training(c, j, k) + 1;
                            break;
                        end
                    end
                end
            end
        end
    end
    for c = 1: cNum
        p_total(c) = total(c) / sum(total);
        for j = 1: fNum
            for k = 1: aNum
                probability(k, c, j) = (1/aNum + training(c, j, k)) / (total(c) + 1);
            end
        end
    end
    y1 = probability; 
    y2 = p_total;
    