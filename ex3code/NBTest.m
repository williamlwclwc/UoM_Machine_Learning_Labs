%  Naive Bayes Testing model
%
%  Y = NBTest(p_train, p_total, testAttributeSet, validLabel)
%
%  Arguments:
%  'p_train' a M*C*57 matrix from NBTrain, representing the probability table 
% M is the number of training examples, C is the number of classes
%
%  'p_total' an array length of the number of classes, representing the overall probability of each classes 
% 
%  'testAttributeSet' a N*57 matrix where each row represent the number of test instances.
% N is the number of test instances
%
% ‘validLabel' a N*1 column vector where elements represent the labels corresponding to test instances in
% testAttributeSet (used to achieve the accuracy of a classifier)
%
% Return value: predicted label set and testing accuracy

function [y1, y2] = NBTest(p_train, p_total, testAttributeSet, validLabel)
    testNum = length(validLabel);
    fNum = length(testAttributeSet(1, :));
    cNum = max(validLabel) + 1;
    aNum = max(testAttributeSet(:)) + 1;
    correct = 0;
    predicted = zeros(testNum, 1);
    parameter = 1.1e-40;
    if round(testAttributeSet) - testAttributeSet == 0
        flag = 0; % discrete
    else 
        flag = 1; % continuous
    end
    for i = 1: testNum
        probability = ones(cNum, 1);
        for c = 1: cNum
            for j = 1: fNum
                if flag == 0
                    % multiply the probabilty of this attribute value (discrete)
                    probability(c) = probability(c) * p_train(testAttributeSet(i, j) + 1, c, j);
                else
                    if p_train(2, c, j) ~= 0
                        % if std is not 0, use normal distribution to calculate probability
                        probability(c) = probability(c) * normpdf(testAttributeSet(i, j), p_train(1, c, j) , p_train(2, c, j));
                    else
                        if testAttributeSet(i, j) == p_train(1, c, j)
                            % if its a common value, it should be a high probability
                            probability(c) = probability(c) * 1;
                        else
                            % if its a not common value, it should be a small probability
                            probability(c) = probability(c) * parameter;
                        end
                    end
                end
            end
            probability(c) = probability(c) * p_total(c);
        end
        [value, index] = max(probability);
        if index - 1 == validLabel(i)
            correct = correct + 1;
        end
        predicted(i) = index - 1;
    end
    accuracy = correct / testNum;
    y1 = predicted;
    y2 = accuracy;

