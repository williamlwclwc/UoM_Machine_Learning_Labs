close all;
clear all;
fname = input('Input the filename of the dataset:', 's');
load(fname);

if strcmp(fname, 'spambase.data') == 0
    % Put your NB training function below
    [probability, p_total] = NBTrain(AttributeSet, LabelSet); % NB training
    % Put your NB test function below
    [predictLabel, accuracy] = NBTest(probability, p_total, testAttributeSet, validLabel); % NB test
    c_matrix = confusionmat(predictLabel, validLabel)
    if max(LabelSet) + 1 == 2
        plotconfusion(validLabel',predictLabel');
    end
else
    % 10 fold cross validation for spambase.data
    acc = zeros(10, 1);
    % random the dataset
    r = randperm(size(spambase,1));
    spambase = spambase(r, :);
    % extract label data from the dataset
    column_num = length(spambase(1, :));                 
    Label = spambase(:, column_num);
    spambase(:, column_num) = [];
    % 10 fold cross validation
    for i = 1: 10
        AttributeSet = spambase;
        LabelSet = Label;
        testAttributeSet = spambase((i-1)*460+1: i*460, :);
        validLabel = Label((i-1)*460+1: i*460);
        AttributeSet((i-1)*460+1: i*460, :) = [];
        LabelSet((i-1)*460+1: i*460) = [];
        [probability, p_total] = NBTrain(AttributeSet, LabelSet);
        [predictLabel, accuracy] = NBTest(probability, p_total, testAttributeSet, validLabel);
        fprintf('Fold No.%i, average testing accuracy = %f\n', i, accuracy);
        acc(i) = accuracy;
        c_matrix = confusionmat(predictLabel, validLabel)
    end
    accuracy = mean(acc);
    std_acc = std(acc);
    fprintf('10-Fold standard deviation of accuracy = %f\n', std_acc);
    % draw the graph
    figure(1);
    x = 1: 10;
    y = acc;
    plot(x, y, 'r');
    xlabel('x: Number of folds');
    ylabel('y: Test Accuracy');
    title('10-Fold Testing Accuracy');
end

fprintf('********************************************** \n');
fprintf('Overall Accuracy on Dataset %s: %f \n', fname, accuracy);
fprintf('********************************************** \n');