load ORLfacedata
%extract images for all subjects
X = data;
Y = labels;
Xtr = zeros(200, 1024);
Xte = zeros(200, 1024);
Ytr = zeros(200);
Yte = zeros(200);
avr_te = 0;
std_te = 0;
accuracy_te = zeros(50, 1);
accuracy_loo = zeros(400, 1);
% leave one out method to determine k
for k = 1: 399
    c_te = 0;
    for i = 1: 400
        XX = X;
        YY = Y;
        XX(i, :) = [];
        YY(i) = [];
        predict_loo = knearest(k, X(i, :), XX, YY);
        if predict_loo == Y(i)
            c_te = c_te + 1;
        end
    end
    accuracy_loo(k) = c_te / 400.0;
    fprintf('k = %i, average testing accuracy(loo) = %f\n', k, accuracy_loo(k))
end
[best_loo, k] = max(accuracy_loo);
fprintf('best choice of k is %i, with average accuracy: %f\n', k, best_loo)
% k = 1;
w_te = 0;
for i = 1: 50
    %training set containing 5*40 examples
    [Xtr, Xte, Ytr, Yte] = PartitionData(X, Y, 5);
    predict_te = zeros(200, 1);
    c_te = 0;
    for j = 1: 200
        predict_te(j) = knearest(k, Xte(j, :), Xtr, Ytr);
        if predict_te(j) == Yte(j)
            c_te = c_te + 1;
        else
            w_te = w_te + 1;
            err(w_te) = Yte(j);
        end
    end
    accuracy_te(i) = c_te / 200.0;
    avr_te = avr_te + accuracy_te(i);
end
avr_te = avr_te / 50.0;
std_te = std(accuracy_te);
fprintf('k = %i, average testing accuracy = %f, standard deviation = %f\n', k, avr_te, std_te)
%figure(1); ShowResult(Xte, Yte, predict_te, 10);
fprintf('Statistics of subjects, error counts and error rates:\n');
tabulate(err);