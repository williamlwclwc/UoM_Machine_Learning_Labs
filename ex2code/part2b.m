load ORLfacedata
%extract images for all subjects
X = data;
Y = labels;
Xtr = zeros(200, 1024);
Xte = zeros(200, 1024);
Ytr = zeros(200);
Yte = zeros(200);
Y_mul_tr = zeros(200, 40);
Y_mul_te = zeros(200, 40);
avr_te = 0;
w_te = 0;
accuracy_te_linear = zeros(50, 1);
accuracy_te_knn = zeros(50, 1);
for i = 1: 50
    [Xtr, Xte, Ytr, Yte] = PartitionData(X, Y, 5);
    %make the 1 of k matrix
    for j = 1: 200
        Y_mul_tr(j, Ytr(j)) = 1;
        Y_mul_te(j, Yte(j)) = 1;
    end
    %normal equations to get predictions
    X_train = [ones(size(Xtr, 1), 1), Xtr];
    w = pinv(X_train) * Y_mul_tr;
    X_test = [ones(size(Xte, 1), 1), Xte];
    yhat = w' * X_test';
    c_te = 0;
    for j = 1: 200
        %choose the most likely one and compare with the ground truth
        [M, index] = max(yhat(:, j));
        if index == Yte(j)
            c_te = c_te + 1;
        else
            w_te = w_te + 1;
            err_linear(w_te) = Yte(j);
        end
    end
    accuracy_te_linear(i) = c_te / 200.0;
    %fprintf('testing accuracy = %f\n', accuracy_te(i))
    avr_te = avr_te + accuracy_te_linear(i);   
end
avr_te = avr_te / 50.0;
std_te = std(accuracy_te_linear);
fprintf('average testing accuracy = %f, standard deviation = %f\n', avr_te, std_te);
tabulate(err_linear);

k = 1;
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
            err_knn(w_te) = Yte(j);
        end
    end
    accuracy_te_knn(i) = c_te / 200.0;
    avr_te = avr_te + accuracy_te_knn(i);
end
avr_te = avr_te / 50.0;
std_te = std(accuracy_te_knn);
fprintf('k = %i, average testing accuracy = %f, standard deviation = %f\n', k, avr_te, std_te)
tabulate(err_knn);

%draw figure
figure(1);
x = 1:50;
y1 = accuracy_te_linear(x);
y2 = accuracy_te_knn(x);
plot(x, y1, 'r'); %linear in red
hold on;
plot(x, y2, 'b'); %knn in blue
hold off;
xlabel('x: Dataset Number');
ylabel('y: Testing Accuracy');
title('Testing Accuracy Comparison');