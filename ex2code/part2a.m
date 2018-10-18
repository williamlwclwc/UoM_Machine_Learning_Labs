load ORLfacedata;
%extract images for subjects 1 & 30
X = data([1:10, 291:300], :);
Y = labels([1:10, 291:300]);
Xtr = zeros(6, 1024);
Xte = zeros(14, 1024);
Ytr = zeros(6);
Yte = zeros(14);
accuracy_te = zeros(50, 1);
avr_te = 0;
for i = 1: 50
    [Xtr, Xte, Ytr, Yte] = PartitionData(X, Y, 3);
    X_train = [ones(size(Xtr, 1), 1), Xtr];
    w = pinv(X_train) * Ytr;
    X_test = [ones(size(Xte, 1), 1), Xte];
    yhat = w' * X_test';
    c_te = 0;
    for j = 1: 14
        if yhat(j) >= 15.5
            yhat(j) = 30;
        else
            yhat(j) = 1;
        end
        if yhat(j) == Yte(j)
            c_te = c_te + 1;
        end
    end
    accuracy_te(i) = c_te / 14.0;
    fprintf('testing accuracy = %f\n', accuracy_te(i))
    avr_te = avr_te + accuracy_te(i);
    %show result for specific dataset
    if i == 1 
        figure(2); ShowResult(Xte, Yte, yhat, 4);
    end
end
avr_te = avr_te / 50.0;
std_te = std(accuracy_te);
fprintf('average testing accuracy = %f, standard deviation = %f\n', avr_te, std_te);

%draw errorbar graph
figure(1);
x = 1:50;
y = accuracy_te(x);
plot(x, y);
xlabel('x: Dataset Number');
ylabel('y: Testing Accuracy');
title('Testing Accuracy');
