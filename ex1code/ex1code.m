%ex1
ex1 = factorial(15)
%ex2
A = [16 3 2 13; 5 10 11 8; 9 6 7 12; 4 15 14 1]
B = eye(4)
%ex3
ex3 = sum(A(:, 3))
%ex4
ex4 = A(:, 1:2:3)
%ex5
ex5_1 = A(:,randperm(4, 2))
ex5_2 = A(randperm(4, 3),:)
%ex6
column = sum(A)
row = sum(A')'
%row = sum(A,2); %maybe better without '
%diagonal_1 = sum(diag(A))
%diagonal_2 = A(1,4)+A(2,3)+A(3,2)+A(4,1)
sort_row = sort(A')'
%sort_row = sort(A,2); %maybe better without '
%ex7
ex7_1 = A * B
ex7_2 = A.* B
%ex8
ex8 = sum(sum(A>10))
%ex9
x = 1:100;
y = log(x);
plot(x, y, 'r.')
xlabel('x:1 to 100');
ylabel('y:log(x)');
%ex10
B = zeros(4, 4);
for i = 1:4
    for j = 1:4
        B(i,j) = 1/A(i,j);
    end
end
B
sum(B)
sum(1./A)






