load ORLfacedata;
%X = data(1:20,:);
%ShowFace(X,5);

X = data([1:10,31:40,51:60],:);
Y = labels([1:10,31:40,51:60]);
[Xtr,Xte,Ytr,Yte] = PartitionData(X,Y,3);
figure(1); ShowFace(Xtr,3);
figure(2); ShowFace(Xte,7);
