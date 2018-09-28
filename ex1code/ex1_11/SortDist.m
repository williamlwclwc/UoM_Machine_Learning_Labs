%SortDist
function result = SortDist(x,A)
for i = 1:10
    dis(i,1:3) = [sqrt((x(1,1)-A(i,1))^2 + (x(1,2)-A(i,2))^2),A(i,1),A(i,2)];
end
result = sortrows(dis,1);
result = result(1:3,:);



