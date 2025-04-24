function y=insert(A,n)
%这是一个为矩阵插入指定行的函数
%   A表示待插入的矩阵，n表示要插入的行数。
for k=1:1:n-1
    M(k,:)=A(k,:);
end
for k=n+1:1:(size(A,1)+1)
     M(k,:)=A(k-1,:);
end
y=M;
end

