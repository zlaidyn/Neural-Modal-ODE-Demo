function y=insert(A,n)
%����һ��Ϊ�������ָ���еĺ���
%   A��ʾ������ľ���n��ʾҪ�����������
for k=1:1:n-1
    M(k,:)=A(k,:);
end
for k=n+1:1:(size(A,1)+1)
     M(k,:)=A(k-1,:);
end
y=M;
end

