function Z=NewmarkConstantAverageAccleration(K,M,C,P,Z0,deltaT,T)
% Newmark-常加速度法
% Z：计算得到的节点自由度位移/速度/加速度
% node：节点编号
% element：单元节点号
% K/M/C：刚度/质量/阻尼矩阵
% P：荷载向量
% Z0：初始位移矩阵，第1/2/3列向量依次是初始位移/速度/加速度
% deltaT：时间步长
% T：总计算时间
gamma=1/2;beta=1/4;
N=T/deltaT+1;
dof_n=length(M(:,1));
Z=cell(1,3);
Z{1}=zeros(dof_n,N);   %位移
Z{2}=zeros(dof_n,N);   %速度
Z{3}=zeros(dof_n,N);   %加速度
Z{1}(:,1)=Z0(:,1);   %初始自由度位移
Z{2}(:,1)=Z0(:,2);  %初始自由度速度
Z{3}(:,1)=Z0(:,3);  %初始自由度加速度
a0=1/beta/deltaT^2;
a1=gamma/beta/deltaT;
a2=1/beta/deltaT;
a3=1/2/beta-1;
a4=gamma/beta-1;
a5=deltaT/2*(gamma/beta-2);
a6=deltaT*(1-gamma);
a7=gamma*deltaT;
K_eq=K+a0*M+a1*C;
for i=1:N-1
    P_eq(:,i+1)=P(:,i+1)+M*(a0*Z{1}(:,i)+a2*Z{2}(:,i)+...
        a3*Z{3}(:,i))+C*(a1*Z{1}(:,i)+a4*Z{2}(:,i)+a5*Z{3}(:,i));
    Z{1}(:,i+1)=K_eq\P_eq(:,i+1);
    Z{3}(:,i+1)=a0*(Z{1}(:,i+1)-Z{1}(:,i))-a2*Z{2}(:,i)-a3*Z{3}(:,i);
    Z{2}(:,i+1)=Z{2}(:,i)+a6*Z{3}(:,i)+a7*Z{3}(:,i+1);
end
end