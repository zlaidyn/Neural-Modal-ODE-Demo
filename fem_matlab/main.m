%% basic parameters
clc
clear

% beam information
Rou_ai = 2799.673269;  % density of aluminum alloy
E_ai = 7.2e10; % elastic modulus of aluminum alloy
a1 = 0.015;   % dimension of the beam section
t1 = 0.0012;    % thickness of the beam section
BeamA = a1^2-(a1-2*t1)^2;
BeamI = a1^4/12-(a1-2*t1)^4/12;
BeamL = [0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.3,0.2,0.2,0.3,0.2,0.2,0.2,0.2,...
    0.3,0.2,0.2,0.3,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2];
BeamM = 0.5;  % 0.5 is the additional weight

% tower information
TowerL = 0.1;
a2 = 0.015;                  % dimension of the tower section
t2 = 0.002;                 % thickness of the tower section
TowerA = a2^2-(a2-2*t2)^2;
TowerI = a2^4/12-(a2-2*t2)^4/12;
for k = 1:12
    if k <= 8
        TowerM(k,1) = 0.4; % 0.4 is the additional weight on lower tower nodes
    else
        TowerM(k,1) = 0.2; % 0.2 is the additional weight on higer tower nodes
    end
end

% cable information
d1 = 0.00022;         % diameter of outter cables
d2 = 0.00016;         % diameter of inner cables
Rou_steel = 8015.111293;    % density of steel
E_steel = 1.9e11;           % elastic modulus of steel
CableAngle = [-33.6901;-45;180-139.3987;180-150.2551;
    -29.7449;-40.6013;180-135;180-146.3099]*-1;  % counter-clockwise positive (important! otherwize the direction of tower deformation is reversed), from the leftmost cable to rightmost cable
CableA = [d1^2*pi/4;d2^2*pi/4;d2^2*pi/4;d1^2*pi/4*1;
    d1^2*pi/4;d2^2*pi/4;d2^2*pi/4;d1^2*pi/4];
%% FEM
% element information
element(1:28,:)=[1:28;2:29]'; % beam elements
element(29:40,:)=[30:41;31:42]'; % left tower elements
element(41:52,:)=[43:54;44:55]'; % right tower elements
element(53:60,:)=[1,40;4,38;38,10;40,13;17,53;20,51;51,26;53,29]; % cable elements (from left to right)

% node information(column 1: X; column 2: Y)
node(1,:)=[-3,0]; % nodes of beam
for i = 2:29
    node(i,1) = node(i-1,1)+BeamL(i-1);
    node(i,2) = 0;
end
node(30:42,:)=[-1.8*ones(1,13);-0.2:TowerL:1.0]'; % nodes of left tower
node(43:55,:)=[+1.8*ones(1,13);-0.2:TowerL:1.0]'; % nodes of right tower

% degree-of-freedom information
dof_n = length(node)*3;

% beam elements
element_beam_N = 28;
for i = 1:element_beam_N
   K_element{i} = Ke_Beam(E_ai,BeamI,BeamL(i),BeamA);
   T_element{i} = Te(0);
   M_element{i} = Me(Rou_ai*BeamA,BeamL(i),BeamM);
end

% tower elements
element_tower_N = 12;
count = 1;
for i = element_beam_N+1:element_beam_N+element_tower_N
   K_element{i} = Ke_Beam(E_ai,TowerI,TowerL,TowerA);
   T_element{i} = Te(90);
   M_element{i} = Me(Rou_ai*TowerA,TowerL,TowerM(count));
   count = count+1;
end % left tower
count = 1;
for i = element_beam_N+element_tower_N+1:element_beam_N+element_tower_N*2
   K_element{i} = Ke_Beam(E_ai,TowerI,TowerL,TowerA);
   T_element{i} = Te(90);
   M_element{i} = Me(Rou_ai*TowerA,TowerL,TowerM(count));
   count = count+1;
end % right tower

% cable elements
element_cable_N = 8;
count = 1;
for i = element_beam_N+element_tower_N*2+1:length(element)
    x1 = node(element(i,1),1);
    x2 = node(element(i,2),1);
    y1 = node(element(i,1),2);
    y2 = node(element(i,2),2);
    CableL(count) = sqrt((x1-x2)^2+(y1-y2)^2);
    K_element{i} = Ke_Cable(E_steel,CableA(count),CableL(count));
    T_element{i} = Te(CableAngle(count));
    M_element{i} = Me(Rou_steel*CableA(count),CableL(count),0);
    count = count+1;
end

% assemble global matrix
K = zeros(dof_n);
M = zeros(dof_n);
for i = 1:length(element)
    K_element_transformed{i} = T_element{i}'*K_element{i}*T_element{i};
    K = assemble(K,K_element_transformed{i},element(i,1),element(i,2));
    M = assemble(M,M_element{i},element(i,1),element(i,2));
end


% apply constrains
K_con = K;
M_con = M;
Constrained_Dof=[1*3-1,4*3-1,7*3-1,23*3-1,26*3-1,29*3-1,...
    30*3-2,30*3-1,30*3-0,...
    43*3-2,43*3-1,43*3-0];
p=0;  % an indicator
for i=1:1:length(Constrained_Dof)
    SubDof=Constrained_Dof(i)-p;
    K_con = K_con([1:(SubDof-1),(SubDof+1):(dof_n-p)],[1:(SubDof-1),(SubDof+1):(dof_n-p)]); % delete constrained dof
    M_con = M_con([1:(SubDof-1),(SubDof+1):(dof_n-p)],[1:(SubDof-1),(SubDof+1):(dof_n-p)]); % delete constrained dof
    p=p+1;
end

%% modal analysis
order1 = 10;
[phis,lamda] = eigs(K_con,M_con,order1,'sm');
[lamda,sort_index] = sort(diag(lamda));
f = sqrt(lamda)/2/pi;
phi = zeros(length(K_con),order1);
for i = 1:order1
    phi(:,i) = phis(:,sort_index(i));
end
for k=1:1:length(Constrained_Dof)
    phi=insert(phi,Constrained_Dof(k));
end
% damping matrix
omega1 = f(1)*2*pi;omega2 = f(order1)*2*pi;
ksi1 = 0.02;ksi2 = 0.02;     % damping ratio
[a0,a1] = DampRayleigh(omega1,omega2,ksi1,ksi2);
C_con = a0*M_con+a1*K_con;

% plot FEM mode shapes
close all
fontsize = 10.5;
figure
for i=1:6
    subplot(3,2,i)
    axis equal
%     phi(:,i)=phi(:,i)/norm/2;  % maximum normalize
    patch('Faces',element,...
        'Vertices',[node(:,1)+phi(1:3:end,i),node(:,2)+phi(2:3:end,i)],...
        'FaceColor','white',...
        'EdgeColor','red');  % plot mode shape
    hold on
    patch('Faces',element,...
        'Vertices',[node(:,1),node(:,2)],...
        'FaceColor','white',...
        'EdgeColor','k',...
        'LineStyle',':');  % plot undeformed FEM
    set(gca,...
        'FontName', 'Times New Roman', ...
        'FontSize', fontsize,...
        'Xlim',[-3.5,+3.5],...
        'Ylim',[-0.6,1.5],...
        'Box','On')
    xlabel( 'X(m)', 'FontName', 'Times New Roman', 'FontSize', fontsize);
    ylabel( 'Y(m)', 'FontName', 'Times New Roman', 'FontSize', fontsize);
    title(['Mode ',num2str(i),', Frequency = ',num2str(f(i)),' Hz'],'FontName','Times New Roman', 'FontSize', fontsize);
%     legend('Modeshape','FEM','Location','N')
end

% save('node.txt','node','-ascii')
% save('phi.txt','phi','-ascii')
% save('f.txt','f','-ascii')
% save('element.txt','element','-ascii')
% save('M.txt','M_con','-ascii')
% save('C.txt','C_con','-ascii')
% save('K.txt','K_con','-ascii')