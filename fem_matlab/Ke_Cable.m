function cablek = Ke_Cable(E,A,L)
% stiffness matrix of a cable element (truss element)
% E:
% A:
% L: element length
cablek = E*A/L*[1 ,0,0,-1,0,0;
               0 ,0,0,0 ,0,0;
               0 ,0,0,0 ,0,0;
               -1,0,0,1 ,0,0;
               0 ,0,0,0 ,0,0;
               0 ,0,0,0 ,0,0];
end 
