function k = Ke_Beam(E,I,L,A)
% This function returns the element stiffness matrix for a beam   
% element with modulus of elasticity E,  
% moment of inertia I, and length L.
% The size of the element stiffness 
% matrix is 6 x 6.
k = [E*A/L , 0          , 0         , -E*A/L, 0          , 0         ;
     0     , 12*E*I/L^3 , 6*E*I/L^2 , 0     , -12*E*I/L^3, 6*E*I/L^2 ;
     0     , 6*E*I/L^2  , 4*E*I/L   , 0     , -6*E*I/L^2 , 2*E*I/L   ;
     -E*A/L, 0          , 0         , E*A/L , 0          , 0         ;
     0     , -12*E*I/L^3, -6*E*I/L^2, 0     , 12*E*I/L^3 , -6*E*I/L^2;
     0     , 6*E*I/L^2  , 2*E*I/L   , 0     , -6*E*I/L^2 , 4*E*I/L]  ;





