function y = Me_Beam(lineRou,L,m)
%BeamElementMass   This function returns the element
%                       mass matrix for a beam
%                       element with distributed mass Rou
%                       and length L.
%                       The size of the element stiffness
%                       matrix is 6 x 6.
% lineRou:
% L:
% m: additional wieght on the node
A = [m/2,0,0,0,0,0;
    0,m/2,0,0,0,0;
    0,0,0,0,0,0;
    0,0,0,m/2,0,0;
    0,0,0,0,m/2,0;
    0,0,0,0,0,0];
B = lineRou*L/420*...
    [140,0,0,70,0,0;
    0,156,22*L,0,54,-13*L;
    0,22*L,4*L^2,0,13*L,-3*L^2;
    70,0,0,140,0,0;
    0,54,13*L,0,156,-22*L;
    0,-13*L,-3*L^2,0,-22*L,4*L^2];
y = A+B;
end