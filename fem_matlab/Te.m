function y = Te(alpha)
% matrix for coordinate transformation
x = alpha/180*pi;
y = [cos(x),sin(x),0,0,0,0;
    -sin(x),cos(x),0,0,0,0;
    0,0,1,0,0,0;
    0,0,0,cos(x),sin(x),0;
    0,0,0,-sin(x),cos(x),0;
    0,0,0,0,0,1];
end