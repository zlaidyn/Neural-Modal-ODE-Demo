function m = Me_Cable(lineRou,l)
% mass matrix of a cable element
% lineRou: density per meter (kg/m);
% l: length of the element
m = lineRou*l/4*[2,0,1,0;0,2,0,1;1,0,2,0;0,1,0,2];
end