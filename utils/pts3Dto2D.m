function pts2D = pts3Dto2D(pts3D,R,t,s)
rotpts = R*pts3D';
pts2D =[s.*(rotpts(1,:)+t(1)); s.*(rotpts(2,:)+t(2))];
