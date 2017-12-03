function e = facecenter( V,F )
%FACENORMALS Compute face normals of triangular mesh
%   V - nverts by 3 matrix containing vertex positions
%   F - ntri by 3 matrix containing triangle vertex indices

% Get the triangle vertices
v1      = F(:, 1);
v2      = F(:, 2);
v3      = F(:, 3);

% Compute the edge vectors
e = (V(v1, :) + V(v1, :) + V(v3,:))/3;

end

