function normals = vertex_normals(FV,R,t,s)
Rr = R;
Rr(4,4)=1;
Sr = eye(4).*s;
Tr = eye(4);
Tr(1:2,4)=t;
T = Tr*Sr*Rr;
clear Tr Sr Rr
% Get the extrinsic transformation matrix
M = T(1: 3, :); % the output does not need to be in homogeneous coordinates

% Get the vertices
V           = FV.vertices;
Nvertices   = size(FV.vertices, 1);

% Compute the transformed vertices
V(:, 4)	= 1;        % use homogeneous coordinates for input
V2   	= V * M.';	% the vertices are transposed

v1      = FV.faces(:, 1);
v2      = FV.faces(:, 2);
v3      = FV.faces(:, 3);

% Get the vertices to render
f       = 1:length(FV.faces);
v       = unique([v1(f); v2(f); v3(f)]);
f       = find(any(ismember(FV.faces, v), 2));
Nfaces  = length(f);
 

% Compute the edge vectors
e1s = V2(v2(f), :) - V2(v1(f), :);
e2s = V2(v3(f), :) - V2(v1(f), :);
e3s = V2(v2(f), :) - V2(v3(f), :);

clear V2

% Normalize the edge vectors
e1s_norm = e1s ./ repmat(sqrt(sum(e1s.^2, 2)), 1, 3);
e2s_norm = e2s ./ repmat(sqrt(sum(e2s.^2, 2)), 1, 3);
e3s_norm = e3s ./ repmat(sqrt(sum(e3s.^2, 2)), 1, 3);

% Compute the angles
angles(:, 1) = acos(sum(e1s_norm .* e2s_norm, 2));
angles(:, 2) = acos(sum(e3s_norm .* e1s_norm, 2));
angles(:, 3) = pi - (angles(:, 1) + angles(:, 2));

% Compute the triangle weighted normals
triangle_normals    = cross(e1s, e3s, 2);
w1_triangle_normals = triangle_normals .* repmat(angles(:, 1), 1, 3);
w2_triangle_normals = triangle_normals .* repmat(angles(:, 2), 1, 3);
w3_triangle_normals = triangle_normals .* repmat(angles(:, 3), 1, 3);

clear e1s e2s e3s e1s_norm e2s_norm e3s_norm angles triangle_normals

% Initialize the vertex normals
normals = zeros(Nvertices, 3);

% Update the vertex normals
for i = 1: Nfaces
    normals(v1(f(i)), :) = normals(v1(f(i)), :) + w1_triangle_normals(i, :);
    normals(v2(f(i)), :) = normals(v2(f(i)), :) + w2_triangle_normals(i, :);
    normals(v3(f(i)), :) = normals(v3(f(i)), :) + w3_triangle_normals(i, :);
end

clear w1_triangle_normals w2_triangle_normals w3_triangle_normals

% Normalize the vertex normals
normals = normals(v, :);
normals = normals ./ repmat(sqrt(sum(normals.^2, 2)), 1, 3);
