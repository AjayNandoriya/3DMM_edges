function [normal,mask,depthMap,L,b,texb,Ireconstruct,FV]=img2facedata(im)
% INPUT
% im: color image value range (0,1) 3D matrix, type double

% OUTPUT
% normal    : face normal(nx,ny,nz) 3D matrix
% mask      : face mask (3D boolean matrix)
% depthMap  : face position (X,Y,Z) normalized (for siggraph17 MTP)
% L         : 27x1 single matrix (Light coefficients)
% b         : 60x1 shape coefficient
% texb      : 60x1 texture coefficient
% I reconstruct: reconstructed face image using shape & texture data
%% 



[imH,imW,imD]=size(im);
addpath('ZhuRamananDetector','optimisations','utils','comparison');

% Load morphable model
load('01_MorphableModel.mat');
% Important to use double precision for use in optimisers later
shapeEV = double(shapeEV);
shapePC = double(shapePC);
shapeMU = double(shapeMU);

texMU = double(texMU)/255;
texPC = double(texPC)/255;
% We need edge-vertex and edge-face lists to speed up finding occluding boundaries
% Either: 1. Load precomputed edge structure for Basel Face Model
load('BFMedgestruct.mat');
% Or: 2. Compute lists for new model:
%TR = triangulation(tl,ones(k,1),ones(k,1),ones(k,1));
%Ev = TR.edges;
%clear TR;
%Ef = meshFaceEdges( tl,Ev );
%save('edgestruct.mat','Ev','Ef');

%% ADJUSTABLE PARAMETERS

% Number of model dimensions to use
ndims = 60;
% Prior weight for initial landmark fitting
w_initialprior=0.7;
% Number of iterations for iterative closest edge fitting
icefniter=7;

options.Ef = Ef;
options.Ev = Ev;
% w1 = weight for edges
% w2 = weight for landmarks
% w3 = 1-w1-w2 = prior weight
options.w1 = 0.45; 
options.w2 = 0.15;

%% Setup basic parameters
edgeim = edge(rgb2gray(im),'canny',0.15);

ZRtimestart = tic;
bs = LandmarkDetector(im);
if(isempty(bs))
    return;
end
ZRtime = toc(ZRtimestart);
disp(['Time for landmark detection: ' num2str(ZRtime) ' seconds']);
[x_landmarks,landmarks]=ZR2BFM( bs,im );
 

%% Initialise using only landmarks

% disp('Fitting to landmarks only...');
% [b,R,t,s] = FitSingleSOP( x_landmarks,shapePC,shapeMU,shapeEV,ndims,landmarks,w_initialprior );
% FV.vertices=reshape(shapePC(:,1:ndims)*b+shapeMU,3,size(shapePC,1)/3)';
FV.faces = tl;

%% Initialise using iterative closest edge fitting (ICEF)

disp('Fitting to edges with iterative closest edge fitting...');
[b,R,t,s] = FitEdges(im,x_landmarks,landmarks,shapePC,shapeMU,shapeEV,options.Ef,options.Ev,tl,ndims, w_initialprior, options.w1, options.w2,icefniter);
FV.vertices = reshape(shapePC(:,1:ndims)*b+shapeMU,3,size(shapePC,1)/3)';
FV.R = R;
FV.t = t;
FV.s = s;
%% Run final optimisation of hard edge cost

disp('Optimising non-convex edge cost...');
maxiter = 5;
iter = 0;
diff = 1;
eps = 1e-9;

[r,c]=find(edgeim);
r = size(edgeim,1)+1-r;

while (iter<maxiter) && (diff>eps)
    
    FV.vertices=reshape(shapePC(:,1:ndims)*b+shapeMU,3,size(shapePC,1)/3)';
    [ options.occludingVertices ] = occludingBoundaryVertices( FV,options.Ef,options.Ev,R );

    X = reshape(shapePC(:,1:ndims)*b+shapeMU,3,size(shapePC(:,1:ndims),1)/3);   
    % Compute position of projected occluding boundary vertices
    x_edge = R*X(:,options.occludingVertices);
    x_edge = x_edge(1:2,:);
    x_edge(1,:)=s.*(x_edge(1,:)+t(1));
    x_edge(2,:)=s.*(x_edge(2,:)+t(2));
    % Find edge correspondences
    [idx,d] = knnsearch([c r],x_edge');
    % Filter edge matches - ignore the worse 5% 
    sortedd=sort(d);
    threshold = sortedd(round(0.95*length(sortedd)));
    idx = idx(d<threshold);
    options.occludingVertices = options.occludingVertices(d<threshold);

    b0 = b;
    [ R,t,s,b ] = optimiseHardEdgeCost( b0,x_landmarks,shapeEV,shapeMU,shapePC,R,t,s,r,c,landmarks,options,tl,false );
    
    diff = norm(b0-b);
    disp(num2str(diff));
    iter = iter+1;
    
end

% Run optimisation for a final time but without limit on number of
% iterations
[ R,t,s,b ] = optimiseHardEdgeCost( b,x_landmarks,shapeEV,shapeMU,shapePC,R,t,s,r,c,landmarks,options,tl,true );
[normal,mask,depthMap] = render_face_normals(FV, im,R,t,s);


pts2D = pts3Dto2D(FV.vertices,R,t,s)';

FV.visiblePoints = visiblevertices(FV,R);
FV.normals = vertex_normals(FV,R,t,s);
pts2D(:,2)=imH+1-pts2D(:,2);
% imshow(img);hold on;plot(pts2D(FV.visiblePoints,1),pts2D(FV.visiblePoints,2),'.');hold off
% figure(2);imshow(img);hold on;plot(pts2D(:,1),pts2D(:,2),'.');

%% find texture param
visible_pointId = false(size(FV.vertices));
visible_pointId(FV.visiblePoints,:)=true;
visible_pointId = reshape(visible_pointId',[],1);
visible_pointId = find(visible_pointId);

pts2D(:,1) = min(max(pts2D(:,1),1),imW);
pts2D(:,2) = min(max(pts2D(:,2),1),imH);
ptsid = sub2ind([imH imW],round(pts2D(:,2)),round(pts2D(:,1)));
im = reshape(im,[],3);
Iv = reshape(im(ptsid,:)',[],1);
Iv_bright= Iv(visible_pointId);
ind_bright = Iv_bright>0.05 & Iv_bright<0.95 ;



A = texPC(visible_pointId,1:ndims)'*texPC(visible_pointId,1:ndims);
B = texPC(visible_pointId,1:ndims)'*(Iv(visible_pointId)-texMU(visible_pointId));
texb = A\B;

C = [0.429043 0.511664 0.743125 0.886227 0.247708];
H = [C(4)*ones(length(FV.normals),1) 2*C(2)*FV.normals(:,2) 2*C(2)*FV.normals(:,3) 2*C(2)*FV.normals(:,1) 2*C(1)*FV.normals(:,1).*FV.normals(:,2) 2*C(1)*FV.normals(:,2).*FV.normals(:,3) C(3)*FV.normals(:,3).*FV.normals(:,3)-C(5) 2*C(1)*FV.normals(:,1).*FV.normals(:,3) C(1)*(FV.normals(:,1).^2-FV.normals(:,2).^2)];
H = reshape(permute(repmat(H,[1 1 3]),[3 1 2]),[],size(H,2));
H(:,10:27)=0;
H(2:3:end,10:18)=H(2:3:end,1:9);
H(3:3:end,19:27)=H(3:3:end,1:9);
H(2:3:end,1:9)=0;
H(3:3:end,1:9)=0;
for iter_l=1:40
    tex_val = texPC(visible_pointId,1:ndims)*texb+texMU(visible_pointId);
    beta_i = tex_val.*H(visible_pointId,:);

    L = (beta_i(ind_bright,:)'*beta_i(ind_bright,:))\(beta_i(ind_bright,:)'*Iv_bright(ind_bright));
%     beta_i = beta_i*norm(L);
%     L = L./norm(L);
    dI = Iv(visible_pointId(ind_bright)) - beta_i(ind_bright,:)*L;
    HL = H(visible_pointId(ind_bright),:)*L;
    d_texb = A\(texPC(visible_pointId(ind_bright),1:ndims)'*(dI./(HL+0.00001)));
    texb = texb+d_texb;
%     norm_L = norm(L);
%     L = L./norm_L;
%     texb = texb*norm_L;
    fprintf('iter=%d dI=%0.5f d_textb= %0.5f\n',iter_l,mean(abs(dI)),mean(abs(d_texb)));
%     tex_val = texPC(:,1:ndims)*texb+texMU;
%     beta_i = tex_val.*H;
%     Ir = reshape(beta_i*L,3,size(texPC,1)/3)';
% 
%     FV.facevertexcdata = Ir;
%     figure(iter_l);imshow(renderFace(FV,img,R,t,s,true));
end

tex_val = texPC(:,1:ndims)*texb+texMU;
beta_i = tex_val.*H;
Ir = reshape(beta_i*L,3,size(texPC,1)/3)';
FV.facevertexcdata = Ir;
Ireconstruct = renderFace(FV,reshape(im,[imH imW imD]),R,t,s,true);
end

