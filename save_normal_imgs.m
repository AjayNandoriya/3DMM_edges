


clear all
tic;
%% 
addpath('ZhuRamananDetector','optimisations','utils','comparison');

% YOU MUST set this to the base directory of the Basel Face Model
BFMbasedir = '\\QCRI-PRECISION-/Documents/ajay/CNN/Face/SOA/3dmm_cnn/3DMM_model/';

% Load morphable model
load(strcat(BFMbasedir,'01_MorphableModel.mat'));
% Important to use double precision for use in optimisers later
shapeEV = double(shapeEV);
shapePC = double(shapePC);
shapeMU = double(shapeMU);
texPC = double(texPC/256);
texMU = double(texMU/256);
% We need edge-vertex and edge-face lists to speed up finding occluding boundaries
% Either: 1. Load precomputed edge structure for Basel Face Model
load('BFMedgestruct.mat');
% Or: 2. Compute lists for new model:
%TR = triangulation(tl,ones(k,1),ones(k,1),ones(k,1));
%Ev = TR.edges;
%clear TR;
%Ef = meshFaceEdges( tl,Ev );
%save('edgestruct.mat','Ev','Ef');


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
%% load shape & camera param
[pid,pidname] = textread('\\QCRI-PRECISION-/Documents/ajay/CNN/Face/dataset/pid_list.txt','%d %s');
piddata= csvread('\\QCRI-PRECISION-/Documents/ajay/CNN/Face/scripts/comb_1508_testing1_different_target.csv');
pidlist = unique(piddata(:,1)');
img = zeros(1300,1030,3);
outdir = '\\QCRI-PRECISION-\Documents\ajay\CNN\Face\SOA\3DMM_edges\mean_img_rotate';
for pid_val = 0:351
    feature_3d_fname = ['\\QCRI-PRECISION-/Documents/ajay/CNN/Face/SOA/3DMM_edges/mean_img_rotate/mean_' sprintf('%03d',pid_val) '_15.mat'];
    if(~exist(feature_3d_fname,'file'))
        continue;
    end
    load(feature_3d_fname);
    % b = csvread('/home/qcri/Documents/ajay/CNN/Face/SOA/3dmm_cnn/demoCode/mean_out/mean_004_1_15.ply.alpha');
    % b=b(1:ndims);
    FV.vertices=reshape(shapePC(:,1:ndims)*b+shapeMU,3,size(shapePC,1)/3)';
    FV.faces = tl;
    img_normal = render_face_normals(FV,img,R,t,s);
    img_mask = isnan(img_normal);
    imwrite((img_normal+1)/2,fullfile(outdir,sprintf('%normals03d_15_normal.png',pid_val)));
    imwrite(img_mask(:,:,1),fullfile(outdir,sprintf('%03d_15_mask.png',pid_val)));
end