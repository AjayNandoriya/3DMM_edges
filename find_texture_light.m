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
for pid_val = pidlist(end:-1:1)
    for lid_val = [50:56 64:70 78:84 92:96 104:108 116:120]
        outfilename = sprintf('\\\\QCRI-PRECISION-/Documents/ajay/CNN/Face/SOA/3DMM_edges/reconstruct/coeff/coeff_%03d_%03d_15.mat',pid_val,lid_val);
        if(exist(outfilename,'file'))
            continue;
        end
        load(['\\QCRI-PRECISION-/Documents/ajay/CNN/Face/SOA/3DMM_edges/mean_img_rotate/mean_' sprintf('%03d',pid_val) '_15.mat']);
        % b = csvread('/home/qcri/Documents/ajay/CNN/Face/SOA/3dmm_cnn/demoCode/mean_out/mean_004_1_15.ply.alpha');
        % b=b(1:ndims);
        FV.vertices=reshape(shapePC(:,1:ndims)*b+shapeMU,3,size(shapePC,1)/3)';
        FV.faces = tl;
        pts2D = pts3Dto2D(FV.vertices,R,t,s)';

        FV.visiblePoints = visiblevertices(FV,R);
        FV.normals = vertex_normals(FV,R,t,s);
        % img = im2double(imread('/home/qcri/Documents/ajay/CNN/Face/dataset/mean_img_rotate/mean_004_1_15.png'));
        img = im2double(imread(['\\DS2015XS\Kilimanjaro/Dropbox_MIT/MERL_facial/' sprintf('%s/refl1_%03d_15.png',pidname{pid_val+1},lid_val)]));
        img = permute(img,[2 1 3]);
        img_ori = img;
        pts2D(:,2)=size(img,1)+1-pts2D(:,2);
        % imshow(img);hold on;plot(pts2D(FV.visiblePoints,1),pts2D(FV.visiblePoints,2),'.');hold off
        % figure(2);imshow(img);hold on;plot(pts2D(:,1),pts2D(:,2),'.');

        %% find texture param
        visible_pointId = false(size(FV.vertices));
        visible_pointId(FV.visiblePoints,:)=true;
        visible_pointId = reshape(visible_pointId',[],1);

        pts2D(:,1) = min(max(pts2D(:,1),1),size(img,2));
        pts2D(:,2) = min(max(pts2D(:,2),1),size(img,1));
        ptsid = sub2ind(size(img(:,:,1)),round(pts2D(:,2)),round(pts2D(:,1)));
        img = reshape(img,[],3);
        Iv = reshape(img(ptsid,:)',[],1);
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
        img = img_ori;
        for iter_l=1:40
            tex_val = texPC(visible_pointId,1:ndims)*texb+texMU(visible_pointId);
            beta_i = tex_val.*H(visible_pointId,:);

            L = (beta_i(ind_bright,:)'*beta_i(ind_bright,:))\(beta_i(ind_bright,:)'*Iv_bright(ind_bright));

            dI = Iv(visible_pointId) - beta_i*L;
            HL = H(visible_pointId,:)*L;
            d_texb = A\(texPC(visible_pointId,1:ndims)'*(dI./(HL+0.00001)));
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
        toc;
        tex_val = texPC(:,1:ndims)*texb+texMU;
        beta_i = tex_val.*H;
        Ir = reshape(beta_i*L,3,size(texPC,1)/3)';
        FV.facevertexcdata = Ir;
        Ireconstruct = renderFace(FV,img,R,t,s,true);
        imwrite(Ireconstruct,['\\QCRI-PRECISION-/Documents/ajay/CNN/Face/SOA/3DMM_edges/reconstruct/' sprintf('reconstruct_%03d_%03d_15.png',pid_val,lid_val)]);
        save(outfilename,'L','texb');
        
        %%
        
%         figure(1);
%         subplot(131);imshow(img);
%         subplot(132);imshow(renderFace(FV,img,R,t,s,true));
%         FV.facevertexcdata = reshape(tex_val,3,size(texPC,1)/3)';
%         Irecn = renderFace(FV,img,R,t,s,true);
%         subplot(133);imshow(Irecn./max(Irecn(:)));
    end
end

% subplot(122);imshow(renderFace(FV,img,R,t,s,true)/256);








    