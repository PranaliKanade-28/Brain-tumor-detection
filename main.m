% function main
clc;
clear all;
close all;

x=0;

[filename, pathname] = uigetfile('*.dcm','MultiSelect','on');

folder_name = strcat(pathname,'\jpg');
    if not(exist(folder_name,'dir'))
        mkdir(folder_name)
    end

    
    
folder_name3 = strcat(pathname,'\Combined');
    if not(exist(folder_name3,'dir'))
        mkdir(folder_name3)
    end
    
for i = 1:numel(filename)
   img = dicomread(fullfile(pathname,filename{i}));
   x=i
   if isequal(filename, 0) || isequal(pathname, 0)   
    disp('Image input canceled.');  
   else
    [X,MAP]=dicomread(fullfile(pathname, filename{i}));
    image8 = uint8(255 * mat2gray(X));

    %fname=strcat('C:\Users\kanad\OneDrive\Desktop\RA work\Project\Images\Yes\jpg\',num2str(i),'.jpg')
    fname=strcat(pathname,'jpg\',num2str(i),'.jpg')
    %imwrite(image8,'C:\Users\kanad\OneDrive\Desktop\RA work\Project\Images\Yes\jpg\newfile{i}.jpg', 'jpg')
    imwrite(image8,fname, 'jpg')

    %imshow(image8, []);
    end;
    
%im=imread('C:\Users\kanad\OneDrive\Desktop\RA work\Project\jpg\myfile.jpg');
im=imread(fname);
im=im2gray(im);
%im=dicomread('C:\Users\kanad\OneDrive\Desktop\RA work\Project\Images\Yes\y(1).dcm');
img=im
%subplot(2,1,1),imshow(im);
%subplot(2,1,2),imhist(im(:,:,1));
%title('INPUT IMAGE HISTOGRAM');%figure,imhist(im(:,:,2)),title('blue');figure,imhist(im(:,:,3)),title('Green');

%figure;
I = imnoise(im2gray(im),'salt & pepper',0.02);
%subplot(1,2,1),imshow(I);
%title('Noise adition and removal using median filter');
K = medfilt2(I);
%subplot(1,2,2),imshow(K);


im = double(im);
s_img = size(im);
r = im(:,:,1);
g = im(:,:,1);
b = im(:,:,1);
% [c r] = meshgrid(1:size(i,1), 1:size(i,2));
data_vecs = [r(:) g(:) b(:)];

k= 4;

[ idx C ] = kmeansK( data_vecs, k );
% d = reshape(data_idxs, size(i,1), size(i,2));
% imagesc(d);

palette = round(C);

%Color Mapping
idx = uint8(idx);
outImg = zeros(s_img(1),s_img(2),3);
temp = reshape(idx, [s_img(1) s_img(2)]);
for i = 1 : 1 : s_img(1)
    for j = 1 : 1 : s_img(2)
        outImg(i,j,:) = palette(temp(i,j),:);
    end
end

cluster1 = zeros(size(r));
cluster2 = zeros(size(r));
cluster3 = zeros(size(r));
cluster4 = zeros(size(r));

%figure;
cluster1(find(outImg(:,:,1)==palette(1,1))) = 1;
%subplot(2,2,1), imshow(cluster1);
cluster2(find(outImg(:,:,1)==palette(2,1))) = 1;
%subplot(2,2,2), imshow(cluster2);
cluster3(find(outImg(:,:,1)==palette(3,1))) = 1;
%subplot(2,2,3), imshow(cluster3);
cluster4(find(outImg(:,:,1)==palette(4,1))) = 1;
%subplot(2,2,4), imshow(cluster4);

%subplot(1,2,1),imshow(img)
%title('Original image');
cc = imerode(cluster4,[1 1]);
%figure,imshow(imerode(cluster4,[1 1]));
%subplot(1,2,2),imshow(imerode(cluster4,[1 1]));
%title('eroded image');

[label_im, label_count] = bwlabel(cc,8); 
stats = regionprops(label_im, 'centroid');


%fname2=strcat('C:\Users\kanad\OneDrive\Desktop\RA work\Project\Images\Yes\jpg\Eroded',num2str(x),'.jpg')
fname2=strcat(pathname,'jpg\Eroded',num2str(x),'.jpg')
imwrite(imerode(cluster4,[1 1]),fname2, 'jpg')

end


for i=1:x
    figure;
    %xname=imread(strcat('C:\Users\kanad\OneDrive\Desktop\RA work\Project\Images\Yes\jpg\',num2str(i),'.jpg'));
    xname=imread(strcat(pathname,'jpg\',num2str(i),'.jpg'));
    %subplot(1,2,1), imshow(xname);
    %yname=imread(strcat('C:\Users\kanad\OneDrive\Desktop\RA work\Project\Images\Yes\jpg\Eroded',num2str(i),'.jpg'));
    yname=imread(strcat(pathname,'jpg\Eroded',num2str(i),'.jpg'));
    %subplot(1,2,2), imshow(yname);
    
    combFinal=imfuse(xname,yname,'montage');
    %fname2=strcat('C:\Users\kanad\OneDrive\Desktop\RA work\Project\Images\Yes\jpg\Combined\combFinal',num2str(i),'.jpg');
    fname2=strcat(pathname,'Combined\Combined',num2str(i),'.jpg');
    imwrite(combFinal,fname2,'jpg');
    imshow(combFinal)
    title('Brain scan image with eroded image');
    
    %combImg=imfuse(xname,yname,'montage');
    %imshow(combImg)
    %combFinal=infuse(imread(combFinal),imread(combImg),'montage');
    %combFinal=cat(combFinal,combImg);
    
end

%imshow(combFinal)

%%
%imdsT=imageDatastore('trainingSet','IncludeSubfolders',1,'LabelSource','foldernames')

trainingSet=strcat(pathname,'Combined');

imdsT=imageDatastore(trainingSet,'FileExtensions',{'.jpg'})

T=countEachLabel(imdsT)

imgTotal = length(imdsT.Files)

i=readimage(imdsT,1);
imshow(i)

if (mod(x,5)~=0)
    a=fix(x/5)+1;
else
    a=fix(x/5);
end

b=5;
N=randperm(imgTotal,x);

figure('Color','White'),
Idx=1;

n=1;

for j=1:a
	for k=1:b
        if n>x
            break
        end
		img=readimage(imdsT,N(Idx));
		subplot(a,b,Idx)
		imshow(img);
		Idx=Idx+1;
        n=n+1;
        
    end
    
end
sgtitle('Brain scan images with eroded image');

fname3=strcat(pathname,'Combined\CombinedFinal.jpg');
imwrite(combFinal,fname3,'jpg');

code_end = 1;








