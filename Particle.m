% Read image
fid = fopen('train.csv');
readheader = textscan(fid,'%s %s',2529,'Headerlines',0,'Delimiter',',');

cd 'F:\影像處理final\train_images';


list = string(readheader{1,1});
number = str2double(string(readheader{1,2}));
[train, map] = imread(char(list(20)));
figure
imshow(train);
colormap('gray')
        
train = double(train);

% Remove some noise
img_filtered = medfilt2(train,[31,31],'symmetric');
colormap('gray')
figure
imshow(mat2gray(img_filtered));
colormap('gray')

% Threshold
        
threshold = mean(mean(img_filtered));
product = train<threshold;
figure
subplot(1,2,1);
imshow(train/256);
colormap('gray')
title('Original Image(Particle)')
subplot(1,2,2);
imshow(product);
colormap('gray')
title('Segmentation')
figure
imshow(product);
colormap('gray')

[N,edges] = histcounts(img_filtered);
N = smooth(N);
N = N(round(2/10*length(N)):round(8/10*length(N)));
edges = edges(round(2/10*length(edges)):round(8/10*length(edges)));
[pks,loc] = findpeaks(-N);
threshold2 = edges(loc(pks==max(pks)));
product2 = train<threshold2;
figure
imshow(product2);
colormap('gray')

cd 'F:\影像處理final\';