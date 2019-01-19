A=double(imread('train_images/train_01899.png'));
Original_img=A;
A=A';%tranpose the image
%denoise
A=medfilt2(A,[5 5]);
A=conv2(A,ones(5,5)/25,'same');
threshold=0.3;
[m n]=size(A);
L=50;
diff_img=zeros(m,n-L+1);
A_mean=diff_img;
%differentiating the image
for i=1:n-L+1
    A_mean(:,i)=mean(A(:,i:i+L-1)');
    diff_img(:,i)=conv(A_mean(:,i),[-1;1],'same');
    diff_img(:,i)=conv(diff_img(:,i),ones(9,1)/9,'same');
end
   SEGimg=zeros(size(diff_img));
   %segmentation
   peak=find(abs(diff_img)==(max(max(abs(diff_img(10:end-10,:))))));%because diff_img is averaged by 1x9 mask
   if diff_img(peak)<0
          SEGimg(diff_img<=diff_img(peak)*threshold)=1;
   else diff_img(peak)>=0
          SEGimg(diff_img>=diff_img(peak)*threshold)=1;
   end
   SEGimg(1:9,:)=0;%because diff_img is averaged by 1x9 mask
   SEGimg(end-8:end)=0;
   SEGimg=SEGimg';%tranpose it again
   SEGimg=imresize(SEGimg,[m n]);
figure
subplot(1,2,1)
imshow(Original_img,[])
title('Original Image(Vertical)')
subplot(1,2,2)
imshow(SEGimg,[])
title('Segmantation')
