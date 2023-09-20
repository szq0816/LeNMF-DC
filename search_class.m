function [Fres] = search_class(H,Gt,label_num)
%SEARCH_CLASS 此处显示有关此函数的摘要
%   此处显示详细说明
[nSmp, ~] = size(H);
Fres = zeros(nSmp,1);
Fres(1:label_num) = Gt(1:label_num);
unlabel_smp_num = nSmp - label_num;
for i = 1:unlabel_smp_num
    j = i+label_num;
    Hi = H(j,:);
    [~, indexH] = sort(Hi,'descend');
    Fres(j) = indexH(1);
end
clear nSmp unlabel_smp_num i j Hi indexH H Gt label_num
end

