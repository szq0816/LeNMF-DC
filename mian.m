clear all;
clc
% load('BBCSport.mat');
% load('3-sources.mat');
% load('3-sources.mat');
% load E:\R2020a\bin\datasets\mvdata\Caltech101-20.mat;
load E:\R2020a\bin\datasets\COIL20MV.mat;
% X{1} = bbc';
% X{2} = guardian';
% X{3} = reuters';
% X{1} = data1;
% X{2} = data2;
% X{3} = data3;
% X{4} = data4;
G = gt;
% G = truelabel{1};
% G = truth';
view_num = max(size(X));
for i = 1:view_num
%         X{i} = fea{i}';
        X{i} = double(X{i});                                                                                
end
clear i
% view_num = max(size(X));
C_num = length(unique(G));

k = C_num;  %C_num        %%%%从数据集样本中选择的类数
per = 0.1;     %%%%有标签数样本占总样本的比例
maxiter = 350;
layers = 100;
alpha = 5;
lammda = 1000;
beta = 5;
mu_1 = 0;
mu_2 = 1;
%%HW2:5,1000,1,1,1 BBCS:10,0.1,0.5,0.1,0.01,0.01  COIL20:5,1000,5,1,1
% % % % % % % % % % 数据预处理 % % % % % % % % %
[X, CA, G, num_label] = select(X, G, per, k);

% % % % % % % % % % %模型训练跑10次取平均值 % % % % % % % % % % % % 
bounds = 10;
for i = 1:bounds
    fprintf('the %d-th \n', i);
    [Hres, result3,F1,P1,R1, obj] = LeNMF_DC(X,CA,layers, G, num_label,alpha, lammda,beta,maxiter,mu_1,mu_2);
%     [Finres, Hres, result,obj] = DeepMVC(X, layers, G, maxiter);
    acc(i) = result3(1)  %%ACC MIhat Purity
    NMI(i) = result3(2)
    Pur(i) = result3(3)
    F(i) = F1
    P(i) = P1
    R(i) = R1
%         fprintf('The acc is:%8.4f,NMI is:%8.4f,Pur is:%8.4f in %d-th bounds\n', acc(i),NMI(i),Pur(i),i);
end
acc = mean(acc)
NMI = mean(NMI)
Pur = mean(Pur)
F = mean(F)
P = mean(P)
R = mean(R)
