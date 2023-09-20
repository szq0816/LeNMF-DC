function [Hs, result,F,P,R, obj] = LeNMF_DC(X,CA, K, G, numlabel, alpha, lammda,beta,maxiter,mu_1,mu_2)
%%%三个图正则化项
disp('Label-embedded regularized NMF with dual-graph constraints');
thresh=1e-5;
view_num = max(size(X));
C_num = length(unique(G)); 

for iv = 1:view_num
    [mFea(iv), nSmp] = size(X{iv});
end
clear iv
graph_k = 5;%%12

% =====================   Normalization =====================
for i = 1:view_num
    X{i} = bsxfun(@rdivide,X{i},sqrt(sum(X{i}.^2,1)));
end
% initialize each layer for each view
A = cell(view_num, 1);
Z = cell(view_num, 1); 
W = cell(1,view_num);
Wz = cell(1,view_num);
D = cell(1,view_num);
Dz = cell(1,view_num);
Lz = [];
L_V = zeros(nSmp);
L_A = zeros(nSmp);
tau_A = ones(view_num,1).*(1/view_num);
tau_V = ones(view_num,1).*(1/view_num);
options = [];
options.k = graph_k;
options.WeightMode = 'HeatKernel'; %%'Binary', 'HeatKernel', 'Cosine'
% % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% % % % % % % Get the weight matrix by LLE  % % % % % % %
for i_view = 1:view_num
    X1 = X{i_view};
    W{i_view} = constructW(X1',options);
    Wz{i_view} = constructW(X1,options);
    D{i_view} = diag(sum(constructW(X1', options),2));
    Dz{i_view} = diag(sum(constructW(X1, options),2));
    Li{i_view} = D{i_view} - W{i_view};%需要更改
    Lz{i_view} = Dz{i_view} - Wz{i_view};%需要更改
%     L_V = L_V+tau_V(i_view)*Li{i_view};
%     L_A = L_A+tau_A(i_view)*Li{i_view};
end


% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
% % % % % % initialize input random % % % % % % % % % % % % 
for i = 1:view_num
    Z{i} = rand(mFea(i),K,1);     %U
    A{i} = rand(C_num,K,1);       %
    H{i} = rand(K,nSmp,1);        %A
%     C{i} = rand(nSmp,C_num,1);
end
clear i
% C = rand(nSmp,C_num,1);
 C = CA;
%initialize Consensus matrix H_*
% Hs = H{view_num,layer_num};

% normalization
% initialize wieght

% =====================   updata =====================
for iter = 1:maxiter         
    for i_view = 1:view_num
%         for i = 1:K
%             Dx(i) = 1/norm(Z{i_view}(:,i));%%%2、1范数性质
%         end
%         D1{i_view} = diag(Dx);
        Z1 = X{i_view}*C{i_view}*A{i_view}+lammda*Wz{i_view}*Z{i_view};
        Z2 = Z{i_view}*A{i_view}'*C{i_view}'*C{i_view}*A{i_view}+lammda*Dz{i_view}*Z{i_view};
        Z{i_view} = Z{i_view}.*(Z1 ./ max(Z2,1e-10));
        
        H1 = Z{i_view}'*X{i_view}+(C{i_view}*A{i_view})'+lammda*H{i_view}*W{i_view};
        H2 = Z{i_view}'*Z{i_view}*H{i_view}+H{i_view}+lammda*H{i_view}*D{i_view};
        H{i_view} = H{i_view}.*(H1 ./ max(H2,1e-10));

        A1 = C{i_view}'*X{i_view}'*Z{i_view}+alpha*C{i_view}'*W{i_view}*C{i_view}*A{i_view};
        A2 = C{i_view}'*C{i_view}*A{i_view}*Z{i_view}'*Z{i_view}+alpha*C{i_view}'*D{i_view}*C{i_view}*A{i_view};
        A{i_view} = A{i_view}.*(A1./max(A2,1e-10));

        % update C_{i_view}
        C1 = X{i_view}'*Z{i_view}*A{i_view}'+alpha*W{i_view}*C{i_view}*A{i_view}*A{i_view}'+beta*W{i_view}*C{i_view};
        C21 = C{i_view}*A{i_view}*Z{i_view}'*Z{i_view}*A{i_view}';
        C22 = alpha*D{i_view}*C{i_view}*A{i_view}*A{i_view}'+beta*D{i_view}*C{i_view};
        C2 = C21+C22;
        C{i_view} = C{i_view}.*(C1./max(C2,1e-10));
        clear Z1 Z2 A1 A2
    end %%view_num

        %     C{i_view} = C{i_view}.*(C1{i_view}./max(C2{i_view-1},1e-10));
    clear C1 C2
    
    norm2(i_view) = trace(A{view_num}*H{view_num}*Li{i_view}*H{view_num}'*A{view_num}');
    norm1(i_view) = trace(H{view_num}*Li{i_view}*H{view_num}');
    dnorm2 = sum(norm2);
    dnorm1 = sum(norm1);
    for iv = 1:view_num
        tau_A(iv) = (norm1(iv)/dnorm1)^(1/(1-mu_1));
        tau_V(iv) = (norm2(iv)/dnorm2)^(1/(1-mu_2));
        L_V = L_V+tau_V(iv)*Li{iv};
        L_A = L_A+tau_A(iv)*Li{iv};
    end
    % calculate the objective
    obj(iter) = 0;
    obj1 = 0;
    obj2 = 0;
    obj3 = 0;
    obj4 = 0;
    for i_view = 1:view_num
        obj1 = norm(X{i_view}-Z{i_view}*(C{i_view}*A{i_view})','fro');
        obj2 = lammda*trace(Z{i_view}'*Lz{i_view}*Z{i_view});
        obj3 = alpha*trace((C{i_view}*A{i_view})'*L_V*C{i_view}*A{i_view});
        obj4 = beta*trace(C{i_view}'*L_A*C{i_view});
    end
    obj(iter) = obj(iter)+obj1+obj2+obj3+obj4;
    if(iter > 1)
        diff = abs(obj(iter-1) - obj(iter))/obj(iter-1);
        if(diff < thresh)
            break;
        end
    end
%     Ch = C{view_num};
%     [Hs] = search_class(Ch,G,numlabel);
%     result = ClusteringMeasure(double(G), Hs);
%     ACC(iter) = result(1);
%     NMI(iter) = result(2);
%     Purity(iter) = result(3);
end
%un_numlabel = nSmp-numlabel;
for iv = 1:view_num
    for i = numlabel+1:nSmp
        i_un = i-numlabel;
        Ch{iv}(i_un,:) = C{iv}(i,:);
        Gh(i_un,:) = G(i,:);
    end
end
% C = bsxfun(@rdivide,C,sqrt(sum(C.^2,1)));
% % % % % % % % % % % % % % % % % % % 
CH = Ch{view_num};
[Hs] = search_class(CH,Gh,numlabel);
% =====================  result =====================
result = ClusteringMeasure(double(Gh), Hs); 
[F,P,R] = compute_f(Gh,Hs);
% [ ACC, nmii,AR,F,P,R, H ] = evalResults_multiview( Hs, G);
end
