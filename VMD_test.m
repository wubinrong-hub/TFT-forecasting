clear all;
close all;
clc;
A=xlsread('1.csv');
alpha = 2000;        % moderate bandwidth constraint
tau = 0;            % noise-tolerance (no strict fidelity enforcement)
K = 8;              % 3 modes
DC = 0;             % no DC part imposed
init = 1;           % initialize omegas uniformly
tol = 1e-7;
[M,N]=size(A)
U = zeros(M,N*K);
for n=1:N
    f=A(:,n);
    [u, u_hat, omega] = VMD(f, alpha, tau, K, DC, init, tol);
    for k = 1:K
        U(:,k+(N-1)*K)=u(:,k)
    end
end
xlswrite('VMD.xls',U,'sheet1');
mr=0;
number=length(A);
for a=1:number
    mr_1=0;
    for b=1:K
        mr_1=mr_1+U(a,b);
    end
    mr=mr+abs((A(a)-mr_1)/A(a))
end
mr=mr/number
% a=A(1)-U(1,1)-U(1,2)-U(1,3);
% mr=(A-U(:,1)-U(:,2)-U(:,3))/(A*100)











