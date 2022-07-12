clear;clc;
nx = 4;
nu = 2;
dt = 0.1;
A = [eye(2) dt*eye(2); zeros(2,2) eye(2)];
B = [0.5*dt*dt*eye(2); dt*eye(2)];

Q = diag([1; 1; 0.1; 0.1]);
R = diag([0.3; 0.3]);

N = 2; % Number of stages

D0 = Q;
D1 = R;
L20 = A * inv(D0);
% L20 = zeros(size(D0));
L21 = B * inv(D1);
D2 = -A * L20' -B*L21';
L32 = -inv(D2);
D3 = Q + L32;
D4 = R;
L53 = A * inv(D3);
L54 = B * inv(D4);
D5 = -A * L53' - B*L54';
L65 = -inv(D5);
D6 = Q + L65;

numDecision = N * (nx + nu + nx) + nx;

KKT = [Q zeros(nx, numDecision - nx);
       zeros(nu, nx) R zeros(nu, numDecision - nx - nu);
       A B zeros(nx, numDecision - nx - nu);
       zeros(nx, nx + nu) -eye(nx) Q zeros(nx, nu + nx + nx);
       zeros(nu, nx + nu + nx + nx) R zeros(nu, nx + nx);
       zeros(nx, nx + nu + nx) A B zeros(nx, nx + nx);
       zeros(nx, nx + nu + nx + nx + nu) -eye(nx) Q ];

KKT = KKT + tril(KKT, -1)';

D= blkdiag(D0,D1,D2, D3, D4, D5, D6);
L = [eye(nx) zeros(nx , numDecision - nx);
    zeros(nu, nx) eye(nu) zeros(nu, numDecision - nx - nu);
    L20 L21 eye(nx) zeros(nx, numDecision - nx - nu - nx);
    zeros(nx, nx + nu) L32 eye(nx) zeros(nx, nu + nx + nx);
   zeros(nu, nx + nu + nx + nx) eye(nu) zeros(nu, nx + nx);
   zeros(nx, nx + nu + nx) L53 L54 eye(nx) zeros(nx, nx);
   zeros(nx, numDecision - nx -nx) L65 eye(nx)];
   
max(abs(L*D*L' - KKT), [],"all")
L*D*L' - KKT

%%

numDecision = N * (nu + nx + nx);

KKT_R = KKT(5:end, 5:end);
L_R = L(5:end, 5:end);
D_R = D(5:end, 5:end);
max(abs(L_R*D_R*L_R' - KKT_R), [],"all")
    
L_R*D_R*L_R' - KKT_R
