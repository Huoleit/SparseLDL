clear;clc;
nx = 4;
nu = 2;
dt = 0.1;
eps = 1e-5;
A = [eye(2) dt*eye(2); zeros(2,2) eye(2)];
B = [0.5*dt*dt*eye(2); dt*eye(2)];

Q = diag([1; 1; 0.1; 0.1]);
R = diag([0.3; 0.3]);

N = 2; % Number of stages

D0 = R;
L10 = B * inv(D0);
D1 = -B * L10' - eye(nx) * eps;
L21 = -inv(D1);
D2 = Q + L21;

D3 = R;
L42 = A * inv(D2);
L43 = B * inv(D3);
D4 = -A * L42' - B*L43'- eye(nx) * eps;
L54 = -inv(D4);
D5 = Q + L54;

numDecision = N * (nu + nx + nx);

KKT = [R zeros(nu, numDecision - nu);
       B -eye(nx)*eps zeros(nx, numDecision - nu - nx);
       zeros(nx, nu) -eye(nx) Q zeros(nx, nu + nx + nx);
       zeros(nu, nu + nx + nx) R zeros(nu, nx + nx);
       zeros(nx, nu + nx) A B -eye(nx)*eps zeros(nx, nx);
       zeros(nx, nu + nx + nx + nu) -eye(nx) Q];

KKT = KKT + tril(KKT, -1)';

D= blkdiag(D0,D1,D2, D3, D4, D5);
L = [eye(nu) zeros(nu , numDecision - nu);
     L10 eye(nx) zeros(nx, numDecision - nu - nx);
     zeros(nx,nu) L21 eye(nx) zeros(nx, numDecision - nx - nu - nx);
     zeros(nu, nu + nx + nx) eye(nu) zeros(nu, nx + nx);
     zeros(nx, nu + nx) L42 L43 eye(nx) zeros(nx, nx);
     zeros(nx, numDecision - nx -nx) L54 eye(nx)];
   
max(abs(L*D*L' - KKT), [],"all")
