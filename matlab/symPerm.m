clear;clc;
syms R0 Q1 R1 Q2 R2 Q3 m0 m1 m2 B0 A1 B1 A2 B2 I

H = diag([R0 Q1 R1 Q2 R2 Q3]);
G = [B0 0 0 0 0 0;
     0 A1 B1 -I 0 0;
     0 0  0 A2 B2 -I];
 
 hessian = [H G';G zeros(size(G, 1))]
 
 p = [1 7 2 3 8 4 5 9 6]';
 P = zeros(size(p, 1));
 for i = 1:size(P, 1)
     P(i, p(i)) = 1;
 end
 
 P * hessian * P'