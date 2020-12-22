clc;
clear all;
close all;

t = linspace(0, 0.2, 1000);
f=1/10;

vin=2*cos(2*pi*f*t);
%subplot(2,1,1);
%plot(t,vin);
R=100;
C=50;
v0= vin*(1/(((R*C))*exp(-1/(R*C))));
subplot(2,1,2);
plot(v0);
plot(vin);


