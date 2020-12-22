% $$ %% AM Modulation and Demodulation using Functions in MATLAB
%% Modulation index
h = input('Modulation index = ');
%% Time Pereiod of Simulation :
t = linspace(0, 0.2, 100000);
%% Message Signal :
Am = 20;
fm = 200;
ym = Am*cos(2*pi*fm*t);
figure;
subplot(4, 1, 1);
plot(t(1:10000), ym(1:10000));
title('Message Signal');

xlabel('time(t)');
ylabel('Amplitude');
%% Carrier Signal :
Ac = Am/h;
fc = 2000;
yc = Ac*cos(2*pi*fc*t);
subplot(4, 1, 2);
plot(t(1:10000), yc(1:10000));
title('Carrier Signal');
xlabel('time(t)');
ylabel('Amplitude');

%% addtion of noise to the modulated signal
m=1000;
n= 1;
power = 0.01;
noise = wgn(m,n,power);
plot(noise);
title("noise");
%% Modulated Signal :
y = ammod(ym, fc, 4000,0 , Ac)+noise;

subplot(4, 1, 3);
plot(t(1:10000), y(1:10000));
title('Modulated Signal');
xlabel('time(t)');
ylabel('Amplitude');

%% Demodulated Signal :
z = amdemod(y, fc, 100000, 0, Ac);
subplot(4, 1, 4);
plot(t(1:10000), z(1:10000));
title('Demodulated Signal');
xlabel('time(t)');
ylabel('Amplitude');
ylim([-10, 10]);