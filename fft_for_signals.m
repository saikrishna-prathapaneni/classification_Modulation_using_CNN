x = 0:0.1:2*pi;
y = sin(x);
z = fft(y);
z = fftshift(z);
N = length(y);
f = -0.01:0.01;
plot(f,abs(z))