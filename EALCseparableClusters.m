clear all;
close all;

rng(9);

% Data set generation
x10 = [8*(rand-0.5) 8*(rand-0.5)];
x20 = [8*(rand-0.5) 8*(rand-0.5)];

NofSamples =2000;
for n= 1 : NofSamples
  if rand<= 0.5
    r = 1.2*randn;
    phi =2*pi*rand;
    x1(n)= x10(1) + r * cos(phi);
    x2(n)= x10(2) + r * sin(phi);
    d(n) =1;
    clr(n)= 'm';
  else
    r = 1.2*randn;
    phi= 3*pi*rand;
    x1(n)= x20(1) + r *cos(phi);
    x2(n)= x20(2) + r *sin(phi);
    d(n) = -1;
    clr(n) = 'b';
  end
end

%LMS algorithm
mu = 1e-2;
%LMS with Quantized error correction Red line
w0 = 0.2*(rand(3,1) -0.5);
w = w0;

for k=1:NofSamples
  xw= [1; x1(k) ; x2(k)];
  s= w.'*xw;
    if s>0, y=1 ;
    else y = -1;
    end
  e= d(k) -y;
  w = w +mu*e*xw;
end
m1 = -w(2)/w(3);
c1 = -w(1)/w(3);

%LMS Data error correcton Green line
w = w0;
for k=1: NofSamples
  xw = [ 1; x1(k); x2(k) ];
  s = w.'*xw;
  e = d(k) -s ;
  w = w + mu*e*xw;
end
m2 = -w(2)/w(3);
c2 = -w(1)/w(3);

%Classification line
xmin = -5;
xmax = 5;

x = xmin;
dx = (xmax- xmin)/NofSamples;
for i = 1 : NofSamples
  ff1(i) = m1*x + c1;
  ff2(i) = m2*x + c2;
  xx(i) = x;
  x = x + dx;
end

% Plot results
figure;
hold on;
grid on;
box on;
for n=1: NofSamples
  plot(x1(n), x2(n), '.','color',clr(n), 'LineWidth', 2);
end

ylim([ xmin xmax ]);
xlim([ xmin xmax ]);
title('Two classes problem');

plot(xx, ff1, '.','color','r', 'LineWidth', 2);
plot(xx, ff2, '.','color','g', 'LineWidth', 2);
xlabel('\it{x}_1');
