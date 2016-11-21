% Naive Introduction of Least Squares Method
clear all; close all;
Data = [120 2200; 75 1700; 75 1600; 40 1900];

% Polynomial Regression order and Learing Rate
P = 1;
mu = 0.2;

% internal vectors definition x,d
d = Data(:,2);
x = Data(:,1);
K = length(d);

% Data normalization
max_d = max( Data(:,2));
max_x = max( Data(:,1));

% force zero mean (optional)
mean_d = mean(d);
mean_x = mean(x);
d = (d - mean_d) ./ max_d;
x = (x - mean_x) ./ max_x;

% Simple LMS Algorithm
w = zeros(P+1,1); % initial weights
xx = zeros(P+1,1); % vector for the calculation of the dot product w???*xx
xx(1) = 1; % x^0, always one
for nn = 1 : 500
    for k=1:K
        for p=1:P  % add elements x^p to xx vector for high degree polynomial
            xx(p+1) = x(k)^p;
            end
            e= d(k) - w.'*xx;
            w= w + mu*e*xx;
    end
end

% LS solution
X = zeros(K,P+1); % Data Matrix definition
for i = 1 : K
  X(i,1) = 1;
  X(i,2) = x(i);
  for p = 2:P % add raw to X matrix for high degree polynomial
      X(i,p+1) = x(i)^p;
  end
end
Rxx = 0.0001*eye(P+1) + X.'*X;
wLS = inv(Rxx)*X.'*d;
% Print results
fprintf('Estimated polynomial coeff.s via LMS'); 
w
fprintf('Estimated polynomial coeff.s via LS');
wLS

% Plot results

dxx = (max(x) - min(x)) /K;
x1(1) = min(x);
xx (1)=1;
for n=1:K-1
      for p=1:P % add elements x^p to xx vector for high degree polynomial
       xx(p+1) = x1(n)^p;
      end
      y1(n) = w.'*xx;
      yLS(n) = wLS.'*xx;
      x1(n+1) = x1(n) + dxx; % x-axis for plot
end
y1(n) = w.'*xx;
yLS(n) = wLS.'*xx;

x =x .* max_x + mean_x;
x1 = x1 .* max_x + mean_x;
d =d .* max_d + mean_d;
y1 = y1 .* max_d + mean_d;
yLS = yLS .* max_d + mean_d;

% Scttar plot and regression curve
figure;
hold on;
grid on;
plot(x ,d,'+','LineWidth',2);
plot(x1, y1 , '-', 'color','r', 'LineWidth',2);
plot(x1, yLS,'-', 'color','g', 'LineWidth',2);
title('Rome historic -center apartments rent price ');
xlabel('m^2');
ylabel('Monthly price in Euro');
legend('Data','LMS','LS');
