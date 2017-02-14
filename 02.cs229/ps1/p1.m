%% Stanford CS229 assignment
%
% Chul-Woong Yang
% cwyang@github.com

clear;          % clear all variables
close all;      % close all figures
clc             % clear terminal screen

fprintf('Newton Method\n');

X = load('logistic_x.txt');
y = load('logistic_y.txt');
[m,n] = size(X);
X2 = [ones(m,1) X]; % Add intercept term
theta = zeros(n+1,1);
[J, grad, hess] = costFunction(theta, X2, y);
step=100
for i=1:step
  theta=theta-hess\grad;                                %  theta=theta-inv(hess)*grad;
  [J, grad, hess] = costFunction(theta, X2, y);
  if rem(i,10) == 0
    fprintf("%d step: theta=%f %f %f\n", i, theta);
  end
end

%% ======= plot data =========
f=figure;
hold on;
pos = find(y==1);
neg = find(y==-1);
plot (X(pos,1), X(pos,2), 'k+', 'LineWidth', 2, 'MarkerSize', 5);
plot (X(neg,1), X(neg,2), 'ko', 'LineWidth', 1, 'MarkerSize', 5);
xlabel('x1');
ylabel('x2');
legend('pos', 'neg');
hold off;

plotDecisionBoundary(theta,X2,y);

saveas(f, "ps11.png");
pause;
