%% Stanford CS229 assignment
%
% Chul-Woong Yang
% cwyang@github.com

clear;          % clear all variables
close all;      % close all figures
clc             % clear terminal screen

fprintf('Quasar Regression\n');

% lambdas - A length n = 450 vector of wavelengths {1150, ..., 1599}
% train_qso - A size m-by-n matrix, where m = 200 and n = 450, of noisy
%      observed quasar spectra for training.
% test_qso - A size m-by-n matrix, where m = 200 and n = 450, of noisy observed
%       quasar spectra for testing.
load quasar_train.csv;
lambdas = quasar_train(1, :)';
train_qso = quasar_train(2:end, :);
load quasar_test.csv;
test_qso = quasar_test(2:end, :);

%% ======= unweighted linear regression with normal eq ======
% \theta = (X^TX)^(-1)X^ty
X = [ones(size(lambdas,1),1) lambdas];
y = train_qso(1,:)';
theta=(X'*X)\X'*y

f=figure;
%%set(f, 'visible','off');
hold on;
scatter(X(:,2),y);
xlabel('lambda');
ylabel('Flux');
%% Calculate the decision boundary line
plot_x = [min(lambdas)-2,  max(lambdas)+2];
plot_y = theta(2).*plot_x + theta(1);
plot(plot_x, plot_y);
axis([plot_x]);
h = legend('measured data', 'regression');
%set(h, 'fontsize', 16);
saveas(f, "ps51.png");
close all;
%% ======= weighted linear regression with normal eq ======
f=figure;
%%set(f, 'visible','off');
hold on;
scatter(X(:,2),y);
xlabel('lambda');
ylabel('Flux');
ys = weightedRegression(X,y,5);
plot(lambdas, ys);
axis([min(lambdas), max(lambdas)]);
h = legend('measured data', 'local regression');
set(h, 'fontsize', 16);
saveas(f, "ps52.png");
close all;
%% ======== tau variation ==============
f=figure;
%%set(f, 'visible','off');
hold on;
scatter(X(:,2),y);
xlabel('lambda');
ylabel('Flux');
colors=['r-','g-','b-','m-'];
taus=[1,10,100,1000];
for i=1:4
  ys = weightedRegression(X,y,taus(i));
  handle = plot(lambdas, ys, colors(i));
  set(handle, 'linewidth', 2);
end
axis([min(lambdas), max(lambdas)]);

h = legend('measured data', 'tau=1', ...
           'tau=10', 'tau=100', 'tau=1000');
set(h, 'fontsize', 16);
saveas(f, "ps53.png");
