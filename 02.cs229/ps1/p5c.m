%% Stanford CS229 assignment
%
% Chul-Woong Yang
% cwyang@github.com

clear;          % clear all variables
close all;      % close all figures
clc             % clear terminal screen

fprintf('Prob 5.c) Functional Regression\n');
fflush(stdout);

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

[mm,nn] = size(train_qso);
mtest = size(test_qso,1);
tau=5

%% (i) smooth the data using weighted linear regression
train_smooth = train_qso;
test_smooth = test_qso;
X = [ones(nn,1) lambdas]; %
for i=1:mm
  fprintf("regression on training sample %d\n", i);
  fflush(stdout);
  train_smooth(i,:) = weightedRegression(X,train_qso(i,:)', tau)'; % to row vector
end

for i=1:mtest
  fprintf("regression on testing sample %d\n", i);
  fflush(stdout);
  test_smooth(i,:) = weightedRegression(X,test_qso(i,:)', tau)'; % to row vector
end

train_right= train_smooth(:,151:end); % 1300~1599
train_left = train_smooth(:,1:50);     % 1150~1199
test_right = test_smooth(:,151:end);
test_left  = test_smooth(:,1:50);

dist = zeros(mm,mm);
for i=1:mm
  for j=1:mm
    if j > i
      dist(i,j) = norm(train_right(i, :) - train_right(j, :))^2;
    else
      dist(i,j) = dist (j,i);
    end
  end
end
dist = dist / max(dist(:));

f_left_estimate=zeros(mm:50);
ker=@(t) max(1-t,0);

knear=3;
for i=1:mm
  [near_ks, knn]=sort(dist(i,:));
  knn=knn(1:knear);
  near_ks=near_ks(1:knear)';
  f_left = train_left(knn,:); % 3-by-n
  
  h=max(dist(:,i)); % 0.40
  kern = ker(near_ks/h); % 3-by-1
  f_left_estimate(i,:) = kern' * f_left; % 1-by-n
  f_left_estimate(i,:) /= sum(kern);
end

error = sum((train_left(:) - f_left_estimate(:)).^2) / mm;
fprintf("f_left estimated error of training set = %f\n", error);

%% regresssion on test set
%% it's important to measure test against train set.
%% it's so puzzling that I refered solution many times
train_to_test_dist = zeros(mm,mtest);

for i=1:mm
  for j=1:mtest
    train_to_test_dist(i,j) = norm(train_right(i, :) - test_right(j, :))^2;
  end
end
train_to_test_dist = train_to_test_dist / max(train_to_test_dist(:));

f_left_estimate=zeros(mtest:50);
ker=@(t) max(1-t,0);
for i=1:mtest
  [near_ks, knn]=sort(train_to_test_dist(:,i));
  knn=knn(1:knear);
  near_ks=near_ks(1:knear);
  f_left = train_left(knn,:); % 3-by-n
  
  h=max(train_to_test_dist(:,i)); % 0.40
  kern = ker(near_ks/h); % 3-by-1
  f_left_estimate(i,:) = kern' * f_left; % 1-by-n
  f_left_estimate(i,:) /= sum(kern);
end
error = sum((test_left(:) - f_left_estimate(:)).^2) / mtest;
fprintf("f_left estimated error of test set = %f\n", error);

%% plot
f=figure;
subplot(2,1,1);
plot(lambdas, test_smooth(1,:), 'k-');
axis([min(lambdas), max(lambdas)]);
hold on;
plot(lambdas(1:50),f_left_estimate(1,:), 'r-', 'linewidth', 2);
subplot(2,1,2);
plot(lambdas, test_smooth(6,:), 'k-');
hold on;
plot(lambdas(1:50),f_left_estimate(6,:), 'r-', 'linewidth', 2);
axis([min(lambdas), max(lambdas)]);
saveas(f,"ps54.png")
