clc; clear; clf;

load fisheriris

X = meas;       % features
Y = species;    % label

C = randperm(size(X, 1));
trainX = X(C(1:130), :);
trainY = Y(C(1:130));
testX = X(C(131:150), :);
testY = Y(C(131:150));

accuracy = zeros(1, 50);

tic;
for k = 1:200
    model = fitcknn(trainX, trainY, 'NumNeighbors', k);
    predictY = predict(model, testX);
    accuracy(k) = sum(strcmp(predictY, testY)) / length(testY);
end

toc;

% plot the accuracy
plot(1:200, accuracy);
xlabel('Number of Neighbors (k)');
ylabel('Accuracy');
title('K-NN Accuracy for Different k Values');