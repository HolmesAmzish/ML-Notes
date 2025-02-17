data = load('ex1data1.txt');
X = data(:, 1); y = data(:, 2);
m = length(y);
plot(X, y, 'rx', 'MarkerSize', 10);
ylabel('Profit in $10,100s');
xlabel('Population of City in 10,000s');