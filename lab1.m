clear;
clc;

Data = readmatrix('Data.txt');

x1 = Data(:,1)';
x2 = Data(:,2)';
d = Data(:,3)'; 

w1 = randn(1);
w2 = randn(1);
b = randn(1);

LR = 0.3;
N = length(d);

for epoch = 1:1000

    for i = 1:N

        v = x1(i)*w1 + x2(i)*w2 + b;

        if v > 0
            y = 1;
        else
            y = -1;
        end

        e = d(i) - y;

        w1 = w1 + LR*e*x1(i);
        w2 = w2 + LR*e*x2(i);
        b = b + LR*e;

    end

    output = 2*double(x1*w1 + x2*w2 + b > 0) - 1;

    total_error = sum(abs(d - output));

    if total_error == 0
        break;
    end

end

fprintf('Epochs: %d\n', epoch);
fprintf('Training error: %d\n', total_error);

files = [dir('apple_*.jpg'); dir('pear_*.jpg')];

correct = 0;

for i = 1:length(files)

    img = imread(files(i).name);

    c1 = spalva_color(img);
    c2 = apvalumas_roundness(img);

    v = c1*w1 + c2*w2 + b;

    if v > 0
        result = 'Apple';
        y = 1;
    else
        result = 'Pear';
        y = -1;
    end

    if startsWith(files(i).name, 'apple_')
        expected = 1;
    else
        expected = -1;
    end

    correct = correct + (y == expected);

    fprintf('%s: %s\n', files(i).name, result);

end

fprintf('Image accuracy: %.2f%%\n', ...
    100*correct/length(files));