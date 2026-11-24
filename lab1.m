clear;
clc;

A1 = imread('apple_04.jpg');
A2 = imread('apple_05.jpg');
A3 = imread('apple_06.jpg');

P1 = imread('pear_01.jpg');
P2 = imread('pear_02.jpg');

x1 = [spalva_color(A1), ...
      spalva_color(A2), ...
      spalva_color(A3), ...
      spalva_color(P1), ...
      spalva_color(P2)];

x2 = [apvalumas_roundness(A1), ...
      apvalumas_roundness(A2), ...
      apvalumas_roundness(A3), ...
      apvalumas_roundness(P1), ...
      apvalumas_roundness(P2)];

d = [1 1 1 -1 -1];

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


    total_error = 0;

    for i = 1:N

        v = x1(i)*w1 + x2(i)*w2 + b;

        if v > 0
            y = 1;
        else
            y = -1;
        end

        total_error = total_error + abs(d(i) - y);

    end

    if total_error == 0
        break;
    end

end

fprintf('Epochs: %d\n', epoch);
fprintf('Training error: %d\n', total_error);

files = {'apple_07.jpg', ...
         'apple_11.jpg', ...
         'apple_12.jpg', ...
         'apple_13.jpg', ...
         'apple_17.jpg', ...
         'apple_19.jpg', ...
         'pear_03.jpg', ...
         'pear_09.jpg'};

expected = [1 1 1 1 1 1 -1 -1];

correct = 0;

for i = 1:length(files)

    img = imread(files{i});

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

    if y == expected(i)
        correct = correct + 1;
    end

    fprintf('%s: %s\n', files{i}, result);

end

accuracy = 100*correct/length(files);

fprintf('Testing accuracy: %.2f%%\n', accuracy);