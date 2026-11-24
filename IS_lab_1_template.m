%Classification using perceptron
clear;
clc;

%Reading apple images
A1=imread('apple_04.jpg');
A2=imread('apple_05.jpg');
A3=imread('apple_06.jpg');
A4=imread('apple_07.jpg');
A5=imread('apple_11.jpg');
A6=imread('apple_12.jpg');
A7=imread('apple_13.jpg');
A8=imread('apple_17.jpg');
A9=imread('apple_19.jpg');

%Reading pears images
P1=imread('pear_01.jpg');
P2=imread('pear_02.jpg');
P3=imread('pear_03.jpg');
P4=imread('pear_09.jpg');

% Read saved features
Data = readmatrix('Data.txt');
T = double(Data(:,3) == 1);

% Set to true to calculate features from images (requires Toolbox)
useImages = false;

if useImages
    %Calculate for each image, colour and roundness
    %For Apples
    %1st apple image(A1)
    hsv_value_A1=spalva_color(A1); %color
    metric_A1=apvalumas_roundness(A1); %roundness
    %2nd apple image(A2)
    hsv_value_A2=spalva_color(A2); %color
    metric_A2=apvalumas_roundness(A2); %roundness
    %3rd apple image(A3)
    hsv_value_A3=spalva_color(A3); %color
    metric_A3=apvalumas_roundness(A3); %roundness
    %4th apple image(A4)
    hsv_value_A4=spalva_color(A4); %color
    metric_A4=apvalumas_roundness(A4); %roundness
    %5th apple image(A5)
    hsv_value_A5=spalva_color(A5); %color
    metric_A5=apvalumas_roundness(A5); %roundness
    %6th apple image(A6)
    hsv_value_A6=spalva_color(A6); %color
    metric_A6=apvalumas_roundness(A6); %roundness
    %7th apple image(A7)
    hsv_value_A7=spalva_color(A7); %color
    metric_A7=apvalumas_roundness(A7); %roundness
    %8th apple image(A8)
    hsv_value_A8=spalva_color(A8); %color
    metric_A8=apvalumas_roundness(A8); %roundness
    %9th apple image(A9)
    hsv_value_A9=spalva_color(A9); %color
    metric_A9=apvalumas_roundness(A9); %roundness

    %For Pears
    %1st pear image(P1)
    hsv_value_P1=spalva_color(P1); %color
    metric_P1=apvalumas_roundness(P1); %roundness
    %2nd pear image(P2)
    hsv_value_P2=spalva_color(P2); %color
    metric_P2=apvalumas_roundness(P2); %roundness
    %3rd pear image(P3)
    hsv_value_P3=spalva_color(P3); %color
    metric_P3=apvalumas_roundness(P3); %roundness
    %2nd pear image(P4)
    hsv_value_P4=spalva_color(P4); %color
    metric_P4=apvalumas_roundness(P4); %roundness

    x1 = [hsv_value_A1 hsv_value_A2 hsv_value_A3 hsv_value_A4 ...
          hsv_value_A5 hsv_value_A6 hsv_value_A7 hsv_value_A8 ...
          hsv_value_A9 hsv_value_P1 hsv_value_P2 hsv_value_P3 hsv_value_P4];
    x2 = [metric_A1 metric_A2 metric_A3 metric_A4 metric_A5 ...
          metric_A6 metric_A7 metric_A8 metric_A9 ...
          metric_P1 metric_P2 metric_P3 metric_P4];
else
    x1 = Data(:,1)';
    x2 = Data(:,2)';
end

% Estimated features
P = [x1; x2];
d = T';
N = length(d);

%% Train single perceptron with two inputs and one output
w1 = randn(1);
w2 = randn(1);
b = randn(1);
LR = 0.3;

e = 1;
epoch = 0;
Y = zeros(1,N);

while e ~= 0 && epoch < 100000
    for i = 1:N
        v = x1(i)*w1 + x2(i)*w2 + b;
        if v > 0
            y = 1; % Apple
        else
            y = 0; % Pear
        end

        err = d(i) - y;
        w1 = w1 + LR*err*x1(i);
        w2 = w2 + LR*err*x2(i);
        b = b + LR*err;
    end

    % Check all examples with updated weights
    e = 0;
    for i = 1:N
        v = x1(i)*w1 + x2(i)*w2 + b;
        if v > 0
            Y(i) = 1;
        else
            Y(i) = 0;
        end
        e = e + abs(d(i) - Y(i));
    end
    epoch = epoch + 1;
end

% Results
names = {'apple_04.jpg','apple_05.jpg','apple_06.jpg', ...
         'apple_07.jpg','apple_11.jpg','apple_12.jpg', ...
         'apple_13.jpg','apple_17.jpg','apple_19.jpg', ...
         'pear_01.jpg','pear_02.jpg','pear_03.jpg','pear_09.jpg'};
for i = 1:N
    if Y(i) == 1
        fprintf('%s: Apple\n', names{i});
    else
        fprintf('%s: Pear\n', names{i});
    end
end

disp('Desired output:');
disp(d);
disp('Actual output:');
disp(Y);
fprintf('Total error: %d\n', e);
fprintf('Epochs: %d\n', epoch);
