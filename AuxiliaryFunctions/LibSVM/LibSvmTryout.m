
numSamplesA = 1000;
numSamplesB = 20;

mDataSamplesA = [3.5 + randn([numSamplesA, 1]), 3.0 + randn([numSamplesA, 1])];
mDataSamplesB = [0.0 + randn([numSamplesB, 1]), 0.0 + randn([numSamplesB, 1])];

mDataSamples = [mDataSamplesA; mDataSamplesB];

vDataLabels = [-9 * ones([numSamplesA, 1]); 2 * ones([numSamplesB, 1])];


[vDataLabels, mDataSamples] = libsvmread('twofeature.txt');

figure();
gscatter(mDataSamples(:, 1), mDataSamples(:, 2), vDataLabels);
hold('on');

svmOptions = '-s 0 -t 0 -c 1 -e 0.001 -h 1 -b 0';
hSvmModelA = svmtrain(vDataLabels, mDataSamples, svmOptions);

% hSvmModelA = svmtrain(vDataLabels, mDataSamples, '-s 0 -t 0');

w = hSvmModelA.SVs' * hSvmModelA.sv_coef;
c = hSvmModelA.rho;

vX = linspace(-10, 10, 10);
vY = (-(w(1) * vX) + c) / w(2);

plot(vX, vY);

vDataLabels(vDataLabels == 1) = 0;


svmOptions = '-s 0 -t 0 -c 1 -e 0.001 -h 1 -b 0 -w0 1 -w0 500';
hSvmModelB = svmtrain(vDataLabels, mDataSamples, svmOptions);

w = hSvmModelB.SVs' * hSvmModelB.sv_coef;
c = hSvmModelB.rho;

vX = linspace(-10, 10, 10);
vY = (-(w(1) * vX) + c) / w(2);

plot(vX, vY);



