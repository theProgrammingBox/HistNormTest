#include <iostream>

/*
IMPORTANT LESSONS:
- the exponential decay terms are necessary because we do not know the starting values of the means and variances.
It basically just pretends that the starting values are the starting entries. It helps stabilize the initial values resulting in stable learning.
If you know the starting values (use some sort of initialization run), you can try to remove the exponential decay terms.
- Given this simple example, it never reaches 0 error, but gets close to 0.0001.
The trade off for this small error is that there is no need for batch normalization.
We just use the history of the means and variances to normalize the inputs.
This is a pseudo batch normalization for models that can't really use batch normalization.
*/

float InvSqrt(float number)
{
	long i = 0x5F1FFFF9 - (*(long*)&number >> 1);
	float tmp = *(float*)&i;
	return tmp * 0.703952253f * (2.38924456f - number * tmp * tmp);
}

float RandomFloat()
{
	return rand() / (float)RAND_MAX;
}

int main()
{
	srand(time(nullptr));

	const int batchSize = 1;
	const float learningRate = 0.1f;
	const float epsilon = 1e-16f;
	const float meanBeta = 0.9f;
	const float varianceBeta = 0.999f;
	const float normMeanBeta = 0.999f;
	const float normVarianceBeta = 0.999f;
	float meanExpDecay = 1;
	float varianceExpDecay = 1;
	float normMeanExpDecay = 1;
	float normVarianceExpDecay = 1;

	float correctInputNormMean = 0;
	float accumulatedInputNormMean = 0;
	float InputNormMean;

	float correctInputNormVariance = 0;
	float accumulatedInputNormVariance = 0;
	float InputNormVariance;

	float weight = 1;
	float weightGradient;

	float bias = 0;
	float biasGradient;

	float weightGradientMean = 0;
	float weightGradientVariance = 0;

	float biasGradientMean = 0;
	float biasGradientVariance = 0;

	while (true)
	{
		InputNormMean = 0;
		InputNormVariance = 0;
		weightGradient = 0;
		biasGradient = 0;

		float avgErr = 0;
		for (int batch = 0; batch < batchSize; ++batch)
		{
			float input = RandomFloat();
			float target = input * 2 + 2;

			InputNormMean += input;
			float delta = input - correctInputNormMean;
			InputNormVariance += delta * delta;
			float shiftedInput = input - correctInputNormMean;
			float inputNorm = shiftedInput * correctInputNormVariance;
			float output = inputNorm * weight + bias;
			float error = target - output;
			float inputNormGradient = error * weight;
			float inputGradient = inputNormGradient * correctInputNormVariance;
			weightGradient += inputNorm * error;
			biasGradient += error;

			avgErr += abs(error);
		}

		printf("Error: %f\n", avgErr / batchSize);

		normMeanExpDecay *= normMeanBeta;
		normVarianceExpDecay *= normVarianceBeta;

		InputNormMean /= batchSize;
		accumulatedInputNormMean = normMeanBeta * accumulatedInputNormMean + (1 - normMeanBeta) * InputNormMean;
		correctInputNormMean = accumulatedInputNormMean / (1 - normMeanExpDecay);

		InputNormVariance /= batchSize;
		accumulatedInputNormVariance = normVarianceBeta * accumulatedInputNormVariance + (1 - normVarianceBeta) * InputNormVariance;
		correctInputNormVariance = InvSqrt(accumulatedInputNormVariance / (1 - normVarianceExpDecay) + epsilon);

		meanExpDecay *= meanBeta;
		varianceExpDecay *= varianceBeta;

		weightGradient /= batchSize;
		weightGradientMean = meanBeta * weightGradientMean + (1 - meanBeta) * weightGradient;
		weightGradientVariance = varianceBeta * weightGradientVariance + (1 - varianceBeta) * weightGradient * weightGradient;
		float weightGradientMeanCorrected = weightGradientMean / (1 - meanExpDecay);
		float weightGradientVarianceCorrected = weightGradientVariance / (1 - varianceExpDecay);
		weight += learningRate * weightGradientMeanCorrected * InvSqrt(weightGradientVarianceCorrected + epsilon);

		biasGradient /= batchSize;
		biasGradientMean = meanBeta * biasGradientMean + (1 - meanBeta) * biasGradient;
		biasGradientVariance = varianceBeta * biasGradientVariance + (1 - varianceBeta) * biasGradient * biasGradient;
		float biasGradientMeanCorrected = biasGradientMean / (1 - meanExpDecay);
		float biasGradientVarianceCorrected = biasGradientVariance / (1 - varianceExpDecay);
		bias += learningRate * biasGradientMeanCorrected * InvSqrt(biasGradientVarianceCorrected + epsilon);
	}

	return 0;
}