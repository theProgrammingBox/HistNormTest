#include <iostream>

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

	const int batchSize = 4;
	const float learningRate = 0.01f;
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
			float inputNorm = (input - correctInputNormMean) * InvSqrt(correctInputNormVariance + epsilon);
			//float inputNorm = (input - accumulatedInputNormMean) * InvSqrt(accumulatedInputNormVariance + epsilon);
			float output = inputNorm * weight + bias;
			float error = target - output;
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
		correctInputNormVariance = accumulatedInputNormVariance / (1 - normVarianceExpDecay);

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