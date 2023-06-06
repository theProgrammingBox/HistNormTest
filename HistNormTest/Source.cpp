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

	const float epsilon = 1e-16f;
	const float meanBeta = 0.999f;
	const float variationBeta = 0.999f;
	float meanDecay = 1;
	float variationDecay = 1;

	float correctedMean = 0;
	float correctedVariation = 1;
	float uncorrectedMean = 0;
	float uncorrectedVariation = 0;

	float bias = 0;

	for (int j = 0; j < 100; ++j)
	{
		float sum = bias;
		float meanSum = bias;
		float variationSum = abs(bias - correctedMean);
		for (int i = 0; i < 10; ++i)
		{
			// normalize and update sum
			sum = (sum - correctedMean) * correctedVariation;
			float a = RandomFloat();
			printf("norm = %f + %f = %f\n", sum, a, sum + a);
			sum += a;

			// update mean and variance
			meanSum += sum;
			variationSum += abs(sum - correctedMean);
		}
		
		meanDecay *= meanBeta;
		variationDecay *= variationBeta;

		uncorrectedMean = meanBeta * uncorrectedMean + (1 - meanBeta) * (meanSum / 10);
		uncorrectedVariation = variationBeta * uncorrectedVariation + (1 - variationBeta) * (variationSum / 10);

		correctedMean = uncorrectedMean / (1 - meanDecay);
		correctedVariation = (1 - variationDecay) / uncorrectedVariation;

		printf("Mean: %f\n", correctedMean);
		printf("Variance: %f\n\n", correctedVariation);
	}

	return 0;
}