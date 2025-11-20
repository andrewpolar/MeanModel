#include <iostream>
#include <cmath>
#include <algorithm>
#include <thread>
#include "Helper.h"
#include "Function.h"

double Validation(const std::vector<std::unique_ptr<Function>>& inner,
	const std::vector<std::unique_ptr<Function>>& outer, const std::vector<std::vector<double>>& features,
	const std::vector<double>& targets, int nInner, int nOuter) {

	size_t nRecords = targets.size();
	size_t nFeatures = features[0].size();
	std::vector<double> models0(nInner);
	std::vector<double> models1(nOuter);
	std::vector<double> predictions(nRecords);

	for (size_t record = 0; record < nRecords; ++record) {
		for (int k = 0; k < nInner; ++k) {
			models0[k] = 0.0;
			for (size_t j = 0; j < nFeatures; ++j) {
				models0[k] += Compute(features[record][j], true, *inner[k * nFeatures + j]);
			}
			models0[k] /= nFeatures;
		}
		for (int k = 0; k < nOuter; ++k) {
			models1[k] = 0.0;
			for (int j = 0; j < nInner; ++j) {
				models1[k] += Compute(models0[j], true, *outer[j]);
			}
			models1[k] /= nInner;
		}
		predictions[record] = models1[0];
	}
	double pearson = Pearson(predictions, targets);
	return pearson;
}

void Training(std::vector<std::unique_ptr<Function>>& inner,
	std::vector<std::unique_ptr<Function>>& outer, const std::vector<std::vector<double>>& features,
	const std::vector<double>& targets, int nInner, int nOuter, int start, int end, double alpha) {

	size_t nFeatures = features[0].size();
	std::vector<double> models0(nInner);
	std::vector<double> models1(nOuter);
	std::vector<double> deltas0(nInner);
	std::vector<double> deltas1(nOuter);

	for (int record = start; record < end; ++record) {
		for (int k = 0; k < nInner; ++k) {
			models0[k] = 0.0;
			for (size_t j = 0; j < nFeatures; ++j) {
				models0[k] += Compute(features[record][j], false, *inner[k * nFeatures + j]);
			}
			models0[k] /= nFeatures;
		}
		for (int k = 0; k < nOuter; ++k) {
			models1[k] = 0.0;
			for (int j = 0; j < nInner; ++j) {
				models1[k] += Compute(models0[j], false, *outer[j]);
			}
			models1[k] /= nInner;
		}
		deltas1[0] = alpha * (targets[record] - models1[0]);
		for (int j = 0; j < nInner; ++j) {
			deltas0[j] = deltas1[0] * ComputeDerivative(*outer[j]);
		}
		for (int k = 0; k < nOuter; ++k) {
			for (int j = 0; j < nInner; ++j) {
				Update(deltas1[k], *outer[j]);
			}
		}
		for (int k = 0; k < nInner; ++k) {
			for (size_t j = 0; j < nFeatures; ++j) {
				Update(deltas0[k], *inner[k * nFeatures + j]);
			}
		}
	}
}

void Determinants44() {
	const int nTrainingRecords = 100'000;
	const int nValidationRecords = 20'000;
	const int nMatrixSize = 4;
	const int nFeatures = nMatrixSize * nMatrixSize;
	double min = 0.0;
	double max = 10.0;

	auto features_training = GenerateInput(nTrainingRecords, nFeatures, min, max);
	auto features_validation = GenerateInput(nValidationRecords, nFeatures, min, max);
	auto targets_training = ComputeDeterminantTarget(features_training, nMatrixSize);
	auto targets_validation = ComputeDeterminantTarget(features_validation, nMatrixSize);

	clock_t start_application = clock();
	clock_t current_time = clock();

	double targetMin = *std::min_element(targets_training.begin(), targets_training.end());
	double targetMax = *std::max_element(targets_training.begin(), targets_training.end());

	const int nInner = 70;
	const int nOuter = 1;
	const double alpha = 0.1;
	const int nInnerPoints = 3;
	const int nOuterPoints = 30;
	const double termination = 0.97;

	std::random_device rd;
	std::mt19937 rng(rd());
	std::vector<std::unique_ptr<Function>> inner;
	for (int i = 0; i < nInner * nFeatures; ++i) {
		auto function = std::make_unique<Function>();
		InitializeFunction(*function, nInnerPoints, min, max, targetMin, targetMax, rng);
		inner.push_back(std::move(function));
	}
	std::vector<std::unique_ptr<Function>> outer;
	for (int i = 0; i < nInner; ++i) {
		auto function = std::make_unique<Function>();
		InitializeFunction(*function, nOuterPoints, targetMin, targetMax, targetMin, targetMax, rng);
		outer.push_back(std::move(function));
	}

	auto innerCopy = CopyVector(inner);
	auto outerCopy = CopyVector(outer);

	printf("Targets are determinants of random 4 * 4 matrices, %d training records\n", nTrainingRecords);
	for (int epoch = 0; epoch < 32; ++epoch) {

		//training
		std::vector<std::thread> threads;
		threads.emplace_back(Training, std::ref(inner), std::ref(outer), std::cref(features_training), std::cref(targets_training),
			nInner, nOuter, 0, nTrainingRecords / 2, alpha);
		threads.emplace_back(Training, std::ref(innerCopy), std::ref(outerCopy), std::cref(features_training), 
			std::cref(targets_training), nInner, nOuter, nTrainingRecords / 2, nTrainingRecords, alpha);
		for (auto& t : threads) {
			t.join();
		}

		//averaging models
		AverageVectors(inner, innerCopy);  //average result is sitting in first name, the last is unchanged
		AverageVectors(outer, outerCopy);

		innerCopy = CopyVector(inner);
		outerCopy = CopyVector(outer);

		//validation
		double pearson = Validation(inner, outer, features_validation, targets_validation, nInner, nOuter);

		current_time = clock();
		printf("%d pearson %4.3f, Time %2.3f\n", epoch, pearson, (double)(current_time - start_application) / CLOCKS_PER_SEC);
		if (pearson >= termination) break;
	}
	printf("\n");
}

int main() {
	Determinants44();
}


