//Concept: Andrew Polar and Mike Poluektov
//Developer Andrew Polar

// License
// If the end user somehow manages to make billions of US dollars using this code,
// and happens to meet the developer begging for change outside a McDonald's,
// they are under no obligation to buy the developer a sandwich.

// Symmetry Clause
// Likewise, if the developer becomes rich and famous by publishing this code,
// and meets an unfortunate end user who went bankrupt using it,
// the developer is also under no obligation to buy the end user a sandwich.

//Publications:
//https://www.sciencedirect.com/science/article/abs/pii/S0016003220301149
//https://www.sciencedirect.com/science/article/abs/pii/S0952197620303742
//https://link.springer.com/article/10.1007/s10994-025-06800-6

//Website:
//http://OpenKAN.org

#include <iostream>
#include <cmath>
#include <algorithm>
#include <thread>
#include <chrono>
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
	const std::vector<double>& targets, int nInner, int nOuter, int start, int end, int nRecords, double alpha) {

	size_t nFeatures = features[0].size();
	std::vector<double> models0(nInner);
	std::vector<double> models1(nOuter);
	std::vector<double> deltas0(nInner);
	std::vector<double> deltas1(nOuter);

	for (int idx = start; idx < end; ++idx) {
		int record = idx;
		if (record >= nRecords) record -= nRecords;
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
    //configuration
    //1.dataset
    const int nTrainingRecords = 100'000;
    const int nValidationRecords = 20'000;
    const int nMatrixSize = 4;
    const int nFeatures = nMatrixSize * nMatrixSize;
    const double min = 0.0;
    const double max = 10.0;

    //2.network
    const int nInner = 70;
    const int nOuter = 1;
    const double alpha = 0.2;
    const int nInnerPoints = 3;
    const int nOuterPoints = 20;
    const double termination = 0.97;

    //3.batches. all constants are arbitrary
    const int nBatchSize = 30'000;
    const int nBatches = 6;
    const int nLoops = 64;
    /////////////////////

    auto features_training = GenerateInput(nTrainingRecords, nFeatures, min, max);
    auto features_validation = GenerateInput(nValidationRecords, nFeatures, min, max);
    auto targets_training = ComputeDeterminantTarget(features_training, nMatrixSize);
    auto targets_validation = ComputeDeterminantTarget(features_validation, nMatrixSize);

    using Clock = std::chrono::steady_clock;
    auto start_application = Clock::now();

    double targetMin = *std::min_element(targets_training.begin(), targets_training.end());
    double targetMax = *std::max_element(targets_training.begin(), targets_training.end());

    std::random_device rd;
    std::mt19937 rng(rd());

    // Create containers sized to nBatches
    std::vector<std::vector<std::unique_ptr<Function>>> inners;
    std::vector<std::vector<std::unique_ptr<Function>>> outers;

    inners.resize(1);   // make sure index 0 exists
    outers.resize(1);

    // Fill batch 0
    inners[0].reserve(nInner * nFeatures);
    for (int i = 0; i < nInner * nFeatures; ++i) {
        auto function = std::make_unique<Function>();
        InitializeFunction(*function, nInnerPoints, min, max, targetMin, targetMax, rng);
        inners[0].push_back(std::move(function));
    }

    outers[0].reserve(nInner);
    for (int i = 0; i < nInner; ++i) {
        auto function = std::make_unique<Function>();
        InitializeFunction(*function, nOuterPoints, targetMin, targetMax, targetMin, targetMax, rng);
        outers[0].push_back(std::move(function));
    }

    // Copy to remaining batches
    for (int b = 1; b < nBatches; ++b) {
        inners.push_back(CopyVector(inners[0]));
        outers.push_back(CopyVector(outers[0]));
    }

    printf("Targets are determinants of random 4 * 4 matrices, %d training records\n", nTrainingRecords);
    int start = 0;
    std::vector<std::thread> threads;
    for (int loop = 0; loop < nLoops; ++loop) {

        // concurrent training of model copies
        threads.clear();
        for (int b = 0; b < nBatches; ++b) {
            int threadStart = start;
            int threadEnd = start + nBatchSize;
            // Launch thread to train inners[b] and outers[b]
            threads.emplace_back(Training, std::ref(inners[b]), std::ref(outers[b]),
                std::cref(features_training), std::cref(targets_training),
                nInner, nOuter, threadStart, threadEnd, nTrainingRecords, alpha);

            // advance start for next batch (wrap-around)
            start += nBatchSize;
            if (start >= nTrainingRecords) start -= nTrainingRecords;
        }

        for (auto& t : threads) {
            t.join();
        }

        // merging concurrently trained models into the first slot (inners[0], outers[0])
        for (int b = 1; b < nBatches; ++b) {
            AddVectors(inners[0], inners[b]); // sum into inners[0]
            AddVectors(outers[0], outers[b]); // sum into outers[0]
        }

        // average the summed model
        ScaleVectors(inners[0], 1.0 / static_cast<double>(nBatches));
        ScaleVectors(outers[0], 1.0 / static_cast<double>(nBatches));

        // redistribute averaged model to all batch copies for next loop
        for (int b = 1; b < nBatches; ++b) {
            inners[b] = CopyVector(inners[0]);
            outers[b] = CopyVector(outers[0]);
        }

        // validation every few loops
        if (0 == loop % 3 && loop > 0) {
            double pearson = Validation(inners[0], outers[0], features_validation, targets_validation, nInner, nOuter);
            auto current = Clock::now();
            double elapsed = std::chrono::duration<double>(current - start_application).count();
            printf("Loop = %d,  pearson = %4.3f, time = %2.3f\n", loop, pearson, elapsed);
            if (pearson >= termination) break;
        }
    }
    printf("\n");
}

int main() {
	Determinants44();
}





