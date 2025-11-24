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
#include <atomic>
#include <mutex>
#include "Helper.h"
#include "Function.h"

double g_pearson = 0.0;   
std::mutex g_validationMutex;
std::atomic<bool> g_validationRunning(false);

void ValidationDeterminant(const std::vector<std::unique_ptr<Function>>& inner,
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
	g_pearson = Pearson(predictions, targets);
}

void TrainingDeterminant(std::vector<std::unique_ptr<Function>>& inner,
	std::vector<std::unique_ptr<Function>>& outer, const std::vector<std::vector<double>>& features,
	const std::vector<double>& targets, int nInner, int nOuter, int start, int end, int nRecords, double alpha, double accuracy) {

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
        if (std::abs(deltas1[0]) < accuracy) continue;
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

void DeterminantsParallel() {
    //configurations
    
    ////Matrices 5 * 5
    ////1. dataset
    //const int nTrainingRecords = 10'000'000;
    //const int nValidationRecords = 2'000'000;
    //const int nMatrixSize = 5;
    //const int nFeatures = nMatrixSize * nMatrixSize;
    //const double min = 0.0;
    //const double max = 10.0;

    ////2.network
    //const int nInner = 160;
    //const int nOuter = 1;
    //const double alpha = 0.1;
    //const int nInnerPoints = 3; 
    //const int nOuterPoints = 30; 
    //const double termination = 0.91;

    ////3.batches. all constants are arbitrary
    //const int nBatchSize = 50'000;
    //const int nBatches = 8;
    //const int nLoops = 200;
    /////////////////////////

    //Matrices 4 * 4
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
    const int nOuterPoints = 25;
    const double termination = 0.97;

    //3.batches. all constants are arbitrary
    const int nBatchSize = 45'000;
    const int nBatches = 6;
    const int nLoops = 10;
    /////////////////////

    //data generation
    printf("Generating data ...\n");
    auto features_training = GenerateInput(nTrainingRecords, nFeatures, min, max);
    auto features_validation = GenerateInput(nValidationRecords, nFeatures, min, max);
    auto targets_training = ComputeDeterminantTarget(features_training, nMatrixSize);
    auto targets_validation = ComputeDeterminantTarget(features_validation, nMatrixSize);
    printf("Data is ready ...\n");

    //processing start
    using Clock = std::chrono::steady_clock;
    auto start_application = Clock::now();

    double targetMin = *std::min_element(targets_training.begin(), targets_training.end());
    double targetMax = *std::max_element(targets_training.begin(), targets_training.end());
    double accuracy = std::abs(targetMin);
    if (accuracy < std::abs(targetMax)) accuracy = std::abs(targetMax);
    accuracy *= 0.01;
 
    std::random_device rd;
    std::mt19937 rng(rd());

    //create containers sized to nBatches
    std::vector<std::vector<std::unique_ptr<Function>>> inners;
    std::vector<std::vector<std::unique_ptr<Function>>> outers;

    inners.resize(1);   // make sure index 0 exists
    outers.resize(1);

    //generate one set as random
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

    //copy to remaining sets
    for (int b = 1; b < nBatches; ++b) {
        inners.push_back(CopyVector(inners[0]));
        outers.push_back(CopyVector(outers[0]));
    }

	printf("Parallel version\n");
    printf("Targets are determinants of random %d * %d matrices, %d training records\n", 
        nMatrixSize, nMatrixSize, nTrainingRecords);
    g_pearson = 0.0;
    int start = 0;
    std::vector<std::thread> threads;
    for (int loop = 0; loop < nLoops; ++loop) {
        // concurrent training of model copies
        threads.clear();
        for (int b = 0; b < nBatches; ++b) {
            int threadStart = start;
            int threadEnd = start + nBatchSize;
            // Launch thread to train inners[b] and outers[b]
            threads.emplace_back(TrainingDeterminant, std::ref(inners[b]), std::ref(outers[b]),
                std::cref(features_training), std::cref(targets_training),
                nInner, nOuter, threadStart, threadEnd, nTrainingRecords, alpha, accuracy);

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
            CopyVector(inners[0], inners[b]);
            CopyVector(outers[0], outers[b]);
        }

        ////this can work only for short validation data, such as 4 * 4 matrices
        ////asynchronous validation every few loops
        //if (0 == loop % 3 && loop > 0) {
        //    auto innerCopy = CopyVector(inners[0]);
        //    auto outerCopy = CopyVector(outers[0]);
        //    std::thread([innerCopy = std::move(innerCopy),
        //        outerCopy = std::move(outerCopy),
        //        &features_validation, &targets_validation,
        //        nInner, nOuter]() mutable
        //        {
        //            std::lock_guard<std::mutex> lock(g_validationMutex); // optional
        //            Validation(innerCopy, outerCopy, features_validation, targets_validation, nInner, nOuter);
        //            g_validationRunning = false;
        //        }).detach();
        //}

        auto current = Clock::now();
        double elapsed = std::chrono::duration<double>(current - start_application).count();
        printf("Loop = %d,  pearson = %4.3f, time = %2.3f\n", loop, g_pearson, elapsed);
        if (g_pearson >= termination) break;
    }
    printf("Validation ...\n");
    ValidationDeterminant(inners[0], outers[0], features_validation, targets_validation, nInner, nOuter);
    printf("Pearson %f\n\n", g_pearson);
}

void DeterminantsSequential() {
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

	// Instantiate models
	std::random_device rd;
	std::mt19937 rng(rd());
	std::vector<std::unique_ptr<Function>> innerFunctions;
	for (int i = 0; i < nInner * nFeatures; ++i) {
		auto function = std::make_unique<Function>();
		InitializeFunction(*function, nInnerPoints, min, max, targetMin, targetMax, rng);
		innerFunctions.push_back(std::move(function));
	}
	std::vector<std::unique_ptr<Function>> outerFunctions;
	for (int i = 0; i < nInner; ++i) {
		auto function = std::make_unique<Function>();
		InitializeFunction(*function, nOuterPoints, targetMin, targetMax, targetMin, targetMax, rng);
		outerFunctions.push_back(std::move(function));
	}

	//auxiliary buffers
	std::vector<double> models0(nInner);
	std::vector<double> models1(nOuter);

	std::vector<double> deltas0(nInner);
	std::vector<double> deltas1(nOuter);
	std::vector<double> predictions(nValidationRecords);

	//training
	printf("Sequential version\n");
	printf("Targets are determinants of random 4 * 4 matrices, %d training records\n", nTrainingRecords);
	for (int epoch = 0; epoch < 32; ++epoch) {
		for (int record = 0; record < nTrainingRecords; ++record) {
			for (int k = 0; k < nInner; ++k) {
				models0[k] = 0.0;
				for (int j = 0; j < nFeatures; ++j) {
					models0[k] += Compute(features_training[record][j], false, *innerFunctions[k * nFeatures + j]);
				}
				models0[k] /= nFeatures;
			}

			for (int k = 0; k < nOuter; ++k) {
				models1[k] = 0.0;
				for (int j = 0; j < nInner; ++j) {
					models1[k] += Compute(models0[j], false, *outerFunctions[j]);
				}
				models1[k] /= nInner;
			}

			//in general, for vector target, deltas are vectors and derivatives 
			//are matrices, and computing of next deltas is matrix vector 
			//multiplication, but this is particular case with scalar target
			deltas1[0] = alpha * (targets_training[record] - models1[0]);
			for (int j = 0; j < nInner; ++j) {
				deltas0[j] = deltas1[0] * ComputeDerivative(*outerFunctions[j]);
			}

			for (int k = 0; k < nOuter; ++k) {
				for (int j = 0; j < nInner; ++j) {
					Update(deltas1[k], *outerFunctions[j]);
				}
			}

			for (int k = 0; k < nInner; ++k) {
				for (int j = 0; j < nFeatures; ++j) {
					Update(deltas0[k], *innerFunctions[k * nFeatures + j]);
				}
			}
		}

		//validation
		for (int record = 0; record < nValidationRecords; ++record) {
			for (int k = 0; k < nInner; ++k) {
				models0[k] = 0.0;
				for (int j = 0; j < nFeatures; ++j) {
					models0[k] += Compute(features_validation[record][j], true, *innerFunctions[k * nFeatures + j]);
				}
				models0[k] /= nFeatures;
			}
			for (int k = 0; k < nOuter; ++k) {
				models1[k] = 0.0;
				for (int j = 0; j < nInner; ++j) {
					models1[k] += Compute(models0[j], true, *outerFunctions[j]);
				}
				models1[k] /= nInner;
			}
			predictions[record] = models1[0];
		}
		double pearson = Pearson(predictions, targets_validation);

		current_time = clock();
		printf("%d pearson %4.3f, Time %2.3f\n", epoch, pearson, (double)(current_time - start_application) / CLOCKS_PER_SEC);
		if (pearson >= termination) break;
	}
	printf("\n");
}

void TetrahedronsSequential() {
	//data
	const int nTrainingRecords = 500'000;
	const int nValidationRecords = 50'000;
	const int nFeatures = 12;
	const int nTargets = 4;
	const double min = 0.0;
	const double max = 10.0;

	//generation
	auto features_training = MakeRandomMatrix(nTrainingRecords, nFeatures, min, max);
	auto features_validation = MakeRandomMatrix(nValidationRecords, nFeatures, min, max);
	auto targets_training = ComputeTargetMatrix(features_training);
	auto targets_validation = ComputeTargetMatrix(features_validation);

	//data is ready, we start training
	clock_t start_application = clock();
	clock_t current_time = clock();

	double targetMin = targets_training[0][0];
	double targetMax = targets_training[0][0];
	for (int i = 0; i < nTrainingRecords; ++i) {
		for (int j = 0; j < nTargets; ++j) {
			if (targets_training[i][j] < targetMin) targetMin = targets_training[i][j];
			if (targets_training[i][j] > targetMax) targetMax = targets_training[i][j];
		}
	}

	const int nU0 = 70;
	const int nU1 = 30;
	const int nU2 = nTargets;
	const double alpha = 0.05;
	const int nPoints0 = 3;
	const int nPoints1 = 18;
	const int nPoints2 = 22;
	const int nEpochs = 64;
	const double termination = 0.99;

	//Instantiate models
	std::random_device rd;
	std::mt19937 rng(rd());

	std::vector<std::unique_ptr<Function>> layer0;
	for (int i = 0; i < nU0 * nFeatures; ++i) {
		auto function = std::make_unique<Function>();
		InitializeFunction(*function, nPoints0, min, max, targetMin, targetMax, rng);
		layer0.push_back(std::move(function));
	}

	std::vector<std::unique_ptr<Function>> layer1;
	for (int i = 0; i < nU1 * nU0; ++i) {
		auto function = std::make_unique<Function>();
		InitializeFunction(*function, nPoints1, targetMin, targetMax, targetMin, targetMax, rng);
		layer1.push_back(std::move(function));
	}

	std::vector<std::unique_ptr<Function>> layer2;
	for (int i = 0; i < nU2 * nU1; ++i) {
		auto function = std::make_unique<Function>();
		InitializeFunction(*function, nPoints2, targetMin, targetMax, targetMin, targetMax, rng);
		layer2.push_back(std::move(function));
	}

	//auxiliary buffers
	std::vector<double> models0(nU0);
	std::vector<double> models1(nU1);
	std::vector<double> models2(nU2);

	std::vector<std::vector<double>> derivatives1(nU2, std::vector<double>(nU1));
	std::vector<std::vector<double>> derivatives0(nU1, std::vector<double>(nU0));

	std::vector<double> deltas2(nU2);
	std::vector<double> deltas1(nU1);
	std::vector<double> deltas0(nU0);

	auto actual0 = std::vector<double>(nValidationRecords);
	auto actual1 = std::vector<double>(nValidationRecords);
	auto actual2 = std::vector<double>(nValidationRecords);
	auto actual3 = std::vector<double>(nValidationRecords);

	auto computed0 = std::vector<double>(nValidationRecords);
	auto computed1 = std::vector<double>(nValidationRecords);
	auto computed2 = std::vector<double>(nValidationRecords);
	auto computed3 = std::vector<double>(nValidationRecords);

	printf("Sequential version\n");
	printf("Targets are areas of faces of random tetrahedrons, %d\n", nTrainingRecords);
	for (int epoch = 0; epoch < nEpochs; ++epoch) {
		//training
		for (int record = 0; record < nTrainingRecords; ++record) {
			//steps: forward pass layer by layer
			for (int k = 0; k < nU0; ++k) {
				models0[k] = 0.0;
				for (int j = 0; j < nFeatures; ++j) {
					models0[k] += Compute(features_training[record][j], false, *layer0[k * nFeatures + j]);
				}
				models0[k] /= nFeatures;
			}
			for (int k = 0; k < nU1; ++k) {
				models1[k] = 0.0;
				for (int j = 0; j < nU0; ++j) {
					models1[k] += Compute(models0[j], false, *layer1[k * nU0 + j]);
				}
				models1[k] /= nU0;
			}
			for (int k = 0; k < nU2; ++k) {
				models2[k] = 0.0;
				for (int j = 0; j < nU1; ++j) {
					models2[k] += Compute(models1[j], false, *layer2[k * nU1 + j]);
				}
				models2[k] /= nU1;
			}

			//compute all derivative matrices
			for (int k = 0; k < nU2; ++k) {
				for (int j = 0; j < nU1; ++j) {
					derivatives1[k][j] = ComputeDerivative(*layer2[k * nU1 + j]);
				}
			}
			for (int k = 0; k < nU1; ++k) {
				for (int j = 0; j < nU0; ++j) {
					derivatives0[k][j] = ComputeDerivative(*layer1[k * nU0 + j]);
				}
			}

			//compute deltas
			for (int j = 0; j < nU2; ++j) {
				deltas2[j] = (targets_training[record][j] - models2[j]) * alpha;
			}

			for (int j = 0; j < nU1; ++j) {
				deltas1[j] = 0.0;
				for (int i = 0; i < nU2; ++i) {
					deltas1[j] += derivatives1[i][j] * deltas2[i];
				}
			}
			for (int j = 0; j < nU0; ++j) {
				deltas0[j] = 0.0;
				for (int i = 0; i < nU1; ++i) {
					deltas0[j] += derivatives0[i][j] * deltas1[i];
				}
			}

			//step: update all layers
			for (int k = 0; k < nU2; ++k) {
				for (int j = 0; j < nU1; ++j) {
					Update(deltas2[k], *layer2[k * nU1 + j]);
				}
			}
			for (int k = 0; k < nU1; ++k) {
				for (int j = 0; j < nU0; ++j) {
					Update(deltas1[k], *layer1[k * nU0 + j]);
				}
			}
			for (int k = 0; k < nU0; ++k) {
				for (int j = 0; j < nFeatures; ++j) {
					Update(deltas0[k], *layer0[k * nFeatures + j]);
				}
			}
		}

		//validation
		for (int record = 0; record < nValidationRecords; ++record) {
			for (int k = 0; k < nU0; ++k) {
				models0[k] = 0.0;
				for (int j = 0; j < nFeatures; ++j) {
					models0[k] += Compute(features_training[record][j], true, *layer0[k * nFeatures + j]);
				}
				models0[k] /= nFeatures;
			}
			for (int k = 0; k < nU1; ++k) {
				models1[k] = 0.0;
				for (int j = 0; j < nU0; ++j) {
					models1[k] += Compute(models0[j], true, *layer1[k * nU0 + j]);
				}
				models1[k] /= nU0;
			}
			for (int k = 0; k < nU2; ++k) {
				models2[k] = 0.0;
				for (int j = 0; j < nU1; ++j) {
					models2[k] += Compute(models1[j], true, *layer2[k * nU1 + j]);
				}
				models2[k] /= nU1;
			}

			actual0[record] = targets_validation[record][0];
			actual1[record] = targets_validation[record][1];
			actual2[record] = targets_validation[record][2];
			actual3[record] = targets_validation[record][3];

			computed0[record] = models2[0];
			computed1[record] = models2[1];
			computed2[record] = models2[2];
			computed3[record] = models2[3];
		}
		double p1 = Pearson(computed0, actual0);
		double p2 = Pearson(computed1, actual1);
		double p3 = Pearson(computed2, actual2);
		double p4 = Pearson(computed3, actual3);

		current_time = clock();
		printf("Epoch %d, Pearsons: %f %f %f %f, time %2.3f\n", epoch, p1, p2, p3, p4,
			(double)(current_time - start_application) / CLOCKS_PER_SEC);

		int cnt = 0;
		if (p1 >= termination) ++cnt;
		if (p2 >= termination) ++cnt;
		if (p3 >= termination) ++cnt;
		if (p4 >= termination) ++cnt;

		if (cnt >= 2) break;
	}
	printf("\n");
}

void TrainingTetrahedron(
	std::vector<std::unique_ptr<Function>>& layer0,
	std::vector<std::unique_ptr<Function>>& layer1, 
	std::vector<std::unique_ptr<Function>>& layer2,
	const std::vector<std::vector<double>>& features,
	const std::vector<std::vector<double>>& targets,
	int nU0, int nU1, int nU2, int start, int end, int nRecords, double alpha) {

	size_t nFeatures = features[0].size();
	std::vector<double> models0(nU0);
	std::vector<double> models1(nU1);
	std::vector<double> models2(nU2);

	std::vector<std::vector<double>> derivatives1(nU2, std::vector<double>(nU1));
	std::vector<std::vector<double>> derivatives0(nU1, std::vector<double>(nU0));

	std::vector<double> deltas2(nU2);
	std::vector<double> deltas1(nU1);
	std::vector<double> deltas0(nU0);

	for (int idx = start; idx < end; ++idx) {
		int record = idx;
		if (record >= nRecords) record -= nRecords;
		for (int k = 0; k < nU0; ++k) {
			models0[k] = 0.0;
			for (size_t j = 0; j < nFeatures; ++j) {
				models0[k] += Compute(features[record][j], false, *layer0[k * nFeatures + j]);
			}
			models0[k] /= nFeatures;
		}
		for (int k = 0; k < nU1; ++k) {
			models1[k] = 0.0;
			for (int j = 0; j < nU0; ++j) {
				models1[k] += Compute(models0[j], false, *layer1[k * nU0 + j]);
			}
			models1[k] /= nU0;
		}
		for (int k = 0; k < nU2; ++k) {
			models2[k] = 0.0;
			for (int j = 0; j < nU1; ++j) {
				models2[k] += Compute(models1[j], false, *layer2[k * nU1 + j]);
			}
			models2[k] /= nU1;
		}

		//compute all derivative matrices
		for (int k = 0; k < nU2; ++k) {
			for (int j = 0; j < nU1; ++j) {
				derivatives1[k][j] = ComputeDerivative(*layer2[k * nU1 + j]);
			}
		}
		for (int k = 0; k < nU1; ++k) {
			for (int j = 0; j < nU0; ++j) {
				derivatives0[k][j] = ComputeDerivative(*layer1[k * nU0 + j]);
			}
		}

		//compute deltas
		for (int j = 0; j < nU2; ++j) {
			deltas2[j] = (targets[record][j] - models2[j]) * alpha;
		}

		for (int j = 0; j < nU1; ++j) {
			deltas1[j] = 0.0;
			for (int i = 0; i < nU2; ++i) {
				deltas1[j] += derivatives1[i][j] * deltas2[i];
			}
		}
		for (int j = 0; j < nU0; ++j) {
			deltas0[j] = 0.0;
			for (int i = 0; i < nU1; ++i) {
				deltas0[j] += derivatives0[i][j] * deltas1[i];
			}
		}

		//step: update all layers
		for (int k = 0; k < nU2; ++k) {
			for (int j = 0; j < nU1; ++j) {
				Update(deltas2[k], *layer2[k * nU1 + j]);
			}
		}
		for (int k = 0; k < nU1; ++k) {
			for (int j = 0; j < nU0; ++j) {
				Update(deltas1[k], *layer1[k * nU0 + j]);
			}
		}
		for (int k = 0; k < nU0; ++k) {
			for (size_t j = 0; j < nFeatures; ++j) {
				Update(deltas0[k], *layer0[k * nFeatures + j]);
			}
		}
	}
}

void TetrahedronParallel() {
	//data
	const int nTrainingRecords = 500'000;
	const int nValidationRecords = 50'000;
	const int nFeatures = 12;
	const int nTargets = 4;
	const double min = 0.0;
	const double max = 10.0;

	const int nU0 = 70;
	const int nU1 = 30;
	const int nU2 = nTargets;
	const double alpha = 0.05;
	const int nPoints0 = 3;
	const int nPoints1 = 18;
	const int nPoints2 = 22;
	const double termination = 0.99;

	const int nBatchSize = 50'000;
	const int nBatches = 6;
	const int nLoops = 10;
	/////////////////////

	//generation
	printf("Generating data ...\n");
	auto features_training = MakeRandomMatrix(nTrainingRecords, nFeatures, min, max);
	auto features_validation = MakeRandomMatrix(nValidationRecords, nFeatures, min, max);
	auto targets_training = ComputeTargetMatrix(features_training);
	auto targets_validation = ComputeTargetMatrix(features_validation);
	printf("Data is ready\n");

	//processing start
	using Clock = std::chrono::steady_clock;
	auto start_application = Clock::now();

	double targetMin = targets_training[0][0];
	double targetMax = targets_training[0][0];
	for (int i = 0; i < nTrainingRecords; ++i) {
		for (int j = 0; j < nTargets; ++j) {
			if (targets_training[i][j] < targetMin) targetMin = targets_training[i][j];
			if (targets_training[i][j] > targetMax) targetMax = targets_training[i][j];
		}
	}

	std::random_device rd;
	std::mt19937 rng(rd());

	//create containers sized to nBatches
	std::vector<std::vector<std::unique_ptr<Function>>> layer0;
	std::vector<std::vector<std::unique_ptr<Function>>> layer1;
	std::vector<std::vector<std::unique_ptr<Function>>> layer2;

	layer0.resize(1);   // make sure index 0 exists
	layer1.resize(1);
	layer2.resize(1);

	//generate one set as random
	layer0[0].reserve(nU0 * nFeatures);
	for (int i = 0; i < nU0 * nFeatures; ++i) {
		auto function = std::make_unique<Function>();
		InitializeFunction(*function, nPoints0, min, max, targetMin, targetMax, rng);
		layer0[0].push_back(std::move(function));
	}

	layer1[0].reserve(nU1 * nU0);
	for (int i = 0; i < nU1 * nU0; ++i) {
		auto function = std::make_unique<Function>();
		InitializeFunction(*function, nPoints1, targetMin, targetMax, targetMin, targetMax, rng);
		layer1[0].push_back(std::move(function));
	}

	layer2[0].reserve(nU1 * nU2);
	for (int i = 0; i < nU1 * nU2; ++i) {
		auto function = std::make_unique<Function>();
		InitializeFunction(*function, nPoints2, targetMin, targetMax, targetMin, targetMax, rng);
		layer2[0].push_back(std::move(function));
	}

	//copy to remaining sets
	for (int b = 1; b < nBatches; ++b) {
		layer0.push_back(CopyVector(layer0[0]));
		layer1.push_back(CopyVector(layer1[0]));
		layer2.push_back(CopyVector(layer2[0]));
	}

	printf("Parallel version\n");
	printf("Targets are areas of faces of random tetrahedrons, %d\n", nTrainingRecords);
	g_pearson = 0.0;
	int start = 0;
	std::vector<std::thread> threads;
	for (int loop = 0; loop < nLoops; ++loop) {
		// concurrent training of model copies
		threads.clear();
		for (int b = 0; b < nBatches; ++b) {
			int threadStart = start;
			int threadEnd = start + nBatchSize;
			// Launch thread to train inners[b] and outers[b]
			threads.emplace_back(TrainingTetrahedron, 
				std::ref(layer0[b]), std::ref(layer1[b]), std::ref(layer2[b]),
				std::cref(features_training), std::cref(targets_training),
				nU0, nU1, nU2, threadStart, threadEnd, nTrainingRecords, alpha);

			// advance start for next batch (wrap-around)
			start += nBatchSize;
			if (start >= nTrainingRecords) start -= nTrainingRecords;
		}

		for (auto& t : threads) {
			t.join();
		}

		// merging concurrently trained models into the first slot (inners[0], outers[0])
		for (int b = 1; b < nBatches; ++b) {
			AddVectors(layer0[0], layer0[b]); 
			AddVectors(layer1[0], layer1[b]); 
			AddVectors(layer2[0], layer2[b]);
		}

		// average the summed model
		ScaleVectors(layer0[0], 1.0 / static_cast<double>(nBatches));
		ScaleVectors(layer1[0], 1.0 / static_cast<double>(nBatches));
		ScaleVectors(layer2[0], 1.0 / static_cast<double>(nBatches));

		// redistribute averaged model to all batch copies for next loop
		for (int b = 1; b < nBatches; ++b) {
			CopyVector(layer0[0], layer0[b]);
			CopyVector(layer1[0], layer1[b]);
			CopyVector(layer2[0], layer2[b]);
		}

		////this can work only for short validation data, such as 4 * 4 matrices
		////asynchronous validation every few loops
		//if (0 == loop % 3 && loop > 0) {
		//    auto innerCopy = CopyVector(inners[0]);
		//    auto outerCopy = CopyVector(outers[0]);
		//    std::thread([innerCopy = std::move(innerCopy),
		//        outerCopy = std::move(outerCopy),
		//        &features_validation, &targets_validation,
		//        nInner, nOuter]() mutable
		//        {
		//            std::lock_guard<std::mutex> lock(g_validationMutex); // optional
		//            Validation(innerCopy, outerCopy, features_validation, targets_validation, nInner, nOuter);
		//            g_validationRunning = false;
		//        }).detach();
		//}

		auto current = Clock::now();
		double elapsed = std::chrono::duration<double>(current - start_application).count();
		printf("Loop = %d,  pearson = %4.3f, time = %2.3f\n", loop, g_pearson, elapsed);
		if (g_pearson >= termination) break;
	}

	//validation
	std::vector<double> models0(nU0);
	std::vector<double> models1(nU1);
	std::vector<double> models2(nU2);

	auto actual0 = std::vector<double>(nValidationRecords);
	auto actual1 = std::vector<double>(nValidationRecords);
	auto actual2 = std::vector<double>(nValidationRecords);
	auto actual3 = std::vector<double>(nValidationRecords);

	auto computed0 = std::vector<double>(nValidationRecords);
	auto computed1 = std::vector<double>(nValidationRecords);
	auto computed2 = std::vector<double>(nValidationRecords);
	auto computed3 = std::vector<double>(nValidationRecords);

	for (int record = 0; record < nValidationRecords; ++record) {
		for (int k = 0; k < nU0; ++k) {
			models0[k] = 0.0;
			for (int j = 0; j < nFeatures; ++j) {
				models0[k] += Compute(features_training[record][j], true, *layer0[0][k * nFeatures + j]);
			}
			models0[k] /= nFeatures;
		}
		for (int k = 0; k < nU1; ++k) {
			models1[k] = 0.0;
			for (int j = 0; j < nU0; ++j) {
				models1[k] += Compute(models0[j], true, *layer1[0][k * nU0 + j]);
			}
			models1[k] /= nU0;
		}
		for (int k = 0; k < nU2; ++k) {
			models2[k] = 0.0;
			for (int j = 0; j < nU1; ++j) {
				models2[k] += Compute(models1[j], true, *layer2[0][k * nU1 + j]);
			}
			models2[k] /= nU1;
		}

		actual0[record] = targets_validation[record][0];
		actual1[record] = targets_validation[record][1];
		actual2[record] = targets_validation[record][2];
		actual3[record] = targets_validation[record][3];

		computed0[record] = models2[0];
		computed1[record] = models2[1];
		computed2[record] = models2[2];
		computed3[record] = models2[3];
	}
	double p1 = Pearson(computed0, actual0);
	double p2 = Pearson(computed1, actual1);
	double p3 = Pearson(computed2, actual2);
	double p4 = Pearson(computed3, actual3);

	printf("Validation, pearsons: %f %f %f %f\n", p1, p2, p3, p4);
}

int main() {
	//DeterminantsSequential();
	DeterminantsParallel();
	//TetrahedronsSequential();
	//TetrahedronParallel();
}
