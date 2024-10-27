#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <chrono>
#include <algorithm>
#include <queue>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"
#include <cuda_runtime.h>


#define K 3
#define NUM_CLASSES 10

using namespace std;


__global__ void find_knn(
    float* d_train_matrix,
    float* d_test_matrix,
    int* d_predictions,
    int train_num_instances,
    int num_attributes
)
{
    // 1. each thread reduce 256 trainInstance to knn
    float threadLocalKnn[2 * K];
    for (int i = 0; i < 2 * K; i++) {
        threadLocalKnn[i] = FLT_MAX;
    }
    int testInstanceIdx = blockIdx.x;
    for (int i = threadIdx.x; i < train_num_instances; i += blockDim.x) {
        if (i < train_num_instances) {
            // 1. calculate distance of the pair
            float dist = 0.0f;
            for (int attr_i = 0; attr_i < num_attributes - 1; ++attr_i) {
                float diff = d_train_matrix[i * num_attributes + attr_i] - d_test_matrix[testInstanceIdx * num_attributes + attr_i];
                dist += diff * diff;
            }
            // 2. wirte to threadLocalKnn
            for (int c = 0; c < K; c++) {
                if (dist < threadLocalKnn[2 * c]) {
                    // Found a new candidate
                    // Shift previous candidates down by one
                    for (int x = K - 1; x > c; x--) {
                        threadLocalKnn[2 * x] = threadLocalKnn[2 * (x - 1)];
                        threadLocalKnn[2 * x + 1] = threadLocalKnn[2 * (x - 1) + 1];
                    }

                    // Set key vector as potential k NN
                    threadLocalKnn[2 * c] = dist;
                    threadLocalKnn[2 * c + 1] = d_train_matrix[i * num_attributes + num_attributes - 1];
                    break;
                }
            }
        }
    }

    // write to shared memory
    extern __shared__ float blockLocalKnn[];
    int offset = threadIdx.x * 2 * K;
    for (int i = 0; i < K; i++) {
        blockLocalKnn[offset + i * 2] = threadLocalKnn[i * 2];
        blockLocalKnn[offset + i * 2 + 1] = threadLocalKnn[i * 2 + 1];
    }
    __syncthreads();

    // 2. reduce the instance from the shared memory to knn
    if (threadIdx.x == 0) {
        float knn[2 * K];
        for (int i = 0; i < 2 * K; i++) {
            knn[i] = FLT_MAX;
        }
        for (int i = 0; i < blockDim.x * K; i++) {
            int idx = i * 2;
            if (blockLocalKnn[idx] != FLT_MAX) {
                // 1. calculate distance of the pair
                float dist = blockLocalKnn[idx];
                // 2. wirte to knn
                for (int c = 0; c < K; c++) {
                    if (dist < knn[2 * c]) {
                        // Found a new candidate
                        // Shift previous candidates down by one
                        for (int x = K - 1; x > c; x--) {
                            knn[2 * x] = knn[2 * (x - 1)];
                            knn[2 * x + 1] = knn[2 * (x - 1) + 1];
                        }

                        // Set key vector as potential k NN
                        knn[2 * c] = dist;
                        knn[2 * c + 1] = blockLocalKnn[idx + 1];
                        break;
                    }
                }
            }
            else break;
        }

        // 3. find majority and write to predictions
        // Bincount the candidate labels and pick the most common
        int classCounts[NUM_CLASSES] = { 0 };
        for (int i = 0; i < K; i++) {
            classCounts[(int)knn[2 * i + 1]] += 1;
        }

        int max_value = -1;
        int max_class = 0;
        for (int i = 0; i < NUM_CLASSES; i++) {
            if (classCounts[i] > max_value) {
                max_value = classCounts[i];
                max_class = i;
            }
        }

        // Make prediction with 
        d_predictions[testInstanceIdx] = max_class;
    }
}


vector<int> KNN(ArffData* train, ArffData* test, int k)
{
    int num_classes = train->num_classes();
    int num_attributes = train->num_attributes();
    int train_num_instances = train->num_instances();
    int test_num_instances = test->num_instances();

    vector<int> predictions(test_num_instances);

    float* train_matrix = train->get_dataset_matrix();
    float* test_matrix = test->get_dataset_matrix();

    // Defined dims
    // Define a 1D of grid of 1D of block
    int trainInstancePerThread = 256;
    int threadPerBlock = (train_num_instances + trainInstancePerThread - 1) / trainInstancePerThread;
    int blockPerGrid = test_num_instances;

    // 1. init mem
    float* d_train_matrix, * d_test_matrix;

    cudaMalloc(&d_train_matrix, sizeof(float) * num_attributes * train_num_instances);
    cudaMalloc(&d_test_matrix, sizeof(float) * num_attributes * test_num_instances);

    int* d_predictions;
    cudaMalloc(&d_predictions, sizeof(int) * test_num_instances);

    // 2. Copy to device
    cudaMemcpy(d_train_matrix, train_matrix, sizeof(float) * num_attributes * train_num_instances, cudaMemcpyHostToDevice);
    cudaMemcpy(d_test_matrix, test_matrix, sizeof(float) * num_attributes * test_num_instances, cudaMemcpyHostToDevice);

    // 3. Call kernel
    int sharedMemorySize = 2 * threadPerBlock * K * sizeof(float);
    find_knn << < blockPerGrid, threadPerBlock, sharedMemorySize >> > (
        d_train_matrix,
        d_test_matrix,
        d_predictions,
        train_num_instances,
        num_attributes
    );


    // 4. Copy to host
    cudaMemcpy(predictions.data(), d_predictions, sizeof(int) * test_num_instances, cudaMemcpyDeviceToHost);

    // 5. Free memory
    cudaFree(d_train_matrix);
    cudaFree(d_test_matrix);
    cudaFree(d_predictions);

    return predictions;
}

vector<int> computeConfusionMatrix(const vector<int>& predictions, ArffData* dataset)
{
    int num_classes = dataset->num_classes();
    int num_instances = dataset->num_instances();
    vector<int> confusionMatrix(num_classes * num_classes, 0);

    for (int i = 0; i < num_instances; ++i) {
        int trueClass = static_cast<int>(
            dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator float());
        int predictedClass = predictions[i];
        confusionMatrix[trueClass * num_classes + predictedClass]++;
    }

    return confusionMatrix;
}

float computeAccuracy(const vector<int>& confusionMatrix, ArffData* dataset)
{
    int num_classes = dataset->num_classes();
    int successfulPredictions = 0;

    for (int i = 0; i < num_classes; ++i) {
        successfulPredictions += confusionMatrix[i * num_classes + i];
    }

    return 100.0f * successfulPredictions / dataset->num_instances();
}

int main(int argc, char* argv[])
{
    if (argc != 3) {
        cerr << "Usage: ./program datasets/train.arff datasets/test.arff" << endl;
        return 1;
    }

    ArffParser parserTrain(argv[1]);
    ArffParser parserTest(argv[2]);
    ArffData* train = parserTrain.parse();
    ArffData* test = parserTest.parse();

    // Measure execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    cudaEventRecord(start);

    vector<int> predictions = KNN(train, test, K);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Compute metrics
    vector<int> confusionMatrix = computeConfusionMatrix(predictions, test);
    float accuracy = computeAccuracy(confusionMatrix, test);

    cout << "The " << K << "-NN classifier for " << test->num_instances()
        << " test instances and " << train->num_instances()
        << " train instances required " << milliseconds
        << " ms CPU time for CUDA. Accuracy was "
        << accuracy << "%" << endl;

    return 0;
}