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
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/pair.h>


using namespace std;


__global__ void find_d_knn(
    float* d_train_matrix,
    float* d_test_matrix,
    int* d_predicitons,
    thrust::pair<float, float>* d_knn,
    int k,
    int blockPerTestInstance,
    int trainInstancePerThread,
    int trainInstancePerBlock,
    int train_num_instances,
    int num_attributes,
    int num_classes,
    int threadPerBlock
)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    // 1. Find the test instance index
    int testInstanceIdx = blockIdx.x / blockPerTestInstance;

    // 2. Find the starting point of train instances for the current thread
    // find the idx of the current block for the current test instance
    // TODO: memory access pattern having strides
    int blockIdxForTestInstance = blockIdx.x % blockPerTestInstance;
    int trainInstanceIdx = blockIdxForTestInstance * trainInstancePerBlock + threadIdx.x * trainInstancePerThread;

    // 3. Calculate the #trainInstancePerThread consequtive train instances dist and store them thread locally
    float threadLocalKnn[2 * k];
    for (int i = 0; i < 2 * k; i++) { 
        threadLocalKnn[i] = FLT_MAX; 
    }

    for (int i = trainInstanceIdx; i < trainInstanceIdx + trainInstancePerThread && i < train_num_instances; i++) {
        // 1. calculate distance of the pair
        float dist = 0.0f;
        for (int attr_i = 0; attr_i < num_attributes - 1; ++attr_i) {
            float diff = d_train_matrix[i * num_attributes + attr_i] - d_test_matrix[testInstanceIdx * num_attributes + attr_i];
            dist += diff * diff;
        }
        // 2. wirte to threadLocalKnn
        for (int c = 0; c < k; c++) {
            if (dist < threadLocalKnn[2 * c]) {
                // Found a new candidate
                // Shift previous candidates down by one
                for (int x = k - 1; x > c; x--) {
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

    // 4. Find the block knn
    // store the knn for each thread
    __shared__ thrust::pair<float, float> blockLocalKnnCandidates[threadPerBlock * k];
    // find the thread local knn and write to shared memory blockLocalKnnCandidates
    for (int i = 0; i < k; i++) {
        int sharedMemIndex = threadIdx.x * k + i;
        if (threadLocalKnn[i * 2] != FLT_MAX) {
            blockLocalKnnCandidates[sharedMemIndex] = thrust::make_pair(
                threadLocalKnn[i * 2], 
                threadLocalKnn[i * 2 + 1]
            );
        }
    }

    // make sure all threads in the block have completed
    __syncthreads();

    // 5. Find the knn from the blockLocalKnnCandidates and write to d_knn
    // each thread divisible by 96 reduce the previous 96 or less instances to knn

}

__global__ void find_knn(
    float* d_train_matrix,
    float* d_test_matrix,
    int* d_predicitons,
    thrust::pair<float, float>* d_knn,
    int k,
    int blockPerTestInstance,
    int trainInstancePerThread,
    int trainInstancePerBlock,
    int train_num_instances
)
{

    // TODO: this should be the second kernel, so to gurantee the d_knn are synchronized
    // 6. find the majority if for each block knn
    // 7. write to d_predicitons
}

// Calculates the distance between two instances
__device__ float distance(const float* instance_A, const float* instance_B, int num_attributes)
{
    float sum = 0.0f;
    for (int i = 0; i < num_attributes - 1; ++i) { // Exclude the class label
        float diff = instance_A[i] - instance_B[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

// Candidate neighbor structure
struct Candidate {
    float distance;
    int class_label;
};

// Comparator for the priority queue (max-heap)
struct CandidateComparator {
    bool operator()(const Candidate& lhs, const Candidate& rhs) const
    {
        return lhs.distance < rhs.distance;
    }
};

vector<int> KNN(ArffData* train, ArffData* test, int k)
{
    int num_classes = train->num_classes();
    int num_attributes = train->num_attributes();
    int train_num_instances = train->num_instances();
    int test_num_instances = test->num_instances();

    vector<int> predictions(test_num_instances);

    float* train_matrix = train->get_dataset_matrix();
    float* test_matrix = test->get_dataset_matrix();

    // 0. Defined dims
    // Define a 1D of grid of 1D of block
    int threadPerBlock = 1024;
    // assuming the max shared memory space is 24KB (actually it's 48KB, just to be safe)
    // each shared mem will store pairs of (int, float) representing the (class, distance)
    int trainInstancePerBlock = 24 * 1024 / 8;
    int blockPerTestInstance = (train_num_instances + trainInstancePerBlock - 1) / trainInstancePerBlock;
    int blockPerGrid = blockPerTestInstance * test_num_instances;
    int trainInstancePerThread = trainInstancePerBlock / threadPerBlock;

    // 1. init mem
    float* d_train_matrix, * d_test_matrix;

    cudaMalloc(&d_train_matrix, sizeof(float) * num_attributes * train_num_instances);
    cudaMalloc(&d_test_matrix, sizeof(float) * num_attributes * test_num_instances);

    int* d_predictions;
    cudaMalloc(&d_predictions, sizeof(int) * test_num_instances);

    // store the knns for each local knn for each block for each test instance
    // each block will calculate the knn locally and write to this d_knn global memory
    thrust::pair<float, float>* d_knn;
    cudaMalloc((void**)&d_knn, test_num_instances * blockPerTestInstance * k * sizeof(thrust::pair<int, float>));

    // 2. Copy to device
    cudaMemcpy(d_train_matrix, train_matrix, sizeof(float) * num_attributes * train_num_instances, cudaMemcpyHostToDevice);
    cudaMemcpy(d_test_matrix, test_matrix, sizeof(float) * num_attributes * test_num_instances, cudaMemcpyHostToDevice);

    // Set init value to d_predicitons
    cudaMemset(d_predictions, 0, sizeof(int) * test_num_instances);

    // 4. Call kernel
    find_d_knn << < blockPerGrid, threadPerBlock >> > (
        d_train_matrix,
        d_test_matrix,
        d_predictions,
        d_knn,
        k,
        blockPerTestInstance,
        trainInstancePerThread,
        trainInstancePerBlock,
        train_num_instances,
        num_attributes,
        num_classes,
        threadPerBlock
    );

    find_knn << < blockPerGrid, threadPerBlock >> > (
        d_train_matrix,
        d_test_matrix,
        d_predictions,
        d_knn,
        k,
        blockPerTestInstance,
        trainInstancePerThread,
        trainInstancePerBlock,
        train_num_instances
        );

    // 5. Copy to host
    cudaMemcpy(predictions.data(), d_predictions, sizeof(int) * test_num_instances, cudaMemcpyDeviceToHost);

    // 6. Free memory
    cudaFree(d_train_matrix);
    cudaFree(d_test_matrix);
    cudaFree(d_predictions);
    cudaFree(d_knn);

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
    if (argc != 4) {
        cerr << "Usage: ./program datasets/train.arff datasets/test.arff k" << endl;
        return 1;
    }
    int k = stoi(argv[3]);

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

    vector<int> predictions = KNN(train, test, k);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Compute metrics
    vector<int> confusionMatrix = computeConfusionMatrix(predictions, test);
    float accuracy = computeAccuracy(confusionMatrix, test);


    cout << "The " << k << "-NN classifier for " << test->num_instances()
        << " test instances and " << train->num_instances()
        << " train instances required " << milliseconds
        << " ms CPU time for single-thread. Accuracy was "
        << accuracy << "%" << endl;

    return 0;
}