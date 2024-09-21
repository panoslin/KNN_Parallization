#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <bits/stdc++.h>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"

using namespace std;

// Calculates the distance between two instances
float distance(const float *instance_A, const float *instance_B, int num_attributes)
{
    float sum = 0.0f;
    for (int i = 0; i < num_attributes - 1; ++i)
    { // Exclude the class label
        float diff = instance_A[i] - instance_B[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

// Candidate neighbor structure
struct Candidate
{
    float distance;
    int class_label;
};

// Comparator for the priority queue (max-heap)
struct CandidateComparator
{
    bool operator()(const Candidate &lhs, const Candidate &rhs) const
    {
        return lhs.distance < rhs.distance;
    }
};

// Implements a OpenMP kNN where for each candidate query an in-place priority queue is maintained to identify the nearest neighbors
vector<int> KNN(ArffData *train, ArffData *test, int k)
{
    int num_classes = train->num_classes();
    int num_attributes = train->num_attributes();
    int train_num_instances = train->num_instances();
    int test_num_instances = test->num_instances();

    vector<int> predictions(test_num_instances);

    float *train_matrix = train->get_dataset_matrix();
    float *test_matrix = test->get_dataset_matrix();

    // vector of PQ to store the k nearest neighbors (max-heap based on distance)
    vector<priority_queue<Candidate, vector<Candidate>, CandidateComparator>> candidates(test_num_instances);
    
    #pragma omp parallel for collapse(2)
    for (int queryIndex = 0; queryIndex < test_num_instances; ++queryIndex)
    {
        for (int keyIndex = 0; keyIndex < train_num_instances; ++keyIndex)
        {
            float dist = distance(&test_matrix[queryIndex * num_attributes],
                                  &train_matrix[keyIndex * num_attributes],
                                  num_attributes);

            int class_label = static_cast<int>(
                train_matrix[keyIndex * num_attributes + num_attributes - 1]);

            Candidate candidate{dist, class_label};

            if (static_cast<int>(candidates[queryIndex].size()) < k)
            {
                // If the heap is not full, push the new candidate
                candidates[queryIndex].push(candidate);
            }
            else if (dist < candidates[queryIndex].top().distance)
            {
                // If the new candidate is closer than the farthest in the heap
                candidates[queryIndex].pop();
                candidates[queryIndex].push(candidate);
            }
        }
    }

    vector<vector<int>> classCounts(test_num_instances, vector<int>(num_classes, 0));
    #pragma omp parallel for
    for (int queryIndex = 0; queryIndex < test_num_instances; queryIndex++) {

        // Collect class labels from the k nearest neighbors
        while (!candidates[queryIndex].empty())
        {
            const Candidate &c = candidates[queryIndex].top();
            classCounts[queryIndex][c.class_label]++;
            candidates[queryIndex].pop();
        }

        // Determine the class with the highest vote
        int max_class = distance(classCounts[queryIndex].begin(),
                                 max_element(classCounts[queryIndex].begin(), classCounts[queryIndex].end()));

        predictions[queryIndex] = max_class;
    }

    return predictions;
}

vector<int> computeConfusionMatrix(const vector<int> &predictions, ArffData *dataset)
{
    int num_classes = dataset->num_classes();
    int num_instances = dataset->num_instances();
    vector<int> confusionMatrix(num_classes * num_classes, 0);

    for (int i = 0; i < num_instances; ++i)
    {
        int trueClass = static_cast<int>(
            dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator float());
        int predictedClass = predictions[i];
        confusionMatrix[trueClass * num_classes + predictedClass]++;
    }

    return confusionMatrix;
}

float computeAccuracy(const vector<int> &confusionMatrix, ArffData *dataset)
{
    int num_classes = dataset->num_classes();
    int successfulPredictions = 0;

    for (int i = 0; i < num_classes; ++i)
    {
        successfulPredictions += confusionMatrix[i * num_classes + i];
    }

    return 100.0f * successfulPredictions / dataset->num_instances();
}

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        cerr << "Usage: ./program datasets/train.arff datasets/test.arff k num_threads" << endl;
        return 1;
    }

    // k value for the k-nearest neighbors
    int k = stoi(argv[3]);
    int num_threads = stoi(argv[4]);

    omp_set_num_threads(num_threads);

    // Open the datasets
    ArffParser parserTrain(argv[1]);
    ArffParser parserTest(argv[2]);
    ArffData *train = parserTrain.parse();
    ArffData *test = parserTest.parse();

    // Initialize time measurement
    auto start = chrono::steady_clock::now();

    vector<int> predictions = KNN(train, test, k);

    // Stop time measurement
    auto end = chrono::steady_clock::now();

    // Compute the confusion matrix
    vector<int> confusionMatrix = computeConfusionMatrix(predictions, test);
    // Calculate the accuracy
    float accuracy = computeAccuracy(confusionMatrix, test);

    chrono::duration<double, milli> time_difference = end - start;

    cout << "The " << k << "-NN classifier for " << test->num_instances()
         << " test instances and " << train->num_instances()
         << " train instances required " << time_difference.count()
         << " ms CPU time for OpenMP " << num_threads
         << " Accuracy was " << accuracy << "%" << endl;
}

/*  // Example to print the test dataset
    float* test_matrix = test->get_dataset_matrix();
    for(int i = 0; i < test->num_instances(); i++) {
        for(int j = 0; j < test->num_attributes(); j++)
            printf("%.0f, ", test_matrix[i*test->num_attributes() + j]);
        printf("\n");
    }
*/