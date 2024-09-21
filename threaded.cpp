#include <thread>
#include <memory>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <chrono>
#include <algorithm>
#include <queue>
#include <semaphore>
#include <optional>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"

using namespace std;

// semaphore to limit the working thread number 
optional<counting_semaphore<>> semaphore;

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

void find_knn(
    int queryIndex,
    int train_num_instances,
    int k,
    int num_classes,
    int num_attributes,
    float *train_matrix,
    float *test_matrix,
    shared_ptr<int> prediction)
{
    // limit number of working thread
    semaphore->acquire();
    // Priority queue to store the k nearest neighbors (max-heap based on distance)
    priority_queue<Candidate, vector<Candidate>, CandidateComparator> candidates;
    vector<int> classCounts(num_classes, 0);
    for (int keyIndex = 0; keyIndex < train_num_instances; ++keyIndex)
    {
        float dist = distance(
            &test_matrix[queryIndex * num_attributes],
            &train_matrix[keyIndex * num_attributes],
            num_attributes);

        int class_label = (int)(train_matrix[keyIndex * num_attributes + num_attributes - 1]);

        Candidate candidate{dist, class_label};

        if ((int)(candidates.size()) < k)
        {
            candidates.push(candidate);
        }
        else if (dist < candidates.top().distance)
        {
            candidates.pop();
            candidates.push(candidate);
        }
    }

    // Collect class labels from the k nearest neighbors
    while (!candidates.empty())
    {
        const Candidate &c = candidates.top();
        classCounts[c.class_label]++;
        candidates.pop();
    }

    // Determine the class with the highest vote
    int max_class = distance(classCounts.begin(),
                             max_element(classCounts.begin(), classCounts.end()));

    *prediction = max_class;
    semaphore->release();
}

// Implements a threaded kNN where for each candidate query an in-place priority queue is maintained to identify the nearest neighbors
vector<int> KNN(ArffData *train, ArffData *test, int k, int num_thread)
{
    int num_classes = train->num_classes();
    int num_attributes = train->num_attributes();
    int train_num_instances = train->num_instances();
    int test_num_instances = test->num_instances();

    vector<shared_ptr<int>> predictions_ptr(test_num_instances);

    float *train_matrix = train->get_dataset_matrix();
    float *test_matrix = test->get_dataset_matrix();

    vector<thread> threads;

    // put the outer loop to child threads
    for (int queryIndex = 0; queryIndex < test_num_instances; ++queryIndex)
    {
        predictions_ptr[queryIndex] = make_shared<int>();
        threads.emplace_back(
            find_knn,
            queryIndex,
            train_num_instances,
            k,
            num_classes,
            num_attributes,
            train_matrix,
            test_matrix,
            predictions_ptr[queryIndex]);
    }
    for (int queryIndex = 0; queryIndex < test_num_instances; ++queryIndex)
    {
        threads[queryIndex].join();
    }

    vector<int> predictions(test_num_instances);
    for (int queryIndex = 0; queryIndex < test_num_instances; ++queryIndex)
    {
        predictions[queryIndex] = *predictions_ptr[queryIndex];
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

    // instantiate the semaphore
    semaphore.emplace(num_threads);

    // Open the datasets
    ArffParser parserTrain(argv[1]);
    ArffParser parserTest(argv[2]);
    ArffData *train = parserTrain.parse();
    ArffData *test = parserTest.parse();

    // Initialize time measurement
    auto start = chrono::steady_clock::now();

    vector<int> predictions = KNN(train, test, k, num_threads);

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
         << " ms CPU time for threaded with " << num_threads
         << " threads. Accuracy was " << accuracy << "%" << endl;
}

/*  // Example to print the test dataset
    float* test_matrix = test->get_dataset_matrix();
    for(int i = 0; i < test->num_instances(); i++) {
        for(int j = 0; j < test->num_attributes(); j++)
            printf("%.0f, ", test_matrix[i*test->num_attributes() + j]);
        printf("\n");
    }
*/