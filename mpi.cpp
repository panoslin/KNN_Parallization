#include <stdio.h>
#include <stdlib.h>
#include <bits/stdc++.h>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"
#include "mpi.h"
#include <omp.h>


using namespace std;
const int TERMINATE_TAG = 2;
const int RESULT_TAG = 1;
// Calculates the distance between two instances
float distance(const float* instance_A, const float* instance_B, int num_attributes)
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
    bool operator()(const Candidate& lhs, const Candidate& rhs) const
    {
        return lhs.distance < rhs.distance;
    }
};

// Implements a MPI kNN where for each candidate query an in-place priority queue is maintained to identify the nearest neighbors
vector<int> KNN(ArffData* train, ArffData* test, int k, int mpi_rank, int mpi_num_processes)
{
    int num_classes = train->num_classes();
    int num_attributes = train->num_attributes();
    int train_num_instances = train->num_instances();
    int test_num_instances = test->num_instances();


    float* train_matrix = train->get_dataset_matrix();
    float* test_matrix = test->get_dataset_matrix();

    if (mpi_rank == 0) {
        MPI_Status stat;
        MPI_Request req;
        vector<int> predictions(test_num_instances);

        // send queryIndex for each process, excluding master process
        for (int queryIndex = 0; queryIndex < test_num_instances; ++queryIndex) {
            MPI_Send(&queryIndex, 1, MPI_INT, max(queryIndex % mpi_num_processes, 1), 0, MPI_COMM_WORLD);
        }

        // receive prediction from slave processes
        int maskClass;
        for (int queryIndex = 0; queryIndex < test_num_instances; ++queryIndex) {
            MPI_Recv(&maskClass, 1, MPI_INT, MPI_ANY_SOURCE, RESULT_TAG, MPI_COMM_WORLD, &stat);
            predictions[maskClass / num_classes] = maskClass % num_classes;
        }

        // send termination msg to all slave processes
        int terminate = -1;
        for (int rank = 1; rank < mpi_num_processes; rank++) {
            MPI_Send(&terminate, 1, MPI_INT, rank, TERMINATE_TAG, MPI_COMM_WORLD);
        }
        return predictions;
    }
    else {
        // receiveing tasks from rank 0
        int queryIndex;
        MPI_Status stat;
        vector<int> classCounts(num_classes, 0);
        while (true) {
            MPI_Recv(&queryIndex, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
            if (stat.MPI_TAG == TERMINATE_TAG) {
                // cout << "Rank " << mpi_rank << " terminating" << endl;
                break;
            }
            // cout << "Rank " << mpi_rank << " receives tasks " << queryIndex << endl;
            priority_queue<Candidate, vector<Candidate>, CandidateComparator> candidates;
            for (int keyIndex = 0; keyIndex < train_num_instances; ++keyIndex) {
                float dist = distance(&test_matrix[queryIndex * num_attributes],
                    &train_matrix[keyIndex * num_attributes],
                    num_attributes);

                int class_label = static_cast<int>(
                    train_matrix[keyIndex * num_attributes + num_attributes - 1]);

                Candidate candidate{ dist, class_label };
                if (static_cast<int>(candidates.size()) < k) {
                    // If the heap is not full, push the new candidate
                    candidates.push(candidate);
                }
                else if (dist < candidates.top().distance) {
                    // If the new candidate is closer than the farthest in the heap
                    candidates.pop();
                    candidates.push(candidate);
                }
            }

            // Collect class labels from the k nearest neighbors
            while (!candidates.empty()) {
                const Candidate& c = candidates.top();
                classCounts[c.class_label]++;
                candidates.pop();
            }

            // Determine the class with the highest vote
            int max_class = distance(classCounts.begin(),
                max_element(classCounts.begin(), classCounts.end()));

            int maskClass = queryIndex * num_classes + max_class;
            MPI_Send(&maskClass, 1, MPI_INT, 0, RESULT_TAG, MPI_COMM_WORLD);

            // Reset class counts
            fill(classCounts.begin(), classCounts.end(), 0);
        }
    }
}


vector<int> computeConfusionMatrix(const vector<int>& predictions, ArffData* dataset)
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

float computeAccuracy(const vector<int>& confusionMatrix, ArffData* dataset)
{
    int num_classes = dataset->num_classes();
    int successfulPredictions = 0;

    for (int i = 0; i < num_classes; ++i)
    {
        successfulPredictions += confusionMatrix[i * num_classes + i];
    }

    return 100.0f * successfulPredictions / dataset->num_instances();
}


int main(int argc, char* argv[])
{
    if (argc != 4)
    {
        cerr << "Usage: ./program datasets/train.arff datasets/test.arff k num_threads" << endl;
        return 1;
    }

    // k value for the k-nearest neighbors
    int k = stoi(argv[3]);

    int mpi_rank, mpi_num_processes;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_num_processes);

    // Open the datasets
    ArffParser parserTrain(argv[1]);
    ArffParser parserTest(argv[2]);
    ArffData* train = parserTrain.parse();
    ArffData* test = parserTest.parse();

    // Initialize time measurement
    auto start = chrono::steady_clock::now();

    vector<int> predictions = KNN(train, test, k, mpi_rank, mpi_num_processes);

    // Stop time measurement
    auto end = chrono::steady_clock::now();

    if (mpi_rank == 0) {
        // Compute the confusion matrix
        vector<int> confusionMatrix = computeConfusionMatrix(predictions, test);
        // Calculate the accuracy
        float accuracy = computeAccuracy(confusionMatrix, test);

        chrono::duration<double, milli> time_difference = end - start;

        cout << "The " << k << "-NN classifier for " << test->num_instances()
            << " test instances and " << train->num_instances()
            << " train instances required " << time_difference.count()
            << " ms CPU time for MPI with " << mpi_num_processes
            << " processes Accuracy was " << accuracy << "%" << endl;
    }

    MPI_Finalize();
}

/*  // Example to print the test dataset
    float* test_matrix = test->get_dataset_matrix();
    for(int i = 0; i < test->num_instances(); i++) {
        for(int j = 0; j < test->num_attributes(); j++)
            printf("%.0f, ", test_matrix[i*test->num_attributes() + j]);
        printf("\n");
    }
*/