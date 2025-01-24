#include "neural.hpp"

using namespace std;



int main() {
    int inputs = 7;
    int outputs = 7;

    Network::Network test(inputs, 16, 16, outputs);

    test.randInit();

    std::random_device rd; //seed
    std::mt19937 gen(rd()); //engine
    uniform_real_distribution<> distribution(30, 40);

    double arr[inputs];

    for (int i = 0; i < inputs; i++) {
        arr[i] = distribution(gen);
    }

    test.giveInputs(arr);

    double* returnArr = test.makeAMove();
};

Network::Network::Network(size_t inputs, size_t layers, size_t neurons, size_t outputs) {

    nInputs = inputs;
    nLayers = layers + 2; // Layers plus input layer and output layer
    nNeurons = neurons;
    nOutputs = outputs;

    data = (double**)calloc(nLayers, sizeof(double*));
    weight = (double***)calloc(nLayers, sizeof(double**));
    bias = (double**)calloc(nLayers, sizeof(double*));

    for (int i = 0; i < nLayers; i++) {
        // The first layer is always the inputs
        if (i == 0) {
            *(weight + i) = (double**)calloc(nInputs, sizeof(double*));
            *(data + i) = (double*)calloc(nInputs, sizeof(double));
            *(bias + i) = (double*)calloc(nInputs, sizeof(double));
            for (int j = 0; j < nInputs; j++) {
                *(*(weight + i) + j) = (double*)calloc(nNeurons, sizeof(double));
            }
        } else if (i == nLayers - 1) {
            *(data + i) = (double*)calloc(nOutputs, sizeof(double));
            *(bias + i) = (double*)calloc(nOutputs, sizeof(double));
        } else {
            *(weight + i) = (double**)calloc(nNeurons, sizeof(double*));
            *(data + i) = (double*)calloc(nNeurons, sizeof(double));
            *(bias + i) = (double*)calloc(nNeurons, sizeof(double));
            for (int j = 0; j < nNeurons; j++) {
                *(*(weight + i) + j) = (double*)calloc(nNeurons, sizeof(double));
            }
        }
    }
}

void Network::Network::randInit() {
    std::cout << "Initializing..." << std::endl;
    std::random_device rd; //seed
    std::mt19937 gen(rd()); //engine
    uniform_real_distribution<> distribution(-1.00, 1.00);
    for (int i = 0; i < nLayers; i++) {
        if (i == 0) {
            for (int j = 0; j < nInputs; j++) {
                *(*(bias + i) + j) = distribution(gen);
                for (int k = 0; k < nNeurons; k++) {
                    *(*(*(weight + i) + j) + k) = distribution(gen);
                }
            }
            continue;
        }
        if (i == nLayers - 1) {
            for (int j = 0; j < nOutputs; j++) {
                *(*(bias + i) + j) = distribution(gen);
            }
            continue;
        }
        for (int j = 0; j < nNeurons; j++) {
            *(*(bias + i) + j) = distribution(gen);
            for (int k = 0; k < nNeurons; k++) {
                *(*(*(weight + i) + j) + k) = distribution(gen);
            }
        }
    }
    std::cout << "Initialized" << std::endl;
}

void Network::Network::giveInputs(double* arr) {
    std::cout << "Loading Inputs..." << std::endl;
    for (int i = 0; i < nInputs; i++) {
        *(*data + i) = *(arr + i);
    }
    std::cout << "Inputs Loaded" << std::endl;
}

double Network::Network::ReLU(double x) {
    return x > 0 ? x : 0;
}

void Network::Network::calcLayer(int layerIndex) {

    if (layerIndex <= 0) return;

    std::cout << "Creating " << nNeurons << " Threads..." << std::endl;
    for (int i = 0; i < (layerIndex >= nLayers - 1 ? nOutputs : nNeurons); i++) {
        std::thread t(&Network::calcNeuron, this, layerIndex, i);
        threads.push_back(std::move(t));
    }
    std::cout << "Threads Created and pushed" << std::endl;

    std::cout << "Joining Threads..." << std::endl;
    for (std::thread& t : threads) {
        t.join();
    }
    std::cout << "Threads Joined" << std::endl;

    std::cout << "Clearing Completed Threads..." << std::endl;
    threads.clear();
    std::cout << "Threads Cleared" << std::endl;
}

void Network::Network::calcNeuron(int layerIndex, int neuronIndex) {
    int prevIndex = layerIndex - 1; // Where to retrieve data and weights from
    double sum = 0;
    for (int j = 0; j < ((layerIndex <= 1) ? nInputs : nNeurons); j++) {
        double temp = *(*(data + prevIndex) + j);
        sum += temp * (*(*(*(weight + prevIndex) + j) + neuronIndex));
    }
    *(*(data + layerIndex) + neuronIndex) = (layerIndex == nLayers - 1) ? sum + *(*(bias + layerIndex) + neuronIndex) : ReLU(sum + *(*(bias + layerIndex) + neuronIndex));
}
void Network::Network::calcOutputs(int layerIndex, int neuronIndex) {
    int prevIndex = layerIndex - 1; // Where to retrieve data and weights from
    double sum = 0;
    for (int j = 0; j < ((layerIndex == nLayers - 1) ? nOutputs : nNeurons); j++) {
        double temp = *(*(data + prevIndex) + j);
        sum += temp * ReLU(*(*(*(weight + prevIndex) + j) + neuronIndex));
    }
    *(*(data + layerIndex) + neuronIndex) = ReLU(sum);
}

double* Network::Network::makeAMove() {
    for (int i = 0; i < nLayers; i++) {
        calcLayer(i);
    }

    std::cout << "Done Calculations and Weighting" << std::endl;

    double* returnArr = *(data + nLayers - 1);

    for (int i = 0; i < nOutputs; i++) {
        std::cout << "Last Neuron " << i << " Has Output Value: " << *(*(data + nLayers - 1) + i) << std::endl;
    }

    return returnArr;
}

void backProp() {

}