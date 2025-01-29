#include "neural.hpp"

using namespace std;



int main() {
    Network::Network* test;

    Training::Training trainer(test, "./PCPM/2025/01/Fortune500_PCPM_2025-01-24.csv", 60, 4, 8);

    trainer.train("ACGL", 10000);

    std::cout << "Program Ending!" << std::endl;
};

Network::Network::Network(size_t inputs, size_t layers, size_t neurons, size_t outputs) {

    nInputs = inputs; // The amount of input nodes
    nLayers = layers + 2; // Layers plus input layer and output layer
    nNeurons = neurons; // The amount of neurons per layer
    nOutputs = outputs; // Amount of output nodes in the last layer

    data = (double**)calloc(nLayers, sizeof(double*)); // 2d Matrix to hold the output of each neuron (including initial input nodes)
    weight = (double***)calloc(nLayers, sizeof(double**)); // 3d Matrix to hold the weight between each neuron
    bias = (double**)calloc(nLayers, sizeof(double*)); // 2d Matrix to hold the bias for each neuron/node

    for (int i = 0; i < nLayers; i++) {
        // The first layer is always the inputs
        if (i == 0) {
            // Initializing the pointers within each array
            *(weight + i) = (double**)calloc(nInputs, sizeof(double*));
            *(data + i) = (double*)calloc(nInputs, sizeof(double));
            *(bias + i) = (double*)calloc(nInputs, sizeof(double));
            for (int j = 0; j < nInputs; j++) {
                *(*(weight + i) + j) = (double*)calloc(nNeurons, sizeof(double));
            }
        } else if (i == nLayers - 1) {
            // Output layer doesn't have weights
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
    uniform_real_distribution<> distribution(-1.00, 1.00); // Give random value from -1 to 1 to each neuron
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

void Network::Network::giveInputs(std::vector<double> vec) {
    for (int i = 0; i < nInputs; i++) {
        *(*data + i) = vec[i];
    }
}

double Network::Network::ReLU(double x) {
    return x > 0 ? x : 0.01;
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

double* Network::Network::output() {
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

Training::Data::Data(std::string filePath) {
    ifstream file;
    file.open(filePath); // Opening file

    if (!file.is_open()) {
        std::cout << "Error Opening File" << std::endl; // Error checking for file
    }

    std::string line; // Will store each line pulled from the file

    getline(file, line); // Pulling datetime line
    std::stringstream ss(line); // Making stringstream to separate out the individual datetimes

    std::string temp; // Will hold each entry in each line

    std::cout << "Pushing Dates..." << std::endl;
    while (getline(ss, temp, ',')) {
        if (temp.length() != 16) { // Correct format will have 16 characters until the year 10000 which is someone else's problem
            continue;
        }
        dateVector.push_back(parseDate(temp)); // Add the parsed vector to the collection of date vectors
    }
    std::cout << "Dates Pushed" << std::endl;

    int matrixIndex = 0; // To index matrix
    while (getline(file, line)) {
        std::vector<double> dataVector; // Created to push into the matrix to be accessed properly
        std::stringstream stream(line); // Stream to hold the parsable line
        parseLine(&dataVector, dateVector, std::move(stream)); // Initiating the parsing of the line
        dataMatrix.push_back(std::move(dataVector)); // Move the vector into the matrix
    }

    file.close(); // To close file
}

void Training::Data::parseLine(std::vector<double>* dataVector, std::vector<std::vector<double>> date, std::stringstream line) {
    std::string temp; // To hold each entry in the line
    getline(line, temp, ','); // Getting the ticker
    tickerVector.push_back(temp);
    oneHotTickers.push_back(0);
    int dateIndex = 0; // To make sure the dates line up with each entry
    while (getline(line, temp, ',')) {
        dataVector->push_back(std::stod(temp)); // Push the PCPM value into the vector
        dataVector->insert(dataVector->end(), date[dateIndex].begin(), date[dateIndex].end()); // Push parsed date info after it so it has date context for each value
        dateIndex++; // Increment index to track correct date
    }
}

std::vector<double> Training::Data::parseDate(std::string date) {
    std::vector<double> tempVector; // The return vector
    std::string tempString; // To hold separated number
    std::stringstream dateString(date); // To parse
    getline(dateString, tempString, '-'); // Get year
    tempVector.push_back(std::stod(tempString)); // Push year
    getline(dateString, tempString, '-'); // Get month
    tempVector.push_back(std::sin((2 * M_PI * std::stod(tempString)) / 12)); // Both for cyclical nature of monehts
    tempVector.push_back(std::cos((2 * M_PI * std::stod(tempString)) / 12));
    getline(dateString, tempString, '-'); // Get day
    tempVector.push_back(std::sin((2 * M_PI * std::stod(tempString)) / 31)); // Both for cylical nature of days
    tempVector.push_back(std::cos((2 * M_PI * std::stod(tempString)) / 31));
    getline(dateString, tempString, '-'); // Get hour
    tempVector.push_back(std::sin((2 * M_PI * std::stod(tempString)) / 24)); // Both for cyclical nature of hours
    tempVector.push_back(std::cos((2 * M_PI * std::stod(tempString)) / 24));
    getline(dateString, tempString, '-'); // Get minutes
    tempVector.push_back(std::sin((2 * M_PI * std::stod(tempString)) / 60)); // Both for cyclical nature of minutes
    tempVector.push_back(std::cos((2 * M_PI * std::stod(tempString)) / 60));
    return tempVector;
}

Training::Training::Training(Network::Network* network, std::string trainingDataFilePath, int minuteWindow, int hiddenLayers, int hiddenNeurons) {
    std::cout << "Creating Data Object..." << std::endl;
    dataObject = new Data(trainingDataFilePath);
    std::cout << "Data Object Created" << std::endl;
    std::cout << "Creating Network Object..." << std::endl;
    network = new Network::Network(minuteWindow * 10, hiddenLayers, hiddenNeurons, 1);
    std::cout << "Network Object Created" << std::endl;
    networkRef = network;
    std::cout << "Trainer Initialized" << std::endl;
}

void Training::Training::train(std::string tickerToTrain, int generations) {
    std::vector<double> slidingWindow; // The vector that will hold all concatenated info as we train
    auto tickerIterator = find(dataObject->tickerVector.begin(), dataObject->tickerVector.end(), tickerToTrain); // Getting the iterator that will give us the index of the ticker
    int tickerIndex = distance(dataObject->tickerVector.begin(), tickerIterator); // Getting the index of the ticker
    dataObject->oneHotTickers[tickerIndex] = 1; // Setting the onHotTickers value at the ticker index to 1 to indicate the ticker that we are using
    std::cout << "Ticker Index: " << tickerIndex << " Data Vector Size: " << dataObject->dataMatrix[tickerIndex].size() << " Inputs: " << networkRef->nInputs << std::endl;
    slidingWindow.insert(slidingWindow.begin(), dataObject->oneHotTickers.begin(), dataObject->oneHotTickers.end()); // Add the one hot vector to the input vector
    slidingWindow.insert(slidingWindow.begin(), dataObject->dataMatrix[tickerIndex].begin(), dataObject->dataMatrix[tickerIndex].begin() + networkRef->nInputs); // Add the ticker price info and date info to the input vector
    networkRef->randInit();
    networkRef->giveInputs(slidingWindow); // Load the input vector into the model

    for (int i = 0; i < generations; i++) {
        double error = dataObject->loss(networkRef, tickerIndex, i);
        backProp(networkRef, 0.1, error);
        //adjustInputs(networkRef, &slidingWindow, i, tickerIndex);
        std::cout << "Error For Generation " << i << " Was: " << error << std::endl;
    }
}

double Training::Data::loss(Network::Network* network, int tickerIndex, int generation) {
    double* output = network->output();
    double desiredOut = dataMatrix[tickerIndex][network->nInputs];
    return desiredOut - *output;
}

void Training::Training::adjustInputs(Network::Network* network, std::vector<double>* slidingWindow, int generation, int tickerIndex) {
    slidingWindow->erase(slidingWindow->begin(), slidingWindow->begin() + 11);
    slidingWindow->push_back(**(network->data + network->nLayers - 1));
    slidingWindow->insert(slidingWindow->end(), dataObject->dataMatrix[tickerIndex].begin() + (generation * 10) + network->nInputs + 1, dataObject->dataMatrix[tickerIndex].begin() + ((generation + 1) * 10) + network->nInputs);
}

void Training::Training::backProp(Network::Network* network, double learning_rate, double error) {
    std::vector<double*> deltas(network->nLayers); // Store layer errors

    // Allocate space for each layer's delta values
    for (int i = 0; i < network->nLayers; i++) {
        deltas[i] = (double*)calloc((i == network->nLayers - 1) ? network->nOutputs : network->nNeurons, sizeof(double));
    }

    // Compute output layer error
    for (int i = 0; i < network->nOutputs; i++) {
        double derivative = (*network->data[network->nLayers - 1] > 0) ? 1.0 : 0.0; // ReLU derivative
        deltas[network->nLayers - 1][i] = error * derivative;
    }

    // Backpropagate errors to hidden layers
    for (int l = network->nLayers - 2; l > 0; l--) { // Skip input layer
        for (int j = 0; j < network->nNeurons; j++) {
            double sum = 0.0;
            for (int k = 0; k < ((l == network->nLayers - 2) ? network->nOutputs : network->nNeurons); k++) {
                sum += deltas[l + 1][k] * network->weight[l][j][k];
            }
            double derivative = (network->data[l][j] > 0) ? 1.0 : 0.0; // ReLU derivative
            deltas[l][j] = sum * derivative;
        }
    }

    // Update Weights and Biases
    for (int l = 0; l < network->nLayers - 1; l++) { // Skip output layer for weight updates
        for (int j = 0; j < ((l == 0) ? network->nInputs : network->nNeurons); j++) {
            for (int k = 0; k < ((l == network->nLayers - 2) ? network->nOutputs : network->nNeurons); k++) {
                double grad = deltas[l + 1][k] * network->data[l][j];
                grad = std::max(-5.0, std::min(grad, 5.0)); // Gradient clipping
                network->weight[l][j][k] -= learning_rate * grad;
            }
        }
    }

    // Update biases
    for (int l = 1; l < network->nLayers; l++) {
        for (int j = 0; j < ((l == network->nLayers - 1) ? network->nOutputs : network->nNeurons); j++) {
            network->bias[l][j] -= learning_rate * deltas[l][j];
        }
    }

    // Free allocated memory
    for (int i = 0; i < network->nLayers; i++) {
        free(deltas[i]);
    }
}
