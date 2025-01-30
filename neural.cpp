#include "neural.hpp"

using namespace std;



int main() {
    Network::Network* test;

    Training::Training trainer(test, "./PCPM/2025/01/Fortune500_PCPM_2025-01-24.csv", 60, 16, 16);

    trainer.train(329);

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
    uniform_real_distribution<> distribution(-0.01, 0.01); // Give random value from -1 to 1 to each neuron
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
    //std::cout << "Loading Inputs..." << std::endl;
    for (int i = 0; i < nInputs; i++) {
        *(*data + i) = *(arr + i);
    }
    //std::cout << "Inputs Loaded" << std::endl;
}

void Network::Network::giveInputs(std::vector<double> vec) {
    for (int i = 0; i < nInputs; i++) {
        *(*data + i) = vec[i];
    }
}

double Network::Network::ReLU(double x) {
    return x > 0 ? x : x * 0.01;
}

void Network::Network::calcLayer(int layerIndex) {

    if (layerIndex <= 0) return;

    //std::cout << "Creating " << nNeurons << " Threads..." << std::endl;
#pragma omp parallel for
    for (int i = 0; i < (layerIndex >= nLayers - 1 ? nOutputs : nNeurons); i++) {
        calcNeuron(layerIndex, i);
    }
    //std::cout << "Threads Created and pushed" << std::endl;

    //std::cout << "Joining Threads..." << std::endl;
    //std::cout << "Threads Joined" << std::endl;

    //std::cout << "Clearing Completed Threads..." << std::endl;
    //std::cout << "Threads Cleared" << std::endl;
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
        sum += temp * *(*(*(weight + prevIndex) + j) + neuronIndex);
    }
    *(*(data + layerIndex) + neuronIndex) = ReLU(sum + *(*(bias + layerIndex) + neuronIndex));
}

void Network::Network::output() {
    for (int i = 1; i < nLayers; i++) {
        calcLayer(i);
    }

    //std::cout << "Done Calculations and Weighting" << std::endl;


    outputs = *(data + nLayers - 1);
}

Training::Data::Data(std::string filePath, int fminuteWindow) {
    minuteWindow = fminuteWindow;

    ifstream file;
    file.open(filePath); // Opening file

    if (!file.is_open()) {
        std::cout << "Error Opening File" << std::endl; // Error checking for file
    }

    std::string line; // Will store each line pulled from the file

    getline(file, line); // Pulling datetime line
    std::stringstream ss(line); // Making stringstream to separate out the individual datetimes

    std::string temp; // Will hold each entry in each line

    //std::cout << "Pushing Dates..." << std::endl;
    int counterNoCont = 0;
    int counterWithCont = 0;
    while (getline(ss, temp, ',')) {
        counterNoCont++;
        if (temp.length() != 16) { // Correct format will have 16 characters until the year 10000 which is someone else's problem
            continue;
        }
        parseDate(temp); // Add the parsed vector to the collection of date vectors
        counterWithCont++;
    }
    //std::cout << "Dates Pushed" << std::endl;

    while (getline(file, line)) {
        std::stringstream stream(line); // Stream to hold the parsable line
        parseLine(std::move(stream), minuteWindow); // Initiating the parsing of the line
    }

    file.close(); // To close file
}

void Training::Data::parseLine(std::stringstream line, int minuteWindow) {
    std::string temp; // To hold each entry in the line
    std::vector<double> tempVec; // To hold the parsed vector to return

    getline(line, temp, ','); // Getting the ticker

    tickerVector.push_back(temp);
    oneHotTickers.push_back(0);

    int dateIndex = 0; // To make sure the dates line up with each entry, also used for loop counting
    while (getline(line, temp, ',')) {
        if (dateIndex < minuteWindow) { // While we're parsing the info that is part of the input vector
            tempVec.push_back(std::stod(temp)); // Push the PCPM value into the vector
            tempVec.insert(tempVec.end(), parsedDates[dateIndex].begin(), parsedDates[dateIndex].end()); // Push parsed date info after it so it has date context for each value
        } else { // Once we're out of the input vector range
            inputMatrix.push_back(std::move(tempVec)); // Push our input vector to the matrix
            tempVec.clear(); // Clear the vector to start outputs
            tempVec.push_back(std::stod(temp)); // Push the number we just got on the last condition check
            break;
        }
        dateIndex++; // Increment index to track correct date
    }

    while (getline(line, temp, ',')) { // Keep parsing the line
        tempVec.push_back(std::stod(temp)); // Push them without the dates to the output vector
    }
    desiredOutputs.push_back(tempVec); // Push the parsed outputs to the output matrix
}

void Training::Data::parseDate(std::string date) {
    std::vector<double> tempVector; // The return vector
    std::string tempString; // To hold separated number
    std::stringstream dateString(date); // To parse
    getline(dateString, tempString, '-'); // Get year
    tempVector.push_back(std::stod(tempString) / 10000); // Push year
    getline(dateString, tempString, '-'); // Get month
    tempVector.push_back(std::sin((2 * M_PI * std::stod(tempString)) / 12)); // Both for cyclical nature of months
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
    parsedDates.push_back(tempVector);
}

Training::Training::Training(Network::Network* network, std::string trainingDataFilePath, int minuteWindow, int hiddenLayers, int hiddenNeurons) {
    //std::cout << "Creating Data Object..." << std::endl;
    dataObject = new Data(trainingDataFilePath, minuteWindow);
    //std::cout << "Data Object Created" << std::endl;
    //std::cout << "Creating Network Object..." << std::endl;
    network = new Network::Network(minuteWindow * 10, hiddenLayers, hiddenNeurons, 1);
    //std::cout << "Network Object Created" << std::endl;
    networkRef = network;
    //std::cout << "Trainer Initialized" << std::endl;
}

void Training::Training::train(int generations, std::string tickerToTrain) {

    trainingAll = tickerToTrain.compare("alltickers") ? false : true;

    auto tickerIterator = find(dataObject->tickerVector.begin(), dataObject->tickerVector.end(), tickerToTrain); // Getting the iterator that will give us the index of the ticker
    int tickerIndex = distance(dataObject->tickerVector.begin(), tickerIterator); // Getting the index of the ticker
    dataObject->oneHotTickers[tickerIndex] = 1; // Setting the onHotTickers value at the ticker index to 1 to indicate the ticker that we are using

    networkRef->randInit();
    int cycleCounter = 0;
    while (true) {
        if (trainingAll) {
            for (int i = 0; i < 489; i++) {
                dataObject->inputVector.clear();
                dataObject->inputVector.insert(dataObject->inputVector.begin(), dataObject->inputMatrix[i].begin(), dataObject->inputMatrix[i].end());
                networkRef->giveInputs(dataObject->inputMatrix[i]);
                for (int j = 0; j < generations; j++) {
                    networkRef->output();
                    dataObject->loss(networkRef, i, j);
                    //std::cout << "Backpropagating..." << std::endl;
                    backProp(networkRef, 0.01);
                    //std::cout << "Backpropagated" << std::endl;
                    adjustInputs(networkRef, j, i);
                    if (j == 328) {
                        for (int k = 0; k < networkRef->nOutputs; k++) {
                            std::cout << "Cycle " << cycleCounter << " Ticker Index " << i << " Output Value: " << *(*(networkRef->data + networkRef->nLayers - 1) + k) << " Desired Value: " << dataObject->desiredOutputs[i][j] << " Error: " << abs(dataObject->outputError[0]) << std::endl;
                        }
                        //std::cout << "Error For Generation " << j << " Was: " << abs(dataObject->outputError[0]) << std::endl;
                    }
                }
                dataObject->oneHotTickers[i] = 0; // Changing the ticker
                dataObject->oneHotTickers[i + 1] = 1;
            }
        } else {
            dataObject->inputVector.insert(dataObject->inputVector.begin(), dataObject->inputMatrix[tickerIndex].begin(), dataObject->inputMatrix[tickerIndex].end());
            networkRef->giveInputs(dataObject->inputMatrix[tickerIndex]); // Load the input vector into the model
            for (int i = 0; i < generations; i++) {
                networkRef->output();
                dataObject->loss(networkRef, tickerIndex, i);
                //std::cout << "Backpropagating..." << std::endl;
                backProp(networkRef, 0.01);
                //std::cout << "Backpropagated" << std::endl;
                adjustInputs(networkRef, i, tickerIndex);
                double absError = abs(dataObject->outputError[0]);
                std::cout << "Error For Generation " << i << " Was: " << absError << std::endl;
            }
        }
        cycleCounter++;
        //std::cout << "Ticker Index: " << tickerIndex << " Data Vector Size: " << dataObject->inputMatrix[tickerIndex].size() << " Inputs: " << networkRef->nInputs << " Expanded: " << dataObject->inputMatrix[tickerIndex][2] << std::endl;
    }

}

void Training::Data::loss(Network::Network* network, int tickerIndex, int generation) {
    //std::cout << "Clearing Previous Errors" << std::endl;
    outputError.clear(); // Make sure the previous errors are gone
    //std::cout << "Calculating New Errors..." << std::endl;
    for (int i = 0; i < network->nOutputs; i++) {
        //std::cout << "Entered Loop" << std::endl;
        double desiredOut = desiredOutputs[tickerIndex][generation];
        //std::cout << "Got Desired Out" << std::endl;
        double output = network->outputs[i];
        //std::cout << "Got Real Out" << std::endl;
        outputError.push_back(output - desiredOut);
        //std::cout << "Pushed Error" << std::endl;
    }
    //std::cout << "New Errors Calculated" << std::endl;
}

void Training::Training::adjustInputs(Network::Network* network, int generation, int tickerIndex) {
    //std::cout << "Input Vector Size: " << dataObject->inputVector.size() << std::endl;
    dataObject->inputVector.erase(dataObject->inputVector.begin(), dataObject->inputVector.begin() + 10);
    //std::cout << "Erased Input Vector" << std::endl;
    for (int i = 0; i < network->nOutputs; i++) {
        //std::cout << "Pushing " << network->outputs[i] << " To Input Vector" << std::endl;
        dataObject->inputVector.push_back(dataObject->desiredOutputs[tickerIndex][generation]);
        //std::cout << "Pushed Next Desired Output To Input Vector. Pushing Date From Parsed List of Length: " << dataObject->parsedDates.size() << " Using Index: " << dataObject->minuteWindow + generation + 1 << std::endl;
        dataObject->inputVector.insert(dataObject->inputVector.end(), dataObject->parsedDates[dataObject->minuteWindow + generation + 1].begin(), dataObject->parsedDates[dataObject->minuteWindow + generation + 1].end());
        //std::cout << "Date Pushed To Input Vector" << std::endl;
    }
    //std::cout << "Sending Inputs In Vector Size: " << dataObject->inputVector.size() << " To  Model" << std::endl;
    network->giveInputs(dataObject->inputVector);
}

void Training::Training::backProp(Network::Network* network, double learning_rate) {
    std::vector<double*> deltas(network->nLayers); // Store layer errors

    // Allocate space for each layer's delta values
    for (int i = 0; i < network->nLayers; i++) {
        deltas[i] = (double*)calloc((i == network->nLayers - 1) ? network->nOutputs : network->nNeurons, sizeof(double));
    }

    // Compute output layer delta
#pragma omp parallel for
    for (int i = 0; i < network->nOutputs; i++) {
        double derivative = (network->outputs[i] > 0) ? 1.0 : 0.01; // ReLU derivative
        deltas[network->nLayers - 1][i] = 2 * dataObject->outputError[i] * derivative;
    }

    // Backpropagate errors to hidden layers
    for (int l = network->nLayers - 2; l > 0; l--) { // Skip input layer
#pragma omp parallel for
        for (int j = 0; j < network->nNeurons; j++) {
            double sum = 0.0;
            for (int k = 0; k < ((l == network->nLayers - 2) ? network->nOutputs : network->nNeurons); k++) {
                sum += deltas[l + 1][k] * network->weight[l][j][k];
            }
            double derivative = (network->data[l][j] > 0) ? 1.0 : 0.01; // ReLU derivative
            deltas[l][j] = sum * derivative;
        }
    }

    // Update Weights and Biases
    for (int l = 0; l < network->nLayers - 1; l++) { // Skip output layer for weight updates
#pragma omp parallel for
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
#pragma omp parallel for
        for (int j = 0; j < ((l == network->nLayers - 1) ? network->nOutputs : network->nNeurons); j++) {
            network->bias[l][j] -= learning_rate * deltas[l][j];
        }
    }

    // Free allocated memory
    for (int i = 0; i < network->nLayers; i++) {
        free(deltas[i]);
    }
}
