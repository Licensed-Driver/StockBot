#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <thread>
#include <fstream>
#include <sstream>
#include <bits/stdc++.h>
#include <string>

namespace Network {
    class Network {
    public:
        double** data;
        double*** weight;
        double** bias;
        double* inputs;
        double* outputs; // Array for the output layer to access easier

        int nInputs;
        int nLayers;
        int nNeurons;
        int nOutputs;

        std::vector<std::thread> threads;
        Network(size_t inputs, size_t layers, size_t neurons, size_t outputs);
        void randInit();
        void giveInputs(double* arr);
        void giveInputs(std::vector<double> vec);
        double ReLU(double x);
        void calcLayer(int layerIndex);
        void calcNeuron(int layerIndex, int neuronIndex);
        void calcOutputs(int layerIndex, int neuronIndex);
        void output();
    };
}
namespace Training {

    class Data {
    public:
        Data(std::string filePath, int fminuteWindow);

        std::vector<std::vector<double>> inputMatrix; // The input vectors for each possible ticker ready to use
        std::vector<double> inputVector; // The final vector that will be passed as inputs to the model each generation
        std::vector<std::vector<double>> desiredOutputs; // A matrix of the vectors that hold all price changes after the initial window for each ticker
        std::vector<std::vector<double>> parsedDates; // A vector that holds the parsed vector version of each date
        std::vector<double> outputError; // A vector to hold the error for each output node
        std::vector<std::string> tickerVector; // A vector that holds all ticker names
        std::vector<int> oneHotTickers; // A vector to hold the one-hot ticker representation
        int minuteWindow; // The amount of minutes that you want to train on

        void loss(Network::Network* network, int tickerIndex, int generation);
        void parseLine(std::stringstream line, int minuteWindow);
        void parseDate(std::string date);
    };

    class Training {
    private:
        int inputs;
        int layers;
        int outputs;
        Network::Network* networkRef;
        Data* dataObject;
    public:
        bool trainingAll = false;
        Training(Network::Network* network, std::string trainingDataFilePath, int minuteWindow, int hiddenLayers, int hiddenNeurons);
        void train(int generations = 1, std::string tickerToTrain = "alltickers");
        void adjustInputs(Network::Network* network, int generation, int tickerIndex);
        void backProp(Network::Network* network, double learningRate);
    };
}