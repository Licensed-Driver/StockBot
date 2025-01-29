#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <thread>
#include <fstream>
#include <sstream>
#include <bits/stdc++.h>

namespace Network {
    class Network {
    public:
        double** data;
        double*** weight;
        double** bias;
        double* inputs;

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
        double* output();
    };
}
namespace Training {

    class Data {
    public:
        Data(std::string filePath);
        std::vector<std::string> tickerVector;
        std::vector<int> oneHotTickers;
        std::vector<std::vector<double>> dataMatrix;
        std::vector<std::vector<double>> dateVector;
        double loss(Network::Network* network, int tickerIndex, int generation);
        void parseLine(std::vector <double>* dataVector, std::vector<std::vector<double>> date, std::stringstream line);
        std::vector<double> parseDate(std::string date);
    };

    class Training {
    private:
        int inputs;
        int layers;
        int outputs;
        Network::Network* networkRef;
        Data* dataObject;
    public:
        Training(Network::Network* network, std::string trainingDataFilePath, int minuteWindow, int hiddenLayers, int hiddenNeurons);
        void train(std::string tickerToTrain, int generations = 1);
        void adjustInputs(Network::Network* network, std::vector<double>* slidingWindow, int generation, int tickerIndex);
        void backProp(Network::Network* network, double learningRate, double error);
    };
}