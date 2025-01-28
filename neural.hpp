#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <thread>
#include <fstream>

namespace Network {
    class Network {
    private:
        double** data;
        double*** weight;
        double** bias;
        double* inputs;

        int nInputs;
        int nLayers;
        int nNeurons;
        int nOutputs;

        std::vector<std::thread> threads;
    public:
        Network(size_t inputs, size_t layers, size_t neurons, size_t outputs);
        void randInit();
        void giveInputs(double* arr);
        double ReLU(double x);
        void calcLayer(int layerIndex);
        void calcNeuron(int layerIndex, int neuronIndex);
        void calcOutputs(int layerIndex, int neuronIndex);
        double* output();
    };
}
namespace Training {
    
    class Training {
        private:
        int inputs;
        int layers;
        int outputs;
        public:
        Training(size_t inputs, size_t layers, size_t neurons, size_t outputs);
        void train(Network::Network* network, double* initialData);
        void adjustInputs(Network::Network *network);
        void backProp(Network::Network* network);
    };

    class Data {
        private:
        int* oneHotTickers;
        public:
        Data(string filePath);
        void loss(Network::Network* network, size_t layers, size_t outputs);
    };
}