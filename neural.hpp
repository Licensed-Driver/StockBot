#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <thread>

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
        double* makeAMove();
        void backProp();
        double cost();
    };
}