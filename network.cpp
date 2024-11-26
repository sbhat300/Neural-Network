#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <math.h>
#include <random>

struct Node
{
    float activation;
    float z;
    std::vector<float> weights;
    std::vector<float> activationDeriv;
};

float reluDeriv(float x);
float activate(float inp);
float applyWeights(std::vector<Node>* prevNodes, std::vector<float>* weights);
void readInput(std::vector<std::vector<float>>* data, std::vector<std::vector<float>>* survived);
void testInputs(std::vector<std::vector<float>>* data, std::vector<std::vector<float>>* expected);

const int LAYERS = 5;
const int STRUCTURE[LAYERS] = {10, 20, 10, 5, 2};
const float LEAKY_FACTOR = 0.01f;
const int INP_IGNORE = 120;
const float LEARNING_RATE = 0.01f;
const int ITERATIONS = 1000;
std::string FILE_PATH = "TitanicData/";

//Setup nn
std::vector<Node> layers[LAYERS];
int main()
{
    
    for(int i = 0; i < LAYERS; i++)
    {
        layers[i].resize(STRUCTURE[i]);
    }

    time_t seed = time(NULL);

    //Read input
    std::vector<std::vector<float>> data;
    std::vector<std::vector<float>> expected;
    readInput(&data, &expected);
    std::cout << "Training data size: " << data.size() - INP_IGNORE << "\n\n";

    //Assign weights to each nodes with Kaiming initialization 
    //Also intialize corresponding weights 
    //(Node at layers[i][j].weights[k] = Node at gradients[i][j][k])
    //i = layer, j = neuron in layer, k = weight connecting neuron j to neuron k in prev layer
    std::random_device rd{};
    std::mt19937 gen{rd()}; 
    std::vector<std::vector<std::vector<float>>> gradients;
    gradients.resize(LAYERS);
    for(int i = 1; i < LAYERS; i++)
    {
        gradients[i].resize(STRUCTURE[i]);
        std::normal_distribution nDist{0.0f, std::sqrt(2.0f / STRUCTURE[i - 1])};
        for(int j = 0; j < STRUCTURE[i]; j++)
        {
            layers[i][j].weights.resize(STRUCTURE[i - 1]);
            layers[i][j].activationDeriv.resize(STRUCTURE[i - 1]);
            gradients[i][j].resize(STRUCTURE[i - 1]);
            for(int k = 0; k < STRUCTURE[i - 1]; k++) layers[i][j].weights[k] = nDist(gen);
        }
    }

    testInputs(&data, &expected);

    float loss;
    float initmse;
    for(int it = 0; it < ITERATIONS; it++)
    {
        loss = 0;
        //Reset gradients
        for(int i = 1; i < LAYERS; i++) 
            for(int j = 0; j < STRUCTURE[i]; j++) 
                for(int k = 0; k < STRUCTURE[i - 1]; k++)
                    gradients[i][j][k] = 0;
        
        for(int dat = 0; dat < data.size() - INP_IGNORE; dat++)
        {
            //Add input to network's first layer
            for(int neuron = 0; neuron < STRUCTURE[0]; neuron++) 
                layers[0][neuron].activation = data[dat][neuron];

            //Forward propagation
            for(int l = 1; l < LAYERS; l++)
            {
                for(int neuron = 0; neuron < STRUCTURE[l]; neuron++)
                {
                    layers[l][neuron].z = applyWeights(&layers[l - 1], &layers[l][neuron].weights);
                    layers[l][neuron].activation = activate(layers[l][neuron].z);
                }
            }
            
            //Calculate error and accumulate loss
            float diff[STRUCTURE[LAYERS - 1]];
            for(int output = 0; output < STRUCTURE[LAYERS - 1]; output++) 
            {
                diff[output] = layers[LAYERS - 1][output].activation - expected[dat][output];
                loss += diff[output] * diff[output];
            }
            
            //Back propagation
            for(int neuron = 0; neuron < STRUCTURE[LAYERS - 1]; neuron++)
            {
                for(int weight = 0; weight < STRUCTURE[LAYERS - 2]; weight++)
                {
                    float mult = reluDeriv(layers[LAYERS - 1][neuron].z) * 2 * diff[neuron];
                    layers[LAYERS - 1][neuron].activationDeriv[weight] = layers[LAYERS - 1][neuron].weights[weight]
                                                                        * mult;
                    gradients[LAYERS - 1][neuron][weight] += layers[LAYERS - 2][weight].activation 
                                                            * mult;
                }
            }
            for(int l = LAYERS - 2; l > 0; l--)
            {
                for(int neuron = 0; neuron < STRUCTURE[l]; neuron++)
                {
                    float weightDerivSum = 0;
                    for(int next = 0; next < STRUCTURE[l + 1]; next++) 
                            weightDerivSum += layers[l + 1][next].activationDeriv[neuron];
                    for(int weight = 0; weight < STRUCTURE[l - 1]; weight++)
                    {
                        float mult = reluDeriv(layers[l][neuron].z) * weightDerivSum;
                        gradients[l][neuron][weight] += layers[l - 1][weight].activation * mult;
                        layers[l][neuron].activationDeriv[weight] = layers[l][neuron].weights[weight] * mult;
                    }
                }
            }
        }
        loss /= data.size() - INP_IGNORE;
        if(it == 0) initmse = loss;

        //Apply gradient to weights
        for(int i = 1; i < LAYERS; i++) 
            for(int j = 0; j < STRUCTURE[i]; j++) 
                for(int k = 0; k < STRUCTURE[i - 1]; k++)
                    layers[i][j].weights[k] -= gradients[i][j][k] / (data.size() - INP_IGNORE)
                                                * LEARNING_RATE;
        
    }
    std::cout << "\nInitial MSE: " << initmse << std::endl;

    testInputs(&data, &expected);
    
    std::cout << "MSE: " << loss << std::endl;
    std::cout << "\nTime Elapsed: " << time(NULL) - seed << " seconds" << std::endl;
    return 0;
}
float reluDeriv(float x)
{
    if(x > 0) return 1;
    return LEAKY_FACTOR;
}
float activate(float inp)
{
    if(inp > 0) return inp;
    return LEAKY_FACTOR * inp;
}
float applyWeights(std::vector<Node>* prevNodes, std::vector<float>* weights)
{
    float sum = 0;
    for(int i = 0; i < prevNodes->size(); i++) sum += (*prevNodes)[i].activation * (*weights)[i];
    return sum;
}
void readInput(std::vector<std::vector<float>>* data, std::vector<std::vector<float>>* survived)
{
    std::ifstream input;
    input.open(FILE_PATH + "train.csv");
    std::string output;
    std::getline(input, output);

    //Data index meanings:
    //0 -> Pclass_1
    //1 -> Pclass_2;
    //2 -> Is male?
    //3 -> Age
    //4 -> Number Siblings
    //5 -> Number parents
    //6 -> Fare
    //7 -> is S
    //8 -> is C
    //9 -> ones

    float maxSib = 0;
    float maxAge = 0;
    float maxParch = 0;
    std::vector<float> line;
    line.resize(10);
    char temp;

    //read input
    while(input.good())
    {
        if(input.peek() == -1) break;
        std::getline(input, output, ',');
        std::string tempId = output;

        bool survive;
        input >> survive;
        input >> temp;

        float num = 0;
        input >> num;
        if(num == 1) 
        {
            line[0] = 1;
            line[1] = 0;
        }
        else if(num == 2)
        {
            line[1] = 1;
            line[0] = 0;
        }
        else
        {
            line[0] = 0;
            line[1] = 0;
        }
        input >> temp;

        char prev = temp;

        while(!(prev == '"' && temp == ','))
        {
            prev = temp;
            input >> temp;
        }

        std::getline(input, output, ',');
        if(output == "male") line[2] = 1;
        else line[2] = 0;

        if(input.peek() == ',') 
        {
            std::getline(input, output);
            continue;
        }
        input >> num;
        line[3] = num;
        maxAge = std::max(maxAge, num);
        input >> temp;

        input >> num;
        line[4] = num;
        maxSib = std::max(maxSib, num);
        input >> temp;

        input >> num;
        line[5] = num;
        maxParch = std::max(maxParch, num);
        input >> temp;

        std::getline(input, output, ',');

        input >> num;
        line[6] = num <= 0 ? 0 : log(num);
        input >> temp;

        std::getline(input, output, ',');

        std::getline(input, output);
        if(output[0] == 'S')
        {
            line[7] = 1;
            line[8] = 0;
        }
        else if(output[0] == 'C')
        {
            line[8] = 1;
            line[7] = 0;
        }
        else if(output[0] == 'Q')
        {
            line[7] = 0;
            line[8] = 0;
        }
        else continue;

        line[9] = 1;

        data->push_back(line);
        if(survive) survived->push_back({0, 1});
        else survived->push_back({1, 0});
        // std::vector<float> outVec = {(float)survive};
        // survived->push_back(outVec);
    }
    for(int i = 0; i < data->size(); i++)
    {
        (*data)[i][3] /= maxAge;
        (*data)[i][4] /= maxSib;
        (*data)[i][5] /= maxParch;
    }
}
void testInputs(std::vector<std::vector<float>>* data, std::vector<std::vector<float>>* expected)
{
    //Test Outputs:
    int right = 0;
    int counter = 0;
    for(int i = data->size() - INP_IGNORE; i < data->size(); i++)
    {
        counter++;
        for(int neuron = 0; neuron < STRUCTURE[0]; neuron++) 
                layers[0][neuron].activation = (*data)[i][neuron];
        for(int l = 1; l < LAYERS; l++)
        {
            for(int neuron = 0; neuron < STRUCTURE[l]; neuron++)
            {
                layers[l][neuron].z = applyWeights(&layers[l - 1], &layers[l][neuron].weights);
                layers[l][neuron].activation = activate(layers[l][neuron].z);
            }
        }

        if(STRUCTURE[LAYERS - 1] == 1) 
        {
            float output = layers[LAYERS - 1][0].activation >= 0.5f ? 1.0f : 0.0f;
            if(output == (*expected)[i][0]) right++;
        }
        else
        {
            float max = layers[LAYERS - 1][0].activation;
            int maxNeuron = 0;
            float maxReal = (*expected)[i][0];
            int maxRealInd = 0;
            for(int neuron = 1; neuron < STRUCTURE[LAYERS - 1]; neuron++)
            {
                if((*expected)[i][neuron] > maxReal)
                {
                    maxReal = (*expected)[i][neuron];
                    maxRealInd = neuron;
                }
                if(layers[LAYERS - 1][neuron].activation > max) 
                {
                    max = layers[LAYERS - 1][neuron].activation;
                    maxNeuron = neuron;
                }
            }
            if(maxNeuron == maxRealInd) right++;
        }
    }
    std::cout << "\nProp Correct (test of " << INP_IGNORE << " people): " << 
                (float)right / counter << std::endl;
}