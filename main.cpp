#include <iostream>
#include <vector>
#include <time.h>
#include <cmath>
#include <fstream>
#include <string>

#define EPS 1e-3
#define RATE 1e-3

using namespace std;

float rand_float(){
    return (float) rand() / (float) RAND_MAX;
}

float matrix_mult(vector<float> x, vector<vector<float>> w){
    float r;
    for (int i = 0; i < (int)w[0].size(); i++){
        r += x[i] * w[i][0];
    }
    return r;
}

float sigmoid(float x){
    return 1.0/(1 + exp(-x));
}

class Layer{
    public:
        vector<vector<float>> w;
        vector<float> output;
        vector<float> b;
        vector<float> inputs;

        Layer(int x, int n){
            vector<float> tmp;
            for (int i = 0; i < x; i++){
                tmp.clear();
                for (int j = 0; j < n; j++){ 
                    srand(time(0));
                    tmp.push_back(rand_float());
                }
                w.push_back(tmp);
                b.push_back(rand_float());
            }
        }

        void forward(vector<float> inputs, float c, int n){
            inputs = inputs;
            output.clear();
            vector<vector<float>> tmp;
            for (int i = 0; i < (int)w[0].size(); i++){
                vector<float> tmp2(1);
                for (int j = 0; j < (int)inputs.size(); j++){
                    if (i == n) tmp2.push_back(w[j][i] + c);
                    else tmp2.push_back(w[j][i]);
                    tmp.push_back(tmp2);
                    tmp2.clear();
                }
                if (n == -1) output.push_back(sigmoid(matrix_mult(inputs, tmp)+b[i]+c));
                else output.push_back(sigmoid(matrix_mult(inputs, tmp)+b[i]));
                tmp.clear();
            }
        }

};

float cost(vector<vector<float>> inputs, vector<float> eval, vector<Layer> m, float c, int l, int n){
    float d = 0;
    float r = 0;
    for (int i = 0; i < (int)inputs.size(); i++){
        if( l == 0 ) m[0].forward(inputs[i], c, n);
        else m[0].forward(inputs[i], 0, -2);
        for (int j = 1; j < (int)m.size(); j++){
            if (l == j) m[j].forward(m[j-1].output, c, n);
            else m[j].forward(m[j-1].output, 0, -2);
        }
        d = eval[i] - m[m.size()-1].output[0];
        r+=d*d;
    }
    return r/(float)eval.size();
}

float predict(vector<float> inputs, vector<Layer> m){
    m[0].forward(inputs, 0, -2);
    for (int i = 1; i < (int)m.size(); i++){
        m[i].forward(m[i-1].output, 0, -2);
    }
    return m[m.size()-1].output[0];
}

void train(vector<Layer> m, vector<vector<float>> inputs, int epochs, vector<float> eval){
    float dw;
    float db;
    
    cout << "COST: " << cost(inputs, eval, m, 0, -1, -2) << '\n';
    for (int n = 0; n < epochs; n++){
        for (int i = 0; i < (int)m.size(); i++){
            for (int j = 0; j < (int)m[i].w[0].size(); j++){
                dw = (cost(inputs, eval, m, EPS, i, j) - cost(inputs, eval, m, 0, -1, -2))/EPS;
                db = (cost(inputs, eval, m, EPS, i, -1) - cost(inputs, eval, m, 0, -1, -2))/EPS;
                for (int k = 0; k < (int)m[i].w.size(); k++){
                    m[i].w[k][j] -= RATE*dw;
                }
                m[i].b[j] -= RATE*db;
            }
        }

    }
    cout << "COST: " << cost(inputs, eval, m, 0, -1, -2) << '\n';
}

void save(vector<Layer> m, string name){
    ofstream file(name);
    for (int i = 0; i < (int)m.size(); i++){
        for (int j = 0; j < (int)m[i].w.size(); j++){
            file << "Layers:\n";
            for (int k = 0; k < (int)m[i].w[j].size(); k++){
                file << m[i].w[j][k] << ", ";
            }
            file << "\n";
            file << "Biases:\n";
            file << m[i].b[j] << "\n";
        }
        file << "\n----------------------------\n";
    }
    file.close();
}

int main(void){
    srand(time(0));
    Layer l1(2, 4),l2(4, 1);
    vector<Layer> m = {l1, l2};
    vector<vector<float>> inputs = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
    vector<float> eval = {0, 1, 1, 1};
    save(m, "untrained_model");
    train(m, inputs, 1000, eval);
    save(m, "trained_model");

    return 0;
}
