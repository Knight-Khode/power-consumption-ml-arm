
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#define INPUT_WEIGHT_ROW (8)
#define INPUT_WEIGHT_COL (5)

#define L1_BIAS_ROW (8)
#define L2_BIAS_ROW (8)

#define L1_WEIGHT_ROW (8)
#define L1_WEIGHT_COL (8)

#define OUTPUT_WEIGHT_ROW (1)
#define OUTPUT_WEIGHT_COL (8)

//storing the weights and biases of each stage 

float inputWeights[INPUT_WEIGHT_ROW][INPUT_WEIGHT_COL]={
    
    -0.4807, 0.7316,    0.2220,    0.8183,   -0.6128,
    0.8112,    0.7555,   -0.0680,   -0.4665,    0.7783,
    0.2824,    0.3876,   -0.2816,   -1.3432,   -0.3031,
   -0.0648,   -0.2684,   -0.7844,   -0.6598,   -1.0571,
    0.4356,    0.6726,   -0.3542,   -0.1972,    0.5727,
   -0.8608,    0.5747,   -0.2863,    0.8638,   -0.1327,
   -0.3252,    0.1585,   -0.0562,    0.1936,   -1.0671,
    0.2445,   0.1172,   -0.0801,    1.1039,    0.5165
};


float layer1Weights[L1_WEIGHT_ROW][L1_WEIGHT_COL]= {
    
     0.3144,    0.6757,    0.0642,   -0.0106,   -0.1213,    0.9651,   -0.1812,   -0.8559,
   -0.5479,   -0.1598,    0.7486,    0.8257,    0.5434,    0.9588,    0.3598,   -0.5628,
    0.6559,   -0.3960,   -0.4398,   -0.3532,   -0.1561,   -0.3851,    0.2678,   -0.3482,
   -0.2546,    0.2577,    0.8469,   -1.0314,    0.8130,    0.8638,    0.1223,    0.3308,
    0.3957,    0.2046,   -0.6443,   -0.1329,   -0.0685,    0.5993,    0.7155,    1.8231,
    0.2899,   -0.6619,   -0.5146,   -0.6202,   -0.7351,    0.8748,   -0.7073,   -0.4242,
   -0.9493,   -0.9172,   -0.9130,    0.0405,   -0.1413,    0.2386,   -0.4672,    0.4831,
   -0.4045,    0.8938,   -0.8285,    0.6108,   -0.2574,   -0.2644,   -0.3222,   -0.0696

};


float outputWeights[OUTPUT_WEIGHT_ROW][OUTPUT_WEIGHT_COL]={ 0.7492, 2.3538, 0.4669, -0.7061, -2.9820,  2.0835, 2.5017, 2.5670 };



float layer1Biases[L1_BIAS_ROW]={
    
    -0.7423,
   -0.8563,
    0.0223,
   -0.1843,
   -0.8020,
   -0.7603,
    0.5317,
    1.7831,
};


double layer2Biases[L2_BIAS_ROW]={
    
    -0.2662,
    0.4739,
    0.0513,
    0.0622,
    0.8960,
   -0.7387,
   -0.7509,
   -0.9344,
};

 float outputBiases[1]={-4.0416};

 float L1_OUTPUT[L1_WEIGHT_ROW]; //this is the output after hidden layer 1 (8X1 Matrix)
 
 float L2_OUTPUT[L2_BIAS_ROW]; //output after hidden layer 2 (8x1 Matrix)
 
 float FINAL_OUTPUT[1]; // final output to be fed to sigmoid activation function (1X1 Matrix)
 
 
 //defining activation function: Relu and Sigmoid 
 
 //ReLu max(value, 0)
 float relu( float value){
     
     return value>0? value:0; 
 }

//sigmoid (1/(1+exp(-value)))

float sigmoid (float value){
    
    return 1/(1+exp(-value));
}


//matrix multiplication of (input* inputWeights) + layer1biases

void Layer1(float *INPUT, float *L1_OUTPUT){
    float temp; 
    for(int i=0; i<INPUT_WEIGHT_ROW; i++){
        temp=0; 
        for (int j=0; j<INPUT_WEIGHT_COL; j++){
            temp+= (INPUT[j]*inputWeights[i][j]);
        }
        
        temp= temp+ layer1Biases[i];
        L1_OUTPUT[i]= relu(temp);
    }
    
}


//matrix multiplication of layer2 (layer1Weights*L1_OUPUT + layer2Biases)

void Layer2 (float *L1_OUTPUT, float *L2_OUTPUT){
    float temp;
    for (int i=0; i<L1_WEIGHT_ROW; i++){
        temp=0; 
        for (int j=0; j<L1_WEIGHT_COL; j++){
            
            temp+= (L1_OUTPUT[j]*layer1Weights[i][j]);
        }
        
        temp= temp+ layer2Biases[i];
        L2_OUTPUT[i]= relu(temp);
    }
    
}

// output layer that will be activated by sigmoid 

void outputLayer(float *L2_OUTPUT, float *FINAL_OUTPUT){
    float temp; 
    for(int i=0; i<OUTPUT_WEIGHT_ROW; i++){
        temp=0; 
        for(int j=0; j<OUTPUT_WEIGHT_COL; j++){
            temp+= (L2_OUTPUT[j]*outputWeights[i][j]);
        }
        
        temp= temp+ outputBiases[i];
        FINAL_OUTPUT[i]= sigmoid(temp);
    }
}

//----------beginning of main() program---------------------------

int main()
{
    float INPUT[5]={0.1, 0.2, 0.3, 0.1, 0.02}; 
    
    Layer1(INPUT, L1_OUTPUT); 
    Layer2(L1_OUTPUT, L2_OUTPUT);
    outputLayer(L2_OUTPUT, FINAL_OUTPUT);
    

        
        printf("the number obtained is: %f\n", FINAL_OUTPUT[0]);
 
    return 0;
}