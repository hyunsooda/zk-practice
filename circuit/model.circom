pragma circom 2.1.7;

include "../node_modules/circomlib/circuits/mimcsponge.circom";
include "../node_modules/circomlib/circuits/comparators.circom";
include "lib.circom";
include "arg_max.circom";
include "relu.circom";
include "model_hash.circom";

// TODO: apply freivalds' algorithm (efficient approach)
// description: takes model parameters, hash, and input image
//              checks first the model's checksum and then verify the correctness of the model's output for the given image
// input: fc1In:matrix, fc2In:matrix, image:vector, modelHashSalt:salt, expectedModelHash:hash
// output: `outputSize` of list
template VeriyModel(weightRow, weightCol, outputSize) {
    var NEGATIVE_COMPLEMENT = 10000000;
    signal input fc1In[weightRow][weightCol];
    signal input fc2In[outputSize][weightRow];
    signal input image[weightCol];
    signal input modelHashSalt;
    signal input expectedModelHash;

    // 0. model integrity check
    component getModelHash = GetModelHash(weightRow, weightCol, outputSize);
    fc1In             ==> getModelHash.fc1In;
    fc2In             ==> getModelHash.fc2In;
    modelHashSalt     ==> getModelHash.modelHashSalt;
    expectedModelHash === getModelHash.out;

    // 1. calculate fc1 layer
    component mmv1 = MatMulVec(weightRow, weightCol);
    for (var i=0; i<weightRow; i++) {
        for (var j=0; j<weightCol; j++) {
            fc1In[i][j] ==> mmv1.A[i][j];
        }
    }
    for (var i=0; i<weightCol; i++) {
        image[i] ==> mmv1.x[i];
    }
    signal fc1Out[weightRow] <== mmv1.out;

    // 2. calculate relu layer
    component relu[weightRow];
    for (var i=0; i<weightRow; i++) {
        relu[i] = ReLU();
        fc1Out[i] ==> relu[i].in;
    }

    // 3. calculate fc2 layer
    component mmv2 = MatMulVec(outputSize, weightRow);
    for (var i=0; i<outputSize; i++) {
        for (var j=0; j<weightRow; j++) {
            fc2In[i][j] + NEGATIVE_COMPLEMENT ==> mmv2.A[i][j];
        }
    }
    for (var i=0; i<weightRow; i++) {
        relu[i].out ==> mmv2.x[i];
    }
    component argMax = ArgMax(outputSize);
    mmv2.out ==> argMax.in;
    signal output out <== argMax.out;
}
