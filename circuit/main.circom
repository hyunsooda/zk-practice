pragma circom 2.1.7;

include "model.circom";

template verify(weightRow, weightCol, outputSize) {
    signal input fc1In[weightRow][weightCol];
    signal input fc2In[outputSize][weightRow];
    signal input image[weightCol];
    signal input modelHashSalt;
    signal input expectedModelHash;
    signal input expectedOut;

    component verifyModel = VeriyModel(200, 784, 10); // input size = 784, hidden size = 200, output size = 10
    fc1In             ==> verifyModel.fc1In;
    fc2In             ==> verifyModel.fc2In;
    image             ==> verifyModel.image;
    modelHashSalt     ==> verifyModel.modelHashSalt;
    expectedModelHash ==> verifyModel.expectedModelHash;
    expectedOut       === verifyModel.out;

    signal output out <== verifyModel.out;
}

component main = verify(200, 784, 10); // input size = 784, hidden size = 200, output size = 10
