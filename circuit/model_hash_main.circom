pragma circom 2.1.7;

include "model_hash.circom";

component main = GetModelHash(200, 784, 10); // input size = 784, hidden size = 200, output size = 10
