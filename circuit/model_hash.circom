pragma circom 2.1.7;

include "../node_modules/circomlib/circuits/mimcsponge.circom";

// description: verify a model's checksum
//              pushing all the model's weight is very costly
//              instead, we summing up the model's weight and the sum value(scalar) is hashed with the salt
// input: fc1In:matrix, fc2In:matrix, modelHashSalt:salt, expectedModelHash:hash
// output: hash
template GetModelHash(weightRow, weightCol, outputSize) {
    signal input fc1In[weightRow][weightCol];
    signal input fc2In[outputSize][weightRow];
    signal input modelHashSalt;
    signal fc1Sum[weightRow * weightCol];

    for (var i=0; i<weightRow; i++) {
        fc1Sum[i * weightCol] <== fc1In[i][0];
        for (var j=1; j<weightCol; j++) {
            fc1Sum[i * weightCol + j] <== fc1Sum[i * weightCol + j - 1] + fc1In[i][j];
        }
    }
    var idx = weightRow * weightCol;
    signal fc2Sum[outputSize * weightRow];
    for (var i=0; i<outputSize; i++) {
        fc2Sum[i * weightRow] <== fc2In[i][0];
        for (var j=1; j<weightRow; j++) {
            fc2Sum[i * weightRow + j] <== fc2Sum[i * weightRow + j - 1] + fc2In[i][j];
        }
    }
    signal modelSum <== fc1Sum[weightRow * weightCol - 1] + fc2Sum[outputSize * weightRow - 1];
    component mimc = MiMCSponge(1, 220, 1);
    modelSum          ==> mimc.ins[0];
    modelHashSalt     ==> mimc.k;
    signal output out <== mimc.outs[0];
}
