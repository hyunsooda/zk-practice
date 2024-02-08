pragma circom 2.1.7;

include "../node_modules/circomlib/circuits/comparators.circom";
include "lib.circom";

// description: verify the given maze answer
// input: col:column size, nVars:# of varaibles
// output: maze index
template solve(mazeSize, col, nVars) {
    var HOLD  = -1;
    var LEFT  = 0;
    var RIGHT = 1;
    var DOWN  = 2;
    var UP    = 3;

    signal input goal;
    signal input answer[nVars];

    signal temp[nVars];
    signal leftright[nVars];
    signal down[nVars];
    signal downTemp[nVars];
    signal up[nVars];
    signal sum[nVars + 1];

    component gts2[nVars];
    component gts3[nVars];
    component hold[nVars];

    sum[0] <== 0;
    for (var i=0; i<nVars; i++) {
        assert(answer[i] == HOLD || answer[i] == LEFT || answer[i] == RIGHT || answer[i] == UP || answer[i] == DOWN);
        temp[i] <== 2 * answer[i] - 1;
        gts2[i] = GreaterThan(252);
        gts3[i] = GreaterThan(252);

        answer[i] ==> gts2[i].in[0];
        1         ==> gts2[i].in[1];

        answer[i] ==> gts3[i].in[0];
        2         ==> gts3[i].in[1];

        hold[i] = IsPositive();
        answer[i] ==> hold[i].in;

        // find a function for the given input & output pair
        // f(0) = -1  => L
        // f(1) =  1  => R
        // f(2) = -n  => U
        // f(3) =  n  => D
        leftright[i] <== hold[i].out * temp[i];
        downTemp[i]  <== hold[i].out * gts2[i].out;
        down[i]      <== downTemp[i] * col - gts2[i].out * temp[i];
        up[i]        <== hold[i].out * gts3[i].out * col * -2;

        sum[i+1] <== sum[i] + leftright[i] + down[i] + up[i];
    }
    component eq = IsEqual();
    goal       ==> eq.in[0];
    sum[nVars] ==> eq.in[1];
    signal output out <== eq.out;
}

component main {public [goal]} = solve(20, 5, 15);
