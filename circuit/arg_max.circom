pragma circom 2.1.7;

include "../node_modules/circomlib/circuits/switcher.circom";


// description: ArgMax in ML framework
// input: in:list of values
// output: an index of maximum value

template ArgMax(n) {
    signal input in[n];
    signal output out;

    component gts[n];        // store comparators
    component switchers[n+1];  // switcher for comparing maxs
    component aswitchers[n+1]; // switcher for arg max

    signal maxs[n+1];
    signal amaxs[n+1];

    maxs[0] <== in[0];
    amaxs[0] <== 0;
    for(var i = 0; i < n; i++) {
        gts[i] = GreaterThan(252); // changed to 252 (maximum) for better compatibility
        switchers[i+1] = Switcher();
        aswitchers[i+1] = Switcher();

        gts[i].in[1] <== maxs[i];
        gts[i].in[0] <== in[i];

        switchers[i+1].sel <== gts[i].out;
        switchers[i+1].L <== maxs[i];
        switchers[i+1].R <== in[i];

        aswitchers[i+1].sel <== gts[i].out;
        aswitchers[i+1].L <== amaxs[i];
        aswitchers[i+1].R <== i;
        amaxs[i+1] <== aswitchers[i+1].outL;
        maxs[i+1] <== switchers[i+1].outL;
    }

    out <== amaxs[n];
}
