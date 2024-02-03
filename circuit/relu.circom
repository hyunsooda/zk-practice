pragma circom 2.1.7;


// description: relu in ML framework
// input: element
// output: element
template ReLU() {
    signal input in;
    signal output out;

    component isPositive = IsPositive();

    isPositive.in <== in;
    
    out <== in * isPositive.out;
}