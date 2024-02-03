pragma circom 2.1.7;

include "../node_modules/circomlib/circuits/comparators.circom";
include "../node_modules/circomlib/circuits/sign.circom";

// Range test if an input is in the given range. (lowerbound <= a <= upperbound)
template Range() {
    signal input a, lowerbound, upperbound;
    signal output out;
    // lowerBound < a < upperBound
    component lt1 = LessEqThan(250);
    lowerbound ==> lt1.in[0];
    a          ==> lt1.in[1];

    component lt2 = LessEqThan(250);
    a          ==> lt2.in[0];
    upperbound ==> lt2.in[1];

    out <== lt1.out * lt2.out;
}

// max(a[0], a[1])
template Max2() {
    signal input a[2];

    // a[0] < a[1]
    component lt1 = LessThan(250);
    a[0] ==> lt1.in[0];
    a[1] ==> lt1.in[1];

    // a[1] < a[0]
    component lt2 = LessThan(250);
    a[0] ==> lt2.in[1];
    a[1] ==> lt2.in[0];

    signal out1 <== lt1.out * a[1];
    signal out2 <== lt2.out * a[0];
    signal output out <== out1 + out2;
}

// max(a[0], a[1], a[2])
template Max3() {
    signal input a[3];

    component max1 = Max2();
    a[0] ==> max1.a[0];
    a[1] ==> max1.a[1];

    component max2 = Max2();
    a[2]     ==> max2.a[0];
    max1.out ==> max2.a[1];

    signal output out <== max2.out;
}

// max(a[0], a[1], a[2], a[3])
template Max4() {
    signal input a[4];

    component max1 = Max3();
    a[0] ==> max1.a[0];
    a[1] ==> max1.a[1];
    a[2] ==> max1.a[2];

    component max2 = Max2();
    a[3]     ==> max2.a[0];
    max1.out ==> max2.a[1];

    signal output out <== max2.out;
}

// min(a[0], a[1])
template Min2() {
    signal input a[2];

    // a[0] < a[1]
    component lt1 = LessThan(250);
    a[0] ==> lt1.in[0];
    a[1] ==> lt1.in[1];

    // a[1] < a[0]
    component lt2 = LessThan(250);
    a[0] ==> lt2.in[1];
    a[1] ==> lt2.in[0];

    signal out1 <== lt1.out * a[0];
    signal out2 <== lt2.out * a[1];
    signal output out <== out1 + out2;
}

// main(a[0], a[1], a[2])
template Min3() {
    signal input a[3];

    component min1 = Min2();
    a[0] ==> min1.a[0];
    a[1] ==> min1.a[1];

    component min2 = Min2();
    a[2]     ==> min2.a[0];
    min1.out ==> min2.a[1];

    signal output out <== min2.out;
}

// main(a[0], a[1], a[2], a[3])
template Min4() {
    signal input a[4];

    component min1 = Min3();
    a[0] ==> min1.a[0];
    a[1] ==> min1.a[1];
    a[2] ==> min1.a[1];

    component min2 = Min2();
    a[3]     ==> min2.a[0];
    min1.out ==> min2.a[1];

    signal output out <== min2.out;
}

// description: returns 1 if positive, otherwise returns 0
// input: number
// output: 0 or 1
template IsPositive() {
    signal input in;
    signal output out;

    component num2Bits = Num2Bits(254);
    num2Bits.in <== in;
    component sign = Sign();
    
    for (var i = 0; i < 254; i++) {
        sign.in[i] <== num2Bits.out[i];
    }

    out <== 1 - sign.sign;
}

// test a < b
template LargerThan() {
    signal input in[2];
    signal output out;

    component lt = LessEqThan(250);
    in[0] ==> lt.in[0];
    in[1] ==> lt.in[1];

    out <== 1 - lt.out;
}

// test a <= b
template LargerEqThan() {
    signal input in[2];
    signal output out;

    component lg = LargerThan();
    in[0] + 1 ==> lg.in[0];
    in[1]     ==> lg.in[1];
}

template VecAdd(n) {
    signal input a[n], b[n];
    signal output c[n];

    for (var i=0; i<n; i++) {
        c[i] <== a[i] + b[i];
    }
}

template VecMul(n) {
    signal input a[n], b[n];
    signal output c[n];

    for (var i=0; i<n; i++) {
        c[i] <== a[i] + b[i];
    }
}

template MatAdd(a_n, a_m, b_n, b_m) {
    a_m === b_n;

    signal input a[a_n][a_m], b[b_n][b_m];
    signal output c[a_n][a_m];

    for (var i=0; i<a_m; i++) {
        for (var j=0; j<b_n; j++) {
            c[i][j] <== a[i][j] + b[i][j];
        }
    }
}

// description: matrix and vector multiplication
// input: A:matrix, x:vector
// output: out:matrix
template MatMulVec(m, n) {
    signal input A[m][n];
    signal input x[n];
    signal output out[m];
    
    signal s[m][n + 1];          // Store intermediate sums
    for (var i = 0; i < m; i++) {
        s[i][0] <== 0;
        for (var j = 1; j <= n; j++) {
            s[i][j] <== s[i][j-1] + A[i][j-1] * x[j-1];
        }
        out[i] <== s[i][n];
    }
}

template MatMul(a_n, a_m, b_n, b_m) {
    a_m === b_n;

    signal input A[a_n][a_m], B[b_n][b_m];
    signal output C[a_n][b_m];

    signal lineDot[a_n][b_m][b_n];

    for (var i=0; i<a_n; i++) {
        for (var j=0; j<b_m; j++) {
            lineDot[i][j][0] <== A[i][0] * B[0][j];
            for (var k=1; k<b_n; k++) {
                lineDot[i][j][k] <== A[i][k] * B[k][j] + lineDot[i][j][k - 1];
            }
            C[i][j] <== lineDot[i][j][b_n - 1];
        }
    }
}
