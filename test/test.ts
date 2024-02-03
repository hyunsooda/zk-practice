import * as path from 'path'
import * as fs from 'fs'
import { assert, expect } from 'chai';
import { wasm as wasm_tester } from 'circom_tester';
import { F1Field, Scalar } from 'ffjavascript';
import { execSync } from 'child_process';
exports.p = Scalar.fromString("21888242871839275222246405745257275088548364400416034343698204186575808495617");
const Fr = new F1Field(exports.p);

const parseWeight = function(filePath: string) : Number[][] {
  let weights = [];
  for (let a of fs.readFileSync(filePath, 'utf8').split('\n')) {
    weights.push(a.split(',').map(Number))
  }
  weights.pop();
  return weights;
}

const getModelHash = async function(fc1, fc2, salt) {
  const circuit = await wasm_tester(path.join(__dirname,"../circuit", "model_hash_main.circom"));
  await circuit.loadConstraints();
  let circuitInputs = {
    "fc1In": fc1,
    "fc2In": fc2,
    "modelHashSalt": salt,
  };
  const witness = await circuit.calculateWitness(circuitInputs, true);
  return witness[1];
}

const getCircuitInputs = async function() {
  // run the trained model and makes test output to be used for this unit test
  execSync('just modeltest');
  const [targetIdx, testOut] = fs.readFileSync("test-result.txt", 'utf8').split(':');
  const fc1 = parseWeight("weight/fc1.txt");
  const fc2 = parseWeight("weight/fc2.txt");
  const testImages = parseWeight("input/test.csv");
  const targetImage = testImages[Number(targetIdx) + 1]; // +1 is for ignoring testdata header
  const salt = 123;

  const modelHash = await getModelHash(fc1, fc2, salt);
  return {
    "fc1In": fc1,
    "fc2In": fc2,
    "image": targetImage,
    "modelHashSalt": salt,
    "expectedModelHash": modelHash,
    "expectedOut": testOut,
  };
}

function unstringifyBigInts(o) {
  if ((typeof (o) == "string") && (/^[0-9]+$/.test(o))) {
      return BigInt(o);
  } else if ((typeof (o) == "string") && (/^0x[0-9a-fA-F]+$/.test(o))) {
      return BigInt(o);
  } else if (Array.isArray(o)) {
      return o.map(unstringifyBigInts);
  } else if (typeof o == "object") {
      if (o === null) return null;
      const res = {};
      const keys = Object.keys(o);
      keys.forEach((k) => {
          res[k] = unstringifyBigInts(o[k]);
      });
      return res;
  } else {
      return o;
  }
}

describe("Correctness of model inference", function () {
  this.timeout(100000);

    it("Correctness of model inference", async()=>{
      const circuit = await wasm_tester(path.join(__dirname,"../circuit", "main.circom"));
      await circuit.loadConstraints();

      const circuitInputs = await getCircuitInputs();
      const witness = await circuit.calculateWitness(circuitInputs, true);

      assert(Fr.eq(Fr.e(witness[0]), Fr.e(1)));
      // write a file that contains circuit inputs
      fs.writeFileSync("circuit/artifacts/input.json", JSON.stringify(circuitInputs));
    })

    it("Maze problem solving", async()=>{
      const circuit = await wasm_tester(path.join(__dirname,"../circuit", "maze.circom"));
      await circuit.loadConstraints();

      const witness = await circuit.calculateWitness({
        "answer": [2,2,1,1,3,1,1,3, -1,-1,-1,-1,-1,-1,-1],
        // identical answer
        // "answer": [2,2,1,1,3,1,1,3,0,1, -1,-1,-1,-1,-1],
      }, true);
      assert(Fr.eq(Fr.e(witness[0]), Fr.e(1)));
      assert(Fr.eq(Fr.e(witness[1]), Fr.e(4)));
    })
})
