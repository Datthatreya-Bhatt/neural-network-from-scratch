class FeedForward {
  constructor() {
    this.networkLayer = [];
    this.networkWeight = [];
    this.bias = [];
    this.activationFunction = [];
  }

  addLayers(arr) {
    if (arr.length < 3) {
      throw new Error("Add at least 3 layers");
    }
    const length = arr.length;
    for (let i = 0; i < length; i++) {
      //Adding hiddenLayer where ith layer will have ith node.
      this.networkLayer[i] = new Array(arr[i]).fill(Math.random());

      //Adding weight for all the layer
      if (i + 1 < length) {
        this.networkWeight.push(this.addWeight(arr[i], arr[i + 1]));
      }
    }
  }

  addBias(arr) {
    if (arr.length !== this.networkLayer.length - 2) {
      throw new Error(
        "Input and output layer wont be having bias, add bias to all the other layer."
      );
    }
    this.bias = arr;
  }

  addWeight(fromNodeLength, toNodeLength) {
    let output = [];
    for (let i = 0; i < fromNodeLength; i++) {
      let temp = [];
      for (let j = 0; j < toNodeLength; j++) {
        temp[j] = Math.random();
      }
      output[i] = temp;
    }
    return output;
  }

  multiply() {

  }

  //Use to run the NN
  run(arr) {
    if (!this.bias.length) {
      throw new Error("Add bias before running network");
    }
    // if(!this.activationFunction.length){
    //     throw new Error('Add Activation functions before running the network')
    // }
    if (arr.length !== this.networkLayer[0].length) {
      throw new Error("Given array size should be equal to input node size");
    }

    this.networkLayer[0] = arr;
  }

  //Use to back propagate
  learn() {}

  printConstructor(item) {
    console.log(`${item}`, this[item]);
  }

  help() {
    console.log("Add input node using addInputNode method");
    console.log("Add output node using addOutPutNodes method");
    console.log(
      "Add hidden nodes using addHiddenLayers method, which takes an array as input where each element in is nodes in ith layer"
    );
  }
}

let test = new FeedForward();
test.addLayers([3, 2, 3]);
test.addBias([0.2]);
test.run([1, 2, 3]);
test.printConstructor('networkLayer')
test.printConstructor("networkWeight");
