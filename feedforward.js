class FeedForward {
  constructor() {
    this.networkLayer = [];
    this.networkWeight = [];
    this.bias = [];
    this.activationFunction = [];
    this.learningRate = 0.1; // Learning rate for gradient descent
  }

  // Method to calculate the activation function
  applyActivationFunction(value, activationFunction) {
    switch (activationFunction) {
      case "sigmoid":
        return 1 / (1 + Math.exp(-value));
      case "relu":
        return Math.max(0, value);
      case "tanh":
        return Math.tanh(value);
      default:
        throw new Error("Unsupported activation function");
    }
  }

  // Method to calculate the derivative of the activation function
  activationFunctionDerivative(value, activationFunction) {
    switch (activationFunction) {
      case "sigmoid":
        return value * (1 - value); // sigmoid' = sigmoid * (1 - sigmoid)
      case "relu":
        return value > 0 ? 1 : 0; // ReLU' = 1 if x > 0, else 0
      case "tanh":
        return 1 - value * value; // tanh' = 1 - tanh^2
      default:
        throw new Error("Unsupported activation function");
    }
  }

  // Method to perform a forward pass (multiply input with weights and add biases)
  multiply(previousLayer, weights, biases, activationFunction) {
    return previousLayer.map((_, i) => {
      let sum = 0;
      for (let j = 0; j < previousLayer.length; j++) {
        sum += previousLayer[j] * weights[j][i];
      }
      sum += biases[i];
      return this.applyActivationFunction(sum, activationFunction);
    });
  }

  // Method to run the neural network forward pass
  run(input) {
    if (this.bias.length === 0) {
      throw new Error("Bias values are missing.");
    }

    if (input.length !== this.networkLayer[0].length) {
      throw new Error("Input size does not match the input layer size.");
    }

    // Initialize the first layer with the input
    this.networkLayer[0] = input;

    // Perform forward pass through all layers
    for (let i = 0; i < this.networkLayer.length - 1; i++) {
      const currentLayer = this.networkLayer[i];
      const weights = this.networkWeight[i];
      const biases = this.bias[i] || [];
      const activationFunction = this.activationFunction[i] || "sigmoid"; //Use sigmoid as Default if not set
      this.networkLayer[i + 1] = this.multiply(
        currentLayer,
        weights,
        biases,
        activationFunction
      );
    }
  }

  // Method to compute the error at the output layer
  computeOutputError(target, output) {
    return output.map((o, i) => o - target[i]); // Simple Mean Squared Error derivative
  }

  // Method to update weights and biases using gradient descent
  updateWeightsAndBiases(layerIndex, layerInput, error) {
    const weights = this.networkWeight[layerIndex];
    const biases = this.bias[layerIndex];
    const activationFunction = this.activationFunction[layerIndex];

    // Calculate the gradients for weights and biases
    for (let i = 0; i < weights.length; i++) {
      for (let j = 0; j < weights[i].length; j++) {
        // Gradient for weight is the derivative of the loss w.r.t the weight
        // (error * derivative of activation * input to the neuron)
        const gradient =
          error[j] *
          this.activationFunctionDerivative(
            this.networkLayer[layerIndex + 1][j],
            activationFunction
          ) *
          layerInput[i];
        weights[i][j] -= this.learningRate * gradient; // Update weight using gradient descent
      }
    }

    // Update biases
    for (let j = 0; j < biases.length; j++) {
      const gradient =
        error[j] *
        this.activationFunctionDerivative(
          this.networkLayer[layerIndex + 1][j],
          activationFunction
        );
      biases[j] -= this.learningRate * gradient; // Update bias
    }
  }

  // Method to perform backpropagation and update the network weights and biases
  learn(targetOutput) {
    // Step 1: Compute error at the output layer
    const outputLayer = this.networkLayer[this.networkLayer.length - 1];
    let error = this.computeOutputError(targetOutput, outputLayer);

    // Step 2: Backpropagate the error through the layers
    for (let i = this.networkLayer.length - 2; i >= 0; i--) {
      // Update weights and biases for the current layer
      const currentLayer = this.networkLayer[i];
      this.updateWeightsAndBiases(i, currentLayer, error);

      // Step 3: Propagate the error back to the previous layer
      // Compute the error for the previous layer
      error = this.networkWeight[i].map((_, nodeIndex) => {
        return this.networkWeight[i].reduce(
          (sum, weight, prevLayerNodeIndex) => {
            return sum + weight[nodeIndex] * error[prevLayerNodeIndex];
          },
          0
        );
      });
    }
  }

  // Helper method to add layers with appropriate initialization
  addLayers(layerSizes, activationFunctions) {
    if (layerSizes.length < 3) {
      throw new Error("A neural network should have at least 3 layers.");
    }

    // Initialize layers, weights, and biases
    for (let i = 0; i < layerSizes.length; i++) {
      //initialize all the nodes with random value
      this.networkLayer[i] = new Array(layerSizes[i]).fill(Math.random()); // Randomly initialized layer values

      if (i < layerSizes.length - 1) {
        // Initialize weights for layers (Xavier or He initialization according to the activation function)
        this.networkWeight.push(
          this.addWeight(
            layerSizes[i],
            layerSizes[i + 1],
            activationFunctions[i]
          )
        );
        this.bias.push(new Array(layerSizes[i + 1]).fill(Math.random())); // Random biases for the next layer
      }
    }

    //If activation function is not given then use sigmoid as default activation function
    this.activationFunction =
      activationFunctions || Array(layerSizes.length - 1).fill("sigmoid");

  }

  // Method to initialize weights with Xavier or He initialization based on activation function
  addWeight(fromNodeLength, toNodeLength, activationFunction) {
    let output = [];
    let limit;

    // Xavier Initialization for sigmoid/tanh or He Initialization for ReLU
    if (activationFunction === "sigmoid" || activationFunction === "tanh") {
      limit = Math.sqrt(6 / (fromNodeLength + toNodeLength)); // Xavier initialization
    } else if (activationFunction === "relu") {
      limit = Math.sqrt(2 / fromNodeLength); // He initialization
    } else {
      throw new Error(
        "Unsupported activation function for weight initialization"
      );
    }

    for (let i = 0; i < fromNodeLength; i++) {
      let temp = [];
      for (let j = 0; j < toNodeLength; j++) {
        temp[j] = Math.random() * 2 * limit - limit; // Random values in the range from +limit to -limit
      }
      output[i] = temp;
    }

    return output;
  }

  //Method to get the out put of the network from the last layer
  getLastNetworkLayer() {
    console.log(this.networkLayer);
    return this.networkLayer[this.networkLayer.length - 1];
  }

  //Method to set the learning rate other than default value
  setLearningRate(num) {
    this.learningRate = num;
  }

  // Method to print a specific item from the network
  printConstructor(item) {
    console.log(`${item}:`, this[item]);
  }
}

let test = new FeedForward();
test.addLayers([2, 2, 1], ["sigmoid", "sigmoid"]); // 2 inputs, 2 hidden nodes, 1 output
test.printConstructor("networkLayer");
test.printConstructor("networkWeight");
test.printConstructor("bias");
test.printConstructor("activationFunction");

// XOR Training Data
let xorInputs = [
  [0, 0], // Input 1
  [0, 1], // Input 2
  [1, 0], // Input 3
  [1, 1], // Input 4
];

let xorOutputs = [
  [0], // Output for [0, 0]
  [1], // Output for [0, 1]
  [1], // Output for [1, 0]
  [0], // Output for [1, 1]
];

// Train the network
for (let i = 0; i < 10000; i++) {
  // Train for 10000 iterations
  for (let j = 0; j < xorInputs.length; j++) {
    test.run(xorInputs[j]);
    test.learn(xorOutputs[j]);
  }
}

// Test the network on XOR data
xorInputs.forEach((input) => {
  test.run(input);
  console.log(`Input: ${input} => Output: ${test.getLastNetworkLayer()}`);
});
