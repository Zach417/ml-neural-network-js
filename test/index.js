var NeuralNetworkJS = require("../index.js");
var NeuralNetwork = NeuralNetworkJS.NeuralNetwork;
var Trainer = NeuralNetworkJS.Trainer;

// setup trainer and gather data
var trainer = new Trainer();
var trainX = trainer.data.getTrainingX();
var trainY = trainer.data.getTrainingY();
var testX = trainX;
var testY = trainY;

// architect neural network
var neuralNetwork = new NeuralNetwork();
neuralNetwork.input.size = trainX[0].length;
neuralNetwork.input.setActivation("linear");
neuralNetwork.addHiddenLayer(6, "hyperbolic-tangent");
neuralNetwork.addHiddenLayer(6, "hyperbolic-tangent");
neuralNetwork.addHiddenLayer(6, "hyperbolic-tangent");
neuralNetwork.output.size = trainY[0].length;
neuralNetwork.output.setActivation("hyperbolic-tangent");
neuralNetwork.generate();

// train
trainer.neuralNetwork = neuralNetwork;
trainer.scalar = 1.0;
trainer.lambda = 0.0001;
trainer.train(1000);
trainer.writeToFile();

// print results
console.log("\nTraining data set results");
trainer.printResults(trainX.splice(0,10), trainY.splice(0,10));

//console.log("\nTesting data set results");
//trainer.printResults(testX, testY);
