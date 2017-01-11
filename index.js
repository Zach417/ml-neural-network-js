var NeuralNetwork = require('./models/neuralnetwork');
var Trainer = require('./models/trainer');

// setup trainer and gather data
var trainer = new Trainer();
var trainX = trainer.data.getTrainingX();
var trainY = trainer.data.getTrainingY();
var testX = trainX;
var testY = trainY;

// architect neural network
var neuralNetwork = new NeuralNetwork();
neuralNetwork.inputLayer.size = trainX[0].length;
neuralNetwork.addHiddenLayer(5, "sigmoid");
neuralNetwork.addHiddenLayer(5, "sigmoid");
neuralNetwork.addHiddenLayer(5, "sigmoid");
neuralNetwork.outputLayer.size = trainY[0].length;
neuralNetwork.outputLayer.setActivation("sigmoid");
neuralNetwork.generate();

// train
trainer.neuralNetwork = neuralNetwork;
trainer.scalar = 1.0;
trainer.lambda = 0.0;
trainer.train(1000);

// print results
console.log("\nTraining data set results");
trainer.printResults(trainX, trainY);

console.log("\nTesting data set results");
trainer.printResults(testX, testY);
