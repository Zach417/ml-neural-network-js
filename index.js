var NeuralNetwork = require('./models/neuralnetwork');
var Trainer = require('./models/trainer');

// setup trainer and gather data
var trainer = new Trainer();
var trainX = trainer.dataUtility.getTrainingX();
var trainY = trainer.dataUtility.getTrainingY();

// architect neural network
var neuralNetwork = new NeuralNetwork();
neuralNetwork.inputLayer.size = trainX[0].length;
neuralNetwork.addHiddenLayer(3, "sigmoid");
neuralNetwork.outputLayer.size = trainY[0].length;
neuralNetwork.outputLayer.setActivation("sigmoid");
neuralNetwork.generate();

// train and print results
trainer.neuralNetwork = neuralNetwork;
trainer.train(5);
trainer.printResults();
