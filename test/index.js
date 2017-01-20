var NeuralNetworkJS = require("../index.js");
var NeuralNowUtils = require('neural-now-utils');
var NeuralNetwork = NeuralNetworkJS.NeuralNetwork;
var Trainer = NeuralNetworkJS.Trainer;

NeuralNowUtils.Text.toBigramVector("This is a test.", function (vector) {
  // setup trainer and gather data
  var trainer = new Trainer();
  var trainX = [vector];
  var trainY = [[1]];

  // architect neural network
  var neuralNetwork = new NeuralNetwork();
  neuralNetwork.input.size = trainX[0].length;
  neuralNetwork.input.setActivation("linear");
  neuralNetwork.addHiddenLayer(3, "sigmoid");
  neuralNetwork.addHiddenLayer(3, "sigmoid");
  neuralNetwork.output.size = trainY[0].length;
  neuralNetwork.output.setActivation("sigmoid");
  neuralNetwork.generate();

  // train
  trainer.x = trainX;
  trainer.y = trainY;
  trainer.neuralNetwork = neuralNetwork;
  trainer.scalar = 0.000001;
  trainer.lambda = 0.001;
  trainer.goal = 0.001;
  trainer.train(10000);
  trainer.writeToFile();

  // print results
  console.log("\nTraining data set results");
  trainer.printResults(trainX.splice(0,10), trainY.splice(0,10));
});
