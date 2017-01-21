var NeuralNetworkJS = require('../../index.js');
var NeuralNowUtils = require('neural-now-utils');
var getData = require('./data');

var NeuralNetwork = NeuralNetworkJS.NeuralNetwork;
var Trainer = NeuralNetworkJS.Trainer;

function train (trainX, trainY, testX, testY) {
  // setup trainer and gather data
  var trainer = new Trainer();

  // architect neural network
  var neuralNetwork = new NeuralNetwork();
  neuralNetwork.input.size = trainX[0].length;
  neuralNetwork.input.setActivation("linear");
  neuralNetwork.addHiddenLayer(10, "sigmoid");
  neuralNetwork.addHiddenLayer(10, "sigmoid");
  neuralNetwork.output.size = trainY[0].length;
  neuralNetwork.output.setActivation("sigmoid");
  neuralNetwork.generate();

  // train
  trainer.x = trainX;
  trainer.y = trainY;
  trainer.neuralNetwork = neuralNetwork;
  //trainer.generateFromFile("20170210528.json");
  trainer.scalar = 0.000001;
  trainer.lambda = 0.1;
  trainer.goal = 0.0;
  trainer.train(100);
  trainer.writeToFile();

  // print results
  console.log("\nTraining data set results");
  trainer.printResults(trainX, trainY);

  console.log("\nTesting data set results");
  trainer.printResults(testX, testY);
}

var idx = 0;
getData(function (data) {
  var x = [];
  var y = [];

  for (var i = 0; i < data.x.length; i++) {
    NeuralNowUtils.Text.toBigramVector(data.x[i], function (vector) {
      x.push(vector);
      y.push(data.y[idx]);
      idx++;
      if (idx == data.x.length - 1) {
        var length = Math.ceil(x.length * .75);
        var trainX = x.splice(0,length);
        var trainY = y.splice(0,length);
        var testX = x;
        var testY = y;

        train(trainX, trainY, testX, testY);
      }
    });
  }
});
