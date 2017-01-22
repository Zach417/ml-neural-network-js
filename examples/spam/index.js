var fs = require('fs');
var NeuralNetworkJS = require('../../index.js');
var NeuralNowUtils = require('neural-now-utils');
var getData = require('./data');

var NeuralNetwork = NeuralNetworkJS.NeuralNetwork;
var Trainer = NeuralNetworkJS.Trainer;

function train (trainX, trainY, testX, testY) {
  var trainer = new Trainer();

  // // architect neural network
  // var neuralNetwork = new NeuralNetwork();
  // neuralNetwork.input.size = trainX[0].length;
  // neuralNetwork.input.setActivation("linear");
  // neuralNetwork.addHiddenLayer(3, "sigmoid");
  // neuralNetwork.addHiddenLayer(3, "sigmoid");
  // neuralNetwork.output.size = trainY[0].length;
  // neuralNetwork.output.setActivation("sigmoid");
  // neuralNetwork.generate();
  // trainer.neuralNetwork = neuralNetwork;

  trainer.generateFromFile("spam-classification.json");

  trainer.x = trainX;
  trainer.y = trainY;
  trainer.scalar = 0.000001;
  trainer.lambda = 0.0001;
  trainer.goal = 0.0;
  trainer.train(1000);

  // print results
  console.log("\nTraining data set results");
  trainer.printResults(trainX, trainY);

  console.log("\nTesting data set results");
  trainer.printResults(testX, testY);
}

var trainX = JSON.parse(fs.readFileSync(__dirname + "/trainX.json"));
var trainY = JSON.parse(fs.readFileSync(__dirname + "/trainY.json"));
var testX = JSON.parse(fs.readFileSync(__dirname + "/testX.json"));
var testY = JSON.parse(fs.readFileSync(__dirname + "/testY.json"));
train(trainX, trainY, testX, testY);

// getData(function (data) {
//   var testX = [];
//   var testY = [];
//   var trainX = [];
//   var trainY = [];
//
//   for (var i = 0; i < data.x.length; i++) {
//     var vector = NeuralNowUtils.Text.toBigramVector(data.x[i]);
//     if (Math.random() < .25) {
//       testX.push(vector);
//       testY.push(data.y[i]);
//     } else {
//       trainX.push(vector);
//       trainY.push(data.y[i]);
//     }
//   }
//
//   var hamCount = 0;
//   var spamCount = 0;
//   for (var i = 0; i < testY.length; i++) {
//     if (testY[i][1] === 1) {
//       spamCount++;
//     } else {
//       hamCount++;
//     }
//   }
//   console.log(testY[0]);
//   console.log(hamCount, spamCount);
//
//   var hamCount = 0;
//   var spamCount = 0;
//   for (var i = 0; i < trainY.length; i++) {
//     if (trainY[i][1] === 1) {
//       spamCount++;
//     } else {
//       hamCount++;
//     }
//   }
//   console.log(trainY[0]);
//   console.log(hamCount, spamCount);
//
//   fs.writeFileSync(__dirname + "/trainX.json", JSON.stringify(trainX), "utf-8");
//   fs.writeFileSync(__dirname + "/trainY.json", JSON.stringify(trainY), "utf-8");
//   fs.writeFileSync(__dirname + "/testX.json", JSON.stringify(testX), "utf-8");
//   fs.writeFileSync(__dirname + "/testY.json", JSON.stringify(testY), "utf-8");
//
//   train(trainX, trainY, testX, testY);
// });
