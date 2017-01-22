var fs = require('fs');
var NeuralNetworkJS = require('../../index.js');
var NeuralNowUtils = require('neural-now-utils');
var getData = require('./data');

var NeuralNetwork = NeuralNetworkJS.NeuralNetwork;
var Trainer = NeuralNetworkJS.Trainer;

function train (trainX, trainY, testX, testY) {
  var trainer = new Trainer();
  //trainer.generateFromFile("spam.json");

  NeuralNowUtils.Text.toBigramVector("Check out the amazing offer!!", function (vector) {
    // // architect neural network
    // var neuralNetwork = new NeuralNetwork();
    // neuralNetwork.input.size = trainX[0].length;
    // neuralNetwork.input.setActivation("linear");
    // neuralNetwork.addHiddenLayer(10, "sigmoid");
    // neuralNetwork.addHiddenLayer(10, "sigmoid");
    // neuralNetwork.addHiddenLayer(10, "sigmoid");
    // neuralNetwork.output.size = trainY[0].length;
    // neuralNetwork.output.setActivation("sigmoid");
    // neuralNetwork.generate();
    // trainer.neuralNetwork = neuralNetwork;

    trainer.generateFromFile("new-neural-network.json");

    var x = [vector];
    var y = [[0,1]];
    console.log(trainer.neuralNetwork.forward(x));

    trainer.x = trainX;
    trainer.y = trainY;
    trainer.scalar = 0.8;
    trainer.lambda = 0.01;
    trainer.goal = 0.0;
    trainer.train(100);

    // print results
    console.log("\nTraining data set results");
    trainer.printResults(trainX, trainY);

    console.log("\nTesting data set results");
    trainer.printResults(testX, testY);
  });
}


var trainX = JSON.parse(fs.readFileSync(__dirname + "/trainX.json"));
var trainY = JSON.parse(fs.readFileSync(__dirname + "/trainY.json"));
var testX = JSON.parse(fs.readFileSync(__dirname + "/testX.json"));
var testY = JSON.parse(fs.readFileSync(__dirname + "/testY.json"));
train(trainX, trainY, testX, testY);

// var idx = 0;
// getData(function (data) {
//   var x = [];
//   var y = [];
//
//   var epochs = data.x.length;
//   for (var i = 0; i < epochs; i++) {
//     NeuralNowUtils.Text.toBigramVector(data.x[i], function (vector) {
//       x.push(vector);
//       y.push(data.y[idx]);
//       idx++;
//       if (idx == epochs - 1) {
//         var length = Math.ceil(x.length * .75);
//         var trainX = x.splice(0,length);
//         var trainY = y.splice(0,length);
//         var testX = x;
//         var testY = y;
//
//         fs.writeFileSync(__dirname + "/trainX.json", JSON.stringify(trainX), "utf-8");
//         fs.writeFileSync(__dirname + "/trainY.json", JSON.stringify(trainY), "utf-8");
//         fs.writeFileSync(__dirname + "/testX.json", JSON.stringify(testX), "utf-8");
//         fs.writeFileSync(__dirname + "/testY.json", JSON.stringify(testY), "utf-8");
//
//         train(trainX, trainY, testX, testY);
//       }
//     });
//   }
// });
