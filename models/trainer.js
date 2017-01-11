var nj = require('numjs');
var DataUtility = require('../utils/data');
var mu = require('../utils/matrix');

function Trainer() {
  this.neuralNetwork;
  this.lambda = 0.001;
  this.scalar = 3;
  this.data = new DataUtility();
  this.cost;
  this.gradient;
  this.yHat;
  this.delta = [];
  this.dJdW = [];

  this.train = function (reps) {
    var trainX = this.data.getTrainingX();
    var trainY = this.data.getTrainingY();
    var w = this.neuralNetwork.dendriteLayers;
    console.log("\nComputing cost function - " + reps + " iterations");

    for (var d = 0; d < reps; d++) {
      this.yHat = this.neuralNetwork.forward(trainX);
      var cost = this.computeCost(trainX, trainY);
      this.computeGradient(trainX, trainY);

      for (var i = 0; i < this.dJdW.length; i++) {
        var scalar = this.dJdW[i].assign(this.scalar);
        var newWeights = nj.array(w[i]).subtract(scalar.multiply(this.dJdW[i])).tolist();
        this.neuralNetwork.dendriteLayers[i] = newWeights;
      }

      this.cost = cost;

      if (d * 10 % reps === 0) {
        console.log("Cost: " + this.cost + " (" + Number(d/reps*100).toFixed(0) + "%)");
      }
    }

    console.log("Cost: " + this.cost + " (100%)");
  }

  this.computeCost = function (x, y) {
    var x = nj.array(x);
    var y = nj.array(y);
    var dl = this.neuralNetwork.dendriteLayers;
    var w = 0;
    for (var i = 0; i < dl.length; i++) {
      w += nj.array(dl[i]).pow(2).sum();
    }
    return (0.5 * y.subtract(this.yHat).pow(2).sum() / x.shape[0]) + ((this.lambda / 2) * w);
  }

  this.computeGradient = function (x, y) {
    var x = nj.array(x);
    var y = nj.array(y);
    var zOutput = nj.array(this.neuralNetwork.outputLayer.activatePrime(this.neuralNetwork.outputLayer.z));

    var hl = this.neuralNetwork.hiddenLayers;
    var dl = this.neuralNetwork.dendriteLayers;

    // initial backward propagation from output layer
    this.delta[hl.length] = nj.negative(y.subtract(this.yHat)).multiply(zOutput);
    this.dJdW[hl.length] = nj.dot(hl[hl.length - 1].a.T, this.delta[hl.length]);
    this.dJdW[hl.length] = this.dJdW[hl.length].add(this.dJdW[hl.length].assign(this.lambda).multiply(nj.array(dl[hl.length])));

    // continue backward propagation through hidden layers of network
    for (var i = hl.length - 1; i > 0; i--) {
      this.delta[i] = nj.dot(this.delta[i + 1], nj.array(dl[i + 1]).T).multiply(nj.array(hl[i].activatePrime(hl[i].z)));
      this.dJdW[i] = nj.dot(hl[i].a.T, this.delta[i]);
      this.dJdW[i] = this.dJdW[i].add(this.dJdW[i].assign(this.lambda).multiply(nj.array(dl[i])));
    }

    // final backward propagation
    this.delta[0] = nj.dot(this.delta[1], nj.array(dl[1]).T).multiply(nj.array(hl[0].activatePrime(hl[0].z)));
    this.dJdW[0] = nj.dot(x.T, this.delta[0]);
  }

  this.printResults = function (x, y) {
    var yHat = this.neuralNetwork.forward(x).tolist();
    for (var i = 0; i < yHat.length; i++) {
      var line = "Prediction: " + yHat[i] + "; ";
      line += "Actual: " + y[i] + ";";
      console.log(line);
    }
  }
}

module.exports = Trainer;
