var nj = require('numjs');
var DataUtility = require('../utils/data');
var mu = require('../utils/matrix');

function Trainer() {
  this.neuralNetwork;
  this.dataUtility = new DataUtility();
  this.cost;
  this.gradient;
  this.yHat;
  this.scalar = 3;
  this.delta = [];
  this.dJdW = [];

  this.train = function (reps) {
    var trainX = this.dataUtility.getTrainingX();
    var trainY = this.dataUtility.getTrainingY();
    var w = this.neuralNetwork.dendriteLayers;

    for (var d = 0; d < reps; d++) {
      this.yHat = this.neuralNetwork.forward(trainX);
      var cost = this.computeCost(trainY);
      this.computeGradient(trainX, trainY);

      for (var i = 0; i < this.dJdW.length; i++) {
        var scalar = this.dJdW[i].assign(this.scalar);
        var newWeights = nj.array(w[i]);
        if (!this.cost || cost < this.cost) {
          newWeights = newWeights.subtract(scalar.multiply(this.dJdW[i])).tolist();
        } else {
          newWeights = newWeights.add(scalar.multiply(this.dJdW[i])).tolist();
        }
        this.neuralNetwork.dendriteLayers[i] = newWeights;
      }

      this.cost = cost;

      if (d * 10 % reps === 0) {
        console.log("Cost: " + this.cost);
      }
    }
  }

  this.computeCost = function (y) {
    var y = nj.array(y);
    return 0.5 * y.subtract(this.yHat).pow(2).sum();
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

    // continue backward propagation through hidden layers of network
    for (var i = hl.length - 1; i > 0; i--) {
      this.delta[i] = nj.dot(this.delta[i + 1], nj.array(dl[i + 1]).T).multiply(nj.array(hl[i].activatePrime(hl[i].z)));
      this.dJdW[i] = nj.dot(hl[i].a.T, this.delta[i]);
    }

    // final backward propagation
    this.delta[0] = nj.dot(this.delta[1], nj.array(dl[1]).T).multiply(nj.array(hl[0].activatePrime(hl[0].z)));
    this.dJdW[0] = nj.dot(x.T, this.delta[0]);
  }

  this.printResults = function () {
    var trainX = this.dataUtility.getTrainingX();
    var trainY = this.dataUtility.getTrainingY();
    this.yHat = this.neuralNetwork.forward(trainX);
    var yHatList = this.yHat.tolist();
    for (var i = 0; i < yHatList.length; i++) {
      var line = "Prediction: " + yHatList[i] + "; ";
      line += "Actual: " + trainY[i] + ";";
      console.log(line);
    }
  }
}

module.exports = Trainer;
