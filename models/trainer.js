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

      var scalar1 = this.dJdW1.assign(this.scalar);
      var scalar2 = this.dJdW2.assign(this.scalar);

      if (!this.cost || cost < this.cost) {
        this.neuralNetwork.dendriteLayers[0] = nj.array(w[0]).subtract(scalar1.multiply(this.dJdW1)).tolist();
        this.neuralNetwork.dendriteLayers[1] = nj.array(w[1]).subtract(scalar2.multiply(this.dJdW2)).tolist();
      } else {
        this.neuralNetwork.dendriteLayers[0] = nj.array(w[0]).add(scalar1.multiply(this.dJdW1)).tolist();
        this.neuralNetwork.dendriteLayers[1] = nj.array(w[1]).add(scalar2.multiply(this.dJdW2)).tolist();
      }

      this.cost = cost;

      if (d * 10 % reps === 0) {
        console.log("Cost: " + this.cost);
      }
    }
  }

  this.computeCost = function (y) {
    var y = nj.array(y);
    var yHat = nj.array(this.yHat);
    return 0.5 * y.subtract(yHat).pow(2).sum();
  }

  this.computeGradient = function (x, y) {
    var x = nj.array(x);
    var y = nj.array(y);
    var yHat = nj.array(this.yHat);
    var z3 = nj.array(this.neuralNetwork.outputLayer.activatePrime(this.neuralNetwork.outputLayer.z));
    var a2 = nj.array(this.neuralNetwork.hiddenLayers[0].a);
    var w2 = nj.array(this.neuralNetwork.dendriteLayers[1]);
    var z2 = nj.array(this.neuralNetwork.hiddenLayers[0].activatePrime(this.neuralNetwork.hiddenLayers[0].z));

    var delta3 = nj.negative(y.subtract(yHat)).multiply(z3);
    this.dJdW2 = nj.dot(a2.T, delta3);
    var delta2 = nj.dot(delta3, w2.T).multiply(z2);
    this.dJdW1 = nj.dot(x.T, delta2);
  }

  this.printResults = function () {
    var trainX = this.dataUtility.getTrainingX();
    var trainY = this.dataUtility.getTrainingY();
    this.yHat = this.neuralNetwork.forward(trainX);
    for (var i = 0; i < this.yHat.length; i++) {
      var line = "Prediction: " + this.yHat[i] + "; ";
      line += "Actual: " + trainY[i] + ";";
      console.log(line);
    }
  }
}

module.exports = Trainer;
