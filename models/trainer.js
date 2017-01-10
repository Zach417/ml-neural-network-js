var DataUtility = require('../utils/data');
var mu = require('../utils/matrix');

function Trainer() {
  this.neuralNetwork;
  this.dataUtility = new DataUtility();
  this.cost;
  this.gradient;
  this.yHat;
  this.scalar = 10;

  this.train = function (reps) {
    var trainX = this.dataUtility.getTrainingX();
    var trainY = this.dataUtility.getTrainingY();
    for (var d = 0; d < reps; d++) {
      this.yHat = this.neuralNetwork.forward(trainX);
      var cost = this.computeCost(trainY);
      this.computeGradient(trainX, trainY);

      if (!this.cost || cost < this.cost) {
        this.neuralNetwork.dendriteLayers[0] = mu.subtract(mu.multiplyInt(this.dJdW1, this.scalar), this.neuralNetwork.dendriteLayers[0]);
        this.neuralNetwork.dendriteLayers[1] = mu.subtract(mu.multiplyInt(this.dJdW2, this.scalar), this.neuralNetwork.dendriteLayers[1]);
      } else {
        this.neuralNetwork.dendriteLayers[0] = mu.add(mu.multiplyInt(this.dJdW1, this.scalar), this.neuralNetwork.dendriteLayers[0]);
        this.neuralNetwork.dendriteLayers[1] = mu.add(mu.multiplyInt(this.dJdW2, this.scalar), this.neuralNetwork.dendriteLayers[1]);
      }

      this.cost = cost;
      console.log(this.cost);
    }
  }

  this.computeCost = function (y) {
    return 0.5 * mu.sum(mu.pow(mu.subtract(y,this.yHat), 2));
  }

  this.computeGradient = function (x, y) {
    var differential = mu.multiplyInt(mu.subtract(y,this.yHat), -1);
    var outputLayerAFP = this.neuralNetwork.outputLayer.activatePrime(this.neuralNetwork.outputLayer.z);
    var delta3 = mu.multiply(outputLayerAFP, differential); // element multiplication
    this.dJdW2 = mu.multiply(mu.transpose(this.neuralNetwork.hiddenLayers[0].a), delta3); // matrix multiplication
    var delta2 = mu.multiply(delta3, mu.transpose(this.neuralNetwork.dendriteLayers[1])); // matrix multiplication
    delta2 = mu.multiply(delta2, this.neuralNetwork.hiddenLayers[0].activatePrime(this.neuralNetwork.hiddenLayers[0].z));
    this.dJdW1 = mu.multiply(mu.transpose(x), delta2); // dot multiplication
  }

  this.printResults = function () {
    var trainX = this.dataUtility.getTrainingX();
    var trainY = this.dataUtility.getTrainingY();
    this.yHat = this.neuralNetwork.forward(trainX);
    for (var i = 0; i < this.yHat.length; i++) {
      var line = "Prediction: " + Number(this.yHat[i]).toFixed(4) + "; ";
      line += "Actual: " + Number(trainY[i]).toFixed(4) + ";";
      console.log(line);
    }
  }
}

module.exports = Trainer;
