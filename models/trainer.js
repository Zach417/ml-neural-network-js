var DataUtility = require('../utils/data');
var MatrixUtility = require('../utils/matrix');

function Trainer() {
  this.neuralNetwork;
  this.dataUtility = new DataUtility();
  this.cost;
  this.gradient;
  this.yHat;

  this.train = function (reps) {
    var trainX = this.dataUtility.getTrainingX();
    var trainY = this.dataUtility.getTrainingY();
    for (var d = 0; d < reps; d++) {
      this.computeCost(trainX, trainY)
    }
  }

  this.computeCost = function (x, y) {
    this.yHat = this.neuralNetwork.forward(x);
    this.cost = 0.5 * MatrixUtility.sum(MatrixUtility.pow(MatrixUtility.subtract(y,this.yHat), 2));
    return this.cost;
  }

  this.printResults = function () {
    var trainY = this.dataUtility.getTrainingY();
    for (var i = 0; i < this.yHat.length; i++) {
      var line = "Prediction: " + Number(this.yHat[i]).toFixed(4) + "; ";
      line += "Actual: " + Number(trainY[i]).toFixed(4) + ";";
      console.log(line);
    }
  }
}

module.exports = Trainer;
