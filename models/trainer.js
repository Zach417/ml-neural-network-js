var fs = require('fs');
var path = require('path');

var nj = require('../utils/numjs');
var DataUtility = require('../utils/data');
var mu = require('../utils/matrix');

function Trainer() {
  this.neuralNetwork;
  this.lambda = 0.01;
  this.scalar = 0.000001;
  this.data = new DataUtility();
  this.goal;
  this.cost;
  this.gradient;
  this.yHat;
  this.delta = [];
  this.dJdW = [];
  this.x = this.data.getTrainingX();
  this.y = this.data.getTrainingY();

  this.train = function (epochs) {
    var w = this.neuralNetwork.weights;

    console.log("\nComputing cost function - " + epochs + " iterations");
    for (var d = 0; d < epochs; d++) {
      this.yHat = this.neuralNetwork.forward(this.x);
      var cost = this.computeCost(this.x, this.y);
      this.computeGradient(this.x, this.y);

      for (var i = 0; i < this.dJdW.length; i++) {
        var scalar = this.dJdW[i].assign(this.scalar);
        var newWeights = nj.array(w[i]).subtract(scalar.multiply(this.dJdW[i])).tolist();
        this.neuralNetwork.weights[i] = newWeights;
      }

      this.cost = cost;

      if (d * 10 % epochs === 0) {
        //console.log("Weights: " + this.neuralNetwork.weights);
        console.log("Cost: " + Number(this.cost).toFixed(8) + " (" + Number(d/epochs*100).toFixed(2) + "%)");
      } else {
        process.stdout.write("Cost: " + Number(this.cost).toFixed(8) + " (" + Number(d/epochs*100).toFixed(2) + "%)               \r");
      }

      if (this.goal && this.cost < this.goal) {
        console.log("Performance goal met");
        return;
      }
    }

    console.log("Cost: " + Number(this.cost).toFixed(8) + " (100%)");
  }

  this.computeCost = function (x, y) {
    var x = nj.array(x);
    var y = nj.array(y);
    var dl = this.neuralNetwork.weights;
    var w = 0;
    for (var i = 0; i < dl.length; i++) {
      w += nj.array(dl[i]).pow(2).sum();
    }

    return (0.5 * y.subtract(this.yHat).pow(2).sum() / x.shape[0]) + ((this.lambda / 2) * w);
  }

  this.computeGradient = function (x, y) {
    var x = nj.array(x);
    var y = nj.array(y);
    var zOutput = nj.array(this.neuralNetwork.output.activatePrime(this.neuralNetwork.output.z));

    var hidden = this.neuralNetwork.hidden;
    var weights = this.neuralNetwork.weights;

    // initial backward propagation from output layer
    this.delta[hidden.length] = nj.negative(y.subtract(this.yHat)).multiply(zOutput);
    this.dJdW[hidden.length] = nj.dot(hidden[hidden.length - 1].a.T, this.delta[hidden.length]);
    this.dJdW[hidden.length] = this.dJdW[hidden.length].add(this.dJdW[hidden.length].assign(this.lambda).multiply(nj.array(weights[hidden.length])));

    // continue backward propagation through hidden layers of network
    for (var i = hidden.length - 1; i > 0; i--) {
      this.delta[i] = nj.dot(this.delta[i + 1], nj.array(weights[i + 1]).T).multiply(nj.array(hidden[i].activatePrime(hidden[i].z)));
      this.dJdW[i] = nj.dot(hidden[i - 1].a.T, this.delta[i]);
      this.dJdW[i] = this.dJdW[i].add(this.dJdW[i].assign(this.lambda).multiply(nj.array(weights[i])));
    }

    // final backward propagation
    this.delta[0] = nj.dot(this.delta[1], nj.array(weights[1]).T).multiply(nj.array(hidden[0].activatePrime(hidden[0].z)));
    this.dJdW[0] = nj.dot(x.T, this.delta[0]);
  }

  this.generateFromFile = function (fileName) {
    var filePath;
    if (fileName.startsWith(path.join(__dirname, '..'))) {
      filePath = fileName;
    } else {
      filePath = path.join(__dirname, '..', '/data/weights/', fileName);
    }
    var json = JSON.parse(fs.readFileSync(filePath));
    this.neuralNetwork.generate(json);
  }

  this.writeToFile = function () {
    var neuralJSON = this.neuralNetwork.toJSON();
    var json = JSON.stringify(neuralJSON);
    var filePath = path.join(__dirname, '..', '/data/weights/', neuralJSON.name + '.json');
    fs.writeFile(filePath, json, 'utf8', function (err) {
      if (err) return console.log(err);
      console.log("Neural Network JSON saved to " + filePath)
    });
  }

  this.printResults = function (x, y) {
    var n = 15 / x.length;
    var yHat = this.neuralNetwork.forward(x).tolist();
    for (var i = 0; i < yHat.length; i++) {
      if (Math.random() < n) {
        var line = "Prediction: " + yHat[i] + "; ";
        line += "Actual: " + y[i] + ";";
        console.log(line);
      }
    }
  }
}

module.exports = Trainer;
