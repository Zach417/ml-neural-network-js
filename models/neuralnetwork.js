var nj = require('numjs');
var Layer = require('./layer');

function NeuralNetwork () {
  this.inputLayer = new Layer();
  this.hiddenLayers = []; // array of layers
  this.outputLayer = new Layer();
  this.dendriteLayers = []; // array of matrices (weights)

  this.addHiddenLayer = function (size, activation) {
    var layer = new Layer();
    layer.size = size;
    layer.setActivation(activation);
    this.hiddenLayers.push(layer);
  }

  this.generate = function () {
    this.generateDendrites();
  }

  this.forward = function (x) {
    var z = nj.array(x);
    var a = nj.array(x);

    for (var i = 0; i < this.hiddenLayers.length; i++) {
      z = a.dot(this.dendriteLayers[i]);
      a = this.hiddenLayers[i].activate(z);
      this.hiddenLayers[i].z = z;
      this.hiddenLayers[i].a = a;
    }

    this.outputLayer.z = a.dot(this.dendriteLayers[this.dendriteLayers.length - 1]);
    this.yHat = this.outputLayer.activate(this.outputLayer.z);
    return this.yHat;
  }

  this.generateDendrites = function () {
    if (this.hiddenLayers.length > 0) {
      var z1 = this.inputLayer.size;
      var z2 = z1;
      for (var i = 0; i < this.hiddenLayers.length; i++) {
        z2 = this.hiddenLayers[i].size;
        this.dendriteLayers.push(nj.random(z1, z2));
        z1 = z2;
      }

      z2 = this.outputLayer.size;
      this.dendriteLayers.push(nj.random(z1, z2));
    } else {
      var z1 = this.inputLayer.size;
      var z2 = this.outputLayer.size;
      this.dendriteLayers.push(nj.random(z1, z2));
    }
  }
}

module.exports = NeuralNetwork;
