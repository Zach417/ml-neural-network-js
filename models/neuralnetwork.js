var nj = require('../utils/numjs');
var Layer = require('./layer');

function NeuralNetwork () {
  this.name = "new-neural-network";
  this.input = new Layer();
  this.hidden = [];
  this.output = new Layer();
  this.weights = [];

  this.addHiddenLayer = function (size, activation, bias) {
    var layer = new Layer();
    layer.size = size;
    layer.bias = bias;
    layer.setActivation(activation);
    this.hidden.push(layer);
  }

  this.generate = function (json) {
    if (json) {
      this.generateFromJson(json);
    } else {
      this.generateWeights();
    }
  }

  this.generateFromJson = function (json) {
    this.name = json.name;

    // input layer
    this.input.size = json.input.size;
    this.input.bias = json.input.bias;
    this.input.setActivation(json.input.activation);

    // hidden layer
    this.hidden = [];
    for (var i = 0; i < json.hidden.length; i++) {
      var layer = new Layer();
      layer.size = json.hidden[i].size;
      layer.bias = json.hidden[i].bias;
      layer.setActivation(json.hidden[i].activation);
      this.hidden.push(layer);
    }

    // output layer
    this.output.size = json.output.size;
    this.output.bias = json.output.bias;
    this.output.setActivation(json.output.activation);

    // weights
    this.weights = [];
    for (var i = 0; i < json.weights.length; i++) {
      var w = nj.array(json.weights[i]);
      this.weights.push(w);
    }
  }

  this.forward = function (x) {
    // add bias to input if necessary
    if (this.input.bias && x[0].length == this.input.size) {
      for (var j = 0; j < x.length; j++) {
        x[j].push(1);
      }
    }

    var z = nj.array(x);
    var a = nj.array(x);

    for (var i = 0; i < this.hidden.length; i++) {
      z = a.dot(this.weights[i]);
      a = this.hidden[i].activate(z);
      this.hidden[i].z = z;
      this.hidden[i].a = a;
    }

    this.output.z = a.dot(this.weights[this.weights.length - 1]);
    this.yHat = this.output.activate(this.output.z);
    return this.yHat;
  }

  this.generateWeights = function () {
    if (this.hidden.length > 0) {
      var z1 = this.input.size;
      if (this.input.bias === true) {
        z1++;
      }
      var z2 = z1;
      for (var i = 0; i < this.hidden.length; i++) {
        z2 = this.hidden[i].size;
        if (this.hidden[i].bias === true) {
          z2++;
        }
        this.weights.push(nj.random(z1, z2));
        z1 = z2;
      }

      z2 = this.output.size;
      this.weights.push(nj.random(z1, z2));
    } else {
      var z1 = this.input.size;
      if (this.input.bias === true) {
        z1++;
      }
      var z2 = this.output.size;
      this.weights.push(nj.random(z1, z2));
    }
  }

  this.toJSON = function () {
    var json = {
      name: this.name,
      input: {
        size: this.input.size,
        bias: this.input.bias,
        activation: this.input.activationName,
      },
      hidden: [],
      output: {
        size: this.output.size,
        bias: this.output.bias,
        activation: this.output.activationName,
      },
      weights: [],
    }

    // set weights
    for (var i = 0; i < this.weights.length; i++) {
      var weight = this.weights[i].tolist();
      json.weights.push(weight);
    }

    // set hidden layers
    for (var i = 0; i < this.hidden.length; i++) {
      var layer = this.hidden[i];
      json.hidden.push({
        size: layer.size,
        bias: layer.bias,
        activation: layer.activationName,
      });
    }

    return json;
  }
}

module.exports = NeuralNetwork;
