var nj = require('../utils/numjs');
var Layer = require('./layer');

function NeuralNetwork () {
  this.input = new Layer();
  this.hidden = [];
  this.output = new Layer();
  this.weights = [];

  this.addHiddenLayer = function (size, activation) {
    var layer = new Layer();
    layer.size = size;
    layer.setActivation(activation);
    this.hidden.push(layer);
  }

  this.generate = function (json) {
    if (json) {
      this.generateFromJson(json);
    } else {
      this.generateDendrites();
    }
  }

  this.generateFromJson = function (json) {
    // input layer
    this.input.size = json.input.size;
    this.input.setActivation(json.input.activation);

    // hidden layer
    this.hidden = [];
    for (var i = 0; i < json.hidden.length; i++) {
      var layer = new Layer();
      layer.size = json.hidden[i].size;
      layer.setActivation(json.hidden[i].activation);
      this.hidden.push(layer);
    }

    // output layer
    this.output.size = json.output.size;
    this.output.setActivation(json.output.activation);

    // weights
    this.weights = [];
    for (var i = 0; i < json.weights.length; i++) {
      var w = nj.array(json.weights[i]);
      this.weights.push(w);
    }
  }

  this.forward = function (x) {
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

  this.generateDendrites = function () {
    if (this.hidden.length > 0) {
      var z1 = this.input.size;
      var z2 = z1;
      for (var i = 0; i < this.hidden.length; i++) {
        z2 = this.hidden[i].size;
        this.weights.push(nj.random(z1, z2));
        z1 = z2;
      }

      z2 = this.output.size;
      this.weights.push(nj.random(z1, z2));
    } else {
      var z1 = this.input.size;
      var z2 = this.output.size;
      this.weights.push(nj.random(z1, z2));
    }
  }

  this.toJSON = function () {
    var json = {
      name: "",
      input: {
        size: this.input.size,
        activation: this.input.activationName,
      },
      hidden: [],
      output: {
        size: this.output.size,
        activation: this.output.activationName,
      },
      weights: this.weights,
    }

    // set default name
    var date = new Date();
    json.name =
      date.getFullYear().toString()
      + date.getMonth().toString()
      + date.getDate().toString()
      + date.getHours().toString()
      + date.getMinutes().toString()
      + date.getSeconds().toString();

    // set hidden layers
    for (var i = 0; i < this.hidden.length; i++) {
      var layer = this.hidden[i];
      json.hidden.push({
        size: layer.size,
        activation: layer.activationName,
      });
    }

    return json;
  }
}

module.exports = NeuralNetwork;
