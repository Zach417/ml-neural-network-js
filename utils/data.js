var DataSet = require('../data/train');

function Data() {
  this.set = DataSet;

  this.getTrainingX = function () {
    var x = [];
    for (var i = 0; i < this.set.length; i++) {
      x.push(this.set[i].x);
    }
    return x
  }

  this.getTrainingY = function () {
    var y = [];
    for (var i = 0; i < this.set.length; i++) {
      y.push(this.set[i].y);
    }
    return y
  }
}

module.exports = Data;
