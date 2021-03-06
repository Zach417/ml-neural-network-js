var fs = require('fs');
var path = require('path');

var data = [];
var max = Math.PI * 2;
var epochs = 10000;
var sets = 100;

var f = function (x) {
  return Math.cos(x);
}

// creates multiple y values for the same x
// in theory should improve training performances
for (var x = 0; x < sets; x++) {
  for (var i = 1; i <= (epochs / sets); i++) {
    var angle = max * (i / (epochs / sets));

    var noiseX = Math.random() * 0.0;
    noiseX *= Math.floor(Math.random()*2) == 1 ? 1 : -1;

    var noiseY = Math.random() * 0.0;
    noiseY *= Math.floor(Math.random()*2) == 1 ? 1 : -1;

    data.push({
      x: [angle + noiseX],
      y: [f(angle + noiseX) + noiseY]
    })
  }
}

module.exports = data;
