var fs = require('fs');
var path = require('path');

var data = [];
var max = Math.PI * 2;
var num = 1000;
for (var i = 0; i <= num; i++) {
  var angle = max * (i / num);

  var noise = Math.random() * 0.2;
  noise *= Math.floor(Math.random()*2) == 1 ? 1 : -1;

  data.push({
    x: [angle],
    y: [(Math.sin(angle) + noise)]
  })
}

module.exports = data;
