var fs = require('fs');
var path = require('path');
var readline = require('readline')

module.exports = function (callback) {
  var ham = [];
  var spam = [];

  var lineReader = readline.createInterface({
    input: fs.createReadStream(path.join(__dirname, "SMSSpamCollection.txt"))
  });

  lineReader.on('line', function (line) {
    if (line.startsWith("ham")) {
      ham.push(line.split('\t')[1]);
    } else if (line.startsWith("spam")) {
      spam.push(line.split('\t')[1]);
    }
  });

  lineReader.on('close', function () {
    var x = [];
    var y = [];

    for (var i = 0; i < ham.length; i++) {
      x.push(ham[i]);
      y.push([0]);
    }

    for (var i = 0; i < spam.length; i++) {
      x.push(spam[i]);
      y.push([1]);
    }

    callback({
      x: x,
      y: y,
    });
  });
}
