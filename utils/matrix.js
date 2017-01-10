module.exports = {
  generateMatrix: function (x, y) {
    var result = [];
    for (var i = 0; i < x; i++) {
      var row = [];
      for (var j = 0; j < y; j++) {
        row.push(1);
      }
      result.push(row);
    }
    return result;
  },

  sum: function (m) {
    var result = 0;
    for (var i = 0; i < m.length; i++) {
      result += m[i];
    }
    return result;
  },

  pow: function (m, z) {
    var result = [];
    for (var i = 0; i < m.length; i++) {
      result.push(Math.pow(m[i],z));
    }
    return result;
  },

  subtract: function (m1, m2) {
    var result = [];
    for (var i = 0; i < m1.length; i++) {
      result[i] = [];
      for (var j = 0; j < m2[0].length; j++) {
        var sum = 0;
        for (var k = 0; k < m1[0].length; k++) {
          sum += m2[k][j] - m1[i][k];
        }
        result[i][j] = sum;
      }
    }
    return result;
  },

  multiply: function (m1, m2) {
    var result = [];
    for (var i = 0; i < m1.length; i++) {
      result[i] = [];
      for (var j = 0; j < m2[0].length; j++) {
        var sum = 0;
        for (var k = 0; k < m1[0].length; k++) {
          sum += m1[i][k] * m2[k][j];
        }
        result[i][j] = sum;
      }
    }
    return result;
  }
}
