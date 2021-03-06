module.exports = {
  generateMatrix: function (x, y) {
    var result = [];
    for (var i = 0; i < x; i++) {
      var row = [];
      for (var j = 0; j < y; j++) {
        row.push(Math.random(-10,10));
      }
      result.push(row);
    }
    return result;
  },

  transpose: function (m) {
    var w = m.length ? m.length : 0,
    h = m[0] instanceof Array ? m[0].length : 0;
    if(h === 0 || w === 0) { return []; }
    var i, j, t = [];
    for(i=0; i<h; i++) {
      t[i] = [];
      for(j=0; j<w; j++) {
        t[i][j] = m[j][i];
      }
    }
    return t;
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

  add: function (m1, m2) {
    var result = [];
    if (m1.length == m2.length && m1.length > 0 && m1[0].length == m2[0].length) {
      for (var i = 0; i < m1.length; i++) {
        result[i] = [];
        for (var j = 0; j < m1[i].length; j++) {
          result[i].push(m2[i][j] + m1[i][j]);
        }
      }
    } else {
      for (var i = 0; i < m1.length; i++) {
        result[i] = [];
        for (var j = 0; j < m2[0].length; j++) {
          var sum = 0;
          for (var k = 0; k < m1[0].length; k++) {
            sum += m2[k][j] + m1[i][k];
          }
          result[i][j] = sum;
        }
      }
    }
    return result;
  },

  subtract: function (m1, m2) {
    var result = [];
    if (m1.length == m2.length && m1.length > 0 && m1[0].length == m2[0].length) {
      for (var i = 0; i < m1.length; i++) {
        result[i] = [];
        for (var j = 0; j < m1[i].length; j++) {
          result[i].push(m2[i][j] - m1[i][j]);
        }
      }
    } else {
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
  },

  multiplyInt: function (m, z) {
    var result = [];
    for (var i = 0; i < m.length; i++) {
      result[i] = [];
      for (var j = 0; j < m[i].length; j++) {
        result[i].push(m[i][j]*z);
      }
    }
    return result;
  },
}
