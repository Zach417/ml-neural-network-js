var data = [];
for (var i = 0; i < 500; i++) {
  data.push({
    x: [i / Math.PI],
    y: [Math.sin(i / Math.PI)]
  })
}
module.exports = data;
