// TF
const tf = require('@tensorflow/tfjs')
require('@tensorflow/tfjs-node')

// File handling
const { promisify } = require('util')
const fs = require('fs')
const readFile = promisify(fs.readFile)
const writeFile = promisify(fs.writeFile)

// Data
const data = require('./data/export_500.json')
const json = JSON.parse(data)

function dataToTensor(json) {
    let ys = []
    let xs = []
    let keys = []
    json.forEach(row => {

        // Take bikesavailable as output value (y)
        ys.push([row['X.bikesAvailable.']])
        ys = ys.map(Number)

        // Delete unnecessary stuff from row object
        delete row['X.name.']
        delete row['X.id.']
        delete row['X.weekday.']
        delete row['X.date.']
        delete row['X.bikesAvailable.']

        // Take remaining object values as input values (x)
        xs.push(Object.values(row).map(Number))

        // Store object keys for names
        keys = Object.keys(row)
    })

    // Remove some elements for tests
    yTest = ys.splice(0, 10)
    xTest = xs.splice(0, 10)

    // Store x & y shape
    let xShape = [xs.length, xs[0].length]
    let yShape = [ys.length, 1]

    return {
        xs: xs,
        ys: ys,
        keys: keys,
        xShape: xShape,
        yShape: yShape,
        xTest: xTest,
        yTest: yTest
    }
}

async function dothemodel() {

    // Parse input data into tensors. IRL this should probably return batches
    const arraydata = dataToTensor(json)

    /*
    *  tfjs provides a "Keras" API, which is easier to comprehend than pure tensorflow.
    *  Let's use that. (https://js.tensorflow.org/tutorials/tfjs-layers-for-keras-users.html)
    */

    // Sequential model = everything is applied in sequence
    const model = tf.sequential()

    // Lets define an optimizer, this pokes around the weights to help unstuck the training process
    const optimizer = tf.train.adamax(0.1);

    // Add some layers
    // A dense (fully connected) one, with 72 units
    model.add(tf.layers.dense({units: 72, inputShape: [arraydata.xShape[1]]}))

    // leakyReLU activation function. This determines when the neurons are activated, this one is supposedly cool.
    model.add(tf.layers.leakyReLU())

    // Dropout = disconnect some neurons to make them learn better
    model.add(tf.layers.dropout(0.25))

    // Another dense layer, with a single unit for output
    model.add(tf.layers.dense({units: arraydata.yShape[1], activation: 'linear'}))

    // Compile the model, with the specified loss function (determines how well stuff gets predicted) and the previously defined optimizer
    model.compile({loss: tf.losses.meanSquaredError, optimizer: optimizer});

    // Generate actual tensors out of arraydata xs and ys
    const xs = tf.tensor2d(arraydata.xs, arraydata.xShape)
    const ys = tf.tensor2d(arraydata.ys, arraydata.yShape)

    // Train
    await model.fit(xs, ys, { epochs: 100 })
    let version = await readFile('./version.tag')
    await model.save('file://model/bikes-' + version)
    await writeFile('./version.tag', parseInt(version) + 1, 'utf8')

    // Infer
    let prediction_index = 1
    console.log("Check this, prediction index " + prediction_index + ":")
    const predict_this = tf.tensor2d(arraydata.xTest[prediction_index], [1, arraydata.xShape[1]])
    const output = model.predict(predict_this)
    const values = output.dataSync()
    console.log(values)
    console.log(arraydata.yTest[prediction_index])
}

dothemodel()
