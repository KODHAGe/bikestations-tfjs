// TF
const tf = require('@tensorflow/tfjs')
require('@tensorflow/tfjs-node')

// File handling
const { promisify } = require('util')
const fs = require('fs')
const readFile = promisify(fs.readFile)
const writeFile = promisify(fs.writeFile)

const http = require('http')
const { PORT = 6666 } = process.env

const readdir = promisify(fs.readdir)

const dataDir = './data'

function readDataBatch(index) {
    let data = fs.readFileSync(dataDir + '/stream-' + index + '.json', 'utf-8')
    return data
}

http.createServer((req, res) => {
    if(req.method === 'POST') {
        let body = ''
        req.on('data', function(data) {
            body += data
        })
        req.on('end', function() {
            let post = JSON.parse(body)
            let epochs = post.epochs
            console.log("Run training with " + epochs + " epochs")
            dothemodel(epochs)
        })
    } else {
        res.end('Hello World from Node.js\n')
    }
}).listen(PORT)


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
    yTest = ys.splice(0, ys.length * 0.01)
    xTest = xs.splice(0, xs.length * 0.01)

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

async function dothemodel(epochs) {

    /*
    *  tfjs provides a "Keras" API, which is easier to comprehend than pure tensorflow.
    *  Let's use that. (https://js.tensorflow.org/tutorials/tfjs-layers-for-keras-users.html)
    */

    // Sequential model = everything is applied in sequence
    console.log('create model')
    const model = tf.sequential()

    // Lets define an optimizer, this pokes around the weights to help unstuck the training process
    const optimizer = tf.train.adamax(0.1);

    // Add some layers
    // A dense (fully connected) one, with 72 units
    model.add(tf.layers.dense({units: 72, inputShape: [264]}))

    // leakyReLU activation function. This determines when the neurons are activated, this one is supposedly cool.
    model.add(tf.layers.leakyReLU())

    // Dropout = disconnect some neurons to make them learn better
    model.add(tf.layers.dropout(0.25))

    // Another dense layer, with a single unit for output
    model.add(tf.layers.dense({units: 1, activation: 'linear'}))

    // Compile the model, with the specified loss function (determines how well stuff gets predicted) and the previously defined optimizer
    model.compile({loss: tf.losses.meanSquaredError, optimizer: optimizer});

    // Generate actual tensors out of arraydata xs and ys

    // Train
    console.log('train')
    await train(model, epochs)
    // let version = await readFile('./version.tag')
    await model.save('file://model/bikes-latest')
    //await writeFile('./version.tag', parseInt(version) + 1, 'utf8')
}

async function batchData() {
    try {
        const batches = await readdir(dataDir)
        if(batches.indexOf('.DS_Store') > -1) {
            batches.splice(batches.indexOf('.DS_Store'), 1)
        }
        let batchIds = batches.map((entry) => {
            entry = entry.replace('.json', '')
            entry = entry.replace('stream-', '')
            return entry
        })
        return {
            'length': batches.length,
            'ids': batchIds
        }
    } catch(e) {
        console.log(e)
    }

}

async function train(model, epochs) {
    const batches = await batchData()
    for(let i = 0; i < batches.length; i++) {
        let data = readDataBatch(batches.ids[i])

        // Parse ndjson
        const jsonRows = data.split(/\n|\n\r/);
        let json = []
        jsonRows.forEach((row) => {
            try {
                json.push(JSON.parse(row))
            } catch(e) {
                return false
            }
        })
        let batch = dataToTensor(json)

        if(i === 0) {
            // Store keys to file, once
            let keysJSON = JSON.stringify(batch.keys)
            fs.writeFile('model/keys.json', keysJSON, 'utf8', function(err, result) {
                if(err) console.log('error', err)
            })
        }

        // Make tensors out of batches
        const xs = tf.tensor2d(batch.xs, batch.xShape)
        const ys = tf.tensor2d(batch.ys, batch.yShape)

        // Fit em to the model
        await model.fit(xs, ys, { epochs: epochs })

        // Do inference
        let prediction_index = 1
        console.log("Check this, prediction index " + prediction_index + ":")
        batch.yTest.forEach((predictor, index) => {
            const predict_this = tf.tensor2d(batch.xTest[index], [1, 264])
            const output = model.predict(predict_this)
            const values = output.dataSync()
            console.log(values)
            console.log(predictor)
            console.log('---------------')
        })
    }
}

//dothemodel(1)