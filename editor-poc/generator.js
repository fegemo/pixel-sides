import * as tf from 'https://cdnjs.cloudflare.com/ajax/libs/tensorflow/3.13.0/tf.fesm.js'

const models = {}


async function generate(source, target, model) {
    const resultingImage = tf.tidy(() => {
        const offset = tf.scalar(127.5)
        const sourceData = tf.cast(tf.browser.fromPixels(source, 4), 'float32')
        const normalizedSourceData = sourceData.div(offset).sub(tf.scalar(1))
        const batchedSourceData = normalizedSourceData.reshape([1, 64, 64, 4])
    
        const t0 = tf.util.now();
        const targetData = model.apply(batchedSourceData, { 'training': true })
        const ellapsed = tf.util.now() - t0;
        console.info(`Took ${ellapsed.toFixed(2)}ms to predict`)
        
        const targetDataNormalized = targetData.div(2).add(0.5)
        return targetDataNormalized.reshape([64, 64, 4])
    })
    await tf.browser.toPixels(resultingImage, target)
    console.info('Finished drawing result to canvas')
    tf.dispose(resultingImage)
}

// This is not needed as per the Pix2Pix tutorial from tensorflow,
// we need to provide training=True during inference so BatchNorm
// does not use the aggregate statistics it gathered during training 
//
// function fixNaNWeights(model) {
//     const layer = model.getLayer('sequential_5').getLayer('batch_normalization_4')
//     const weights = layer.getWeights()
//     const fixedMovingAverageWeights = tf.variable(tf.onesLike(weights[3]))
//     layer.setWeights([...weights.slice(0, 3), fixedMovingAverageWeights])
// }

function noop() {
    // doesn't do anything
}

export default {
    async initialize(onProgress = noop) {
        models.f2r = await tf.loadLayersModel('models/front2right/model.json', { onProgress });
        models.f2b = await tf.loadLayersModel('models/front2back/model.json', { onProgress });
        models.f2l = await tf.loadLayersModel('models/front2left/model.json', { onProgress });
        // fixNaNWeights(models.f2r)
        // models.f2r.summary()

        // let min = +Infinity
        // let max = -Infinity
        // let minI = -1
        // let maxI = -1

        // const ws = models.f2r.getWeights()
        // for (let i = 0; i < ws.length; i++) {
        //     const weights = ws[i].dataSync()

        //     for (let j = 0; j < weights.length; j++) {
        //         if (weights[j] < min) {
        //             min = weights[j]
        //             minI = i
        //         }
        //         if (weights[j] > max) {
        //             max = weights[j]
        //             maxI = i
        //         }
        //     }
        // }
        // console.log('min', min, ', index', minI)
        // console.log('max', max, ', index', maxI)

        return {
            f2r: (source, target) => generate(source, target, models.f2r),
            f2b: (source, target) => generate(source, target, models.f2b),
            f2l: (source, target) => generate(source, target, models.f2l)
        }
    }
}