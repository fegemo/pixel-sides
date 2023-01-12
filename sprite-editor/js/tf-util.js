// Simpler reimplementation of tf.browser.toPixels but without receiving a canvas,
// rather returning a byte array, a width and height
// Based on: https://github.com/tensorflow/tfjs/blob/tfjs-v4.1.0/tfjs-core/src/ops/browser.ts#L292-L372
export async function detensorize(tensor) {
  const data = await tensor.data()
  const [height, width, channels] = tensor.shape
  const multiplier = tensor.dtype === 'float32' ? 255 : 1
  const pixels = new Uint8ClampedArray(height * width * 4)
  
  for (let v = 0; v < height * width * channels; v += 4) {
    pixels[v+0] = Math.round(data[v+0] * multiplier)
    pixels[v+1] = Math.round(data[v+1] * multiplier)
    pixels[v+2] = Math.round(data[v+2] * multiplier)
    pixels[v+3] = Math.round(data[v+3] * multiplier)
  }

  return { pixels, width, height }
}
