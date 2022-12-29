import { config, DOMAINS } from './config.js'
import { AsyncTask, dynamicImport, TaskAbortion } from '../task.js'
import { detensorize } from '../tf-util.js'
import { ModelProxy, GeneratorProxy } from './model-proxy.js'

export class LocalModel extends ModelProxy {
  #tfLoaded

  constructor(architecture) {
    super()
    this.architecture = architecture
    this.config = config[architecture]
  }

  async initialize() {
    // loads tensorflow
    this.#tfLoaded = dynamicImport('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.1.0/dist/tf.js', 'tf')
    this.tf = await this.#tfLoaded

    // loads all checkpoints from this architecture
    const checkpoints = this.config.checkpoints

    for (let domain of Object.keys(DOMAINS)) {
      this._generators[domain] = {}
    }

    const loadModelTasks = checkpoints.map(ckpt => new LocalModel.LoadModelTask(this.tf, ckpt))
    const promises = loadModelTasks.map(task => task.run())

    for (let promise of promises) {
      promise.then(({ model, ckpt }) => {
        const generator = new LocalGenerator(this, model, this.tf)
        this._generators[ckpt.source][ckpt.target] = generator
      }).catch(error => {
        if (error instanceof TaskAbortion) {
          console.info(`Silently cancelled model loading task #${promises.indexOf(promise)}`)
        }
      })
    }

    // warms up tensorflow by doing a single inference (doesnt matter which model is used)
    const warmUpPromise = Promise.race(promises).then(({model}) => model.predict(this.tf.zeros([1, 64, 64, 4])).dispose())
    this._loaded = Promise.all([this.#tfLoaded, ...promises, warmUpPromise])

    return loadModelTasks
  }

  selectGenerator(sourceDomains, targetDomains) {
    const isMultiSource = Array.isArray(sourceDomains) && sourceDomains.length > 1
    const isMultiTarget = Array.isArray(targetDomains) && targetDomains.length > 1

    if (isMultiTarget) {
      throw new Error('Not yet implemented "selectGenerator" with multi target on LocalModel')
    }

    if (isMultiSource) {
      throw new Error('Not yet implemented "selectGenerator" with multi source (colla) on LocalModel')
    } else {
      const source = Array.isArray(sourceDomains) ? sourceDomains[0] : sourceDomains
      const fromGenerator = this._generators[source] ?? this._generators[DOMAINS.any]
      return fromGenerator[targetDomains]
    }
  }

  static get LoadModelTask() {
    return class LoadModelTask extends AsyncTask {
      #tf
      #checkpoint
      #shouldCache

      constructor(tf, checkpoint, cache = true) {
        super()
        this.#tf = tf
        this.#checkpoint = checkpoint
        this.#shouldCache = cache
      }

      _execute(signal) {
        const tf = this.#tf
        const file = this.#checkpoint.file

        const onProgress = value => this._progress.set(value)

        const promise = new Promise((resolve, reject) => {
          signal.addEventListener('abort', () => { reject(new TaskAbortion(this)) }, { once: true })

          // try to load from cache
          tf.loadLayersModel(`indexeddb://${file}`)
            .then(model => (this._progress.set(1), model))

            // in case it fails, it's because the model was not cached
            // so we must load from the url  
            .catch(() => {
              const downloadPromise = tf.loadLayersModel(file, { onProgress })
              if (this.#shouldCache) {
                downloadPromise.then(model => model.save(`indexeddb://${file}`))
              }
              return downloadPromise
            })
            .then(model => resolve({ model, ckpt: this.#checkpoint }))
            .catch(reject)
        })

        return promise
      }

      get targetDomain() {
        return this.#checkpoint.target
      }
    }
  }

  static get GenerateLocallyTask() {
    return class GenerateLocallyTask extends AsyncTask {
      #generator
      #tf

      constructor(model, generator, tf, backend = 'webgl', thread = 'ui', waitOn = []) {
        super([model.loaded, ...waitOn])
        this.#generator = generator
        this.#tf = tf
      }

      async _execute(signal, sourceCanvasEl) {
        const generator = this.#generator
        const tf = this.#tf
        const generatedImage = tf.tidy(() => {
          const offset = tf.scalar(127.5)
          const sourceData = tf.cast(tf.browser.fromPixels(sourceCanvasEl, 4), 'float32')
          const normalizedSourceData = sourceData.div(offset).sub(tf.scalar(1))
          const batchedSourceData = normalizedSourceData.reshape([1, 64, 64, 4])

          const t0 = tf.util.now();
          const targetData = generator.apply(batchedSourceData, { training: true })
          const ellapsed = tf.util.now() - t0;
          console.info(`Took ${ellapsed.toFixed(2)}ms to predict`)

          const targetDataNormalized = targetData.div(2).add(0.5)
          return targetDataNormalized.reshape([64, 64, 4])
        })

        const pixels = await detensorize(generatedImage)
        generatedImage.dispose()
        this.progress.set(1)

        return pixels
      }

      async cancel() {
        throw new Error('Not yet implemented "cancel" method on GenerateLocallyTask')
      }
    }
  }
}

class LocalGenerator extends GeneratorProxy {
  #localModel
  #tfModel
  #tf

  constructor(localModel, tfModel, tf) {
    super()
    this.#localModel = localModel
    this.#tfModel = tfModel
    this.#tf = tf
  }

  createGenerationTask() {
    return new LocalModel.GenerateLocallyTask(this.#localModel, this.#tfModel, this.#tf)
  }
}
