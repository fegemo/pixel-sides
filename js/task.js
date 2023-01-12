import { Observable } from './observable.js'

export class AsyncTask {
  _progress
  _done
  _abortController
  #waitOn
  
  constructor(waitOn = []) {
    this._progress = new Observable(0)
    this._abortController = new AbortController()
    this.#waitOn = waitOn
  }
  
  async run(paramList) {
    await Promise.allSettled(this.#waitOn)

    const { signal } = this._abortController
        
    try {
      this._done = this._execute(signal, paramList)
      await this._done

    } catch (error) {
      if (error instanceof TaskAbortion) {
        // do not do anything and return a resolved promise
        this._done = Promise.resolve()
        console.info(`Task of type ${this.constructor.name} was aborted by the user...`)
      } else {
        console.error(error)
        this._done = Promise.reject(error)
      }
    }

    return this._done
  }

  async _execute(signal, paramList) {
    throw new Error('Abstract "execute" method called on AsyncTask.')
  }

  async cancel() {
    this._abortController.abort()
  }

  observeProgress(callback) {
    const watchThenRemove = (value) => {
      callback(value)
      if (value >= 1) {
        this._progress.removeListener(watchThenRemove)
      }
    }
    this._progress.addListener(watchThenRemove)
  }

  get progress() {
    return this._progress
  }

  get done() {
    return this._done
  }
}

export function dynamicImport(path, exportName) {
  dynamicImport.cache = dynamicImport.cache ?? {}
  
  if (dynamicImport.cache[path]) {
    return Promise.resolve(window[exportName])
  }

  const fileDownloaded = (resolve) => {
    const module = window[exportName]
    dynamicImport.cache[path] = module
    resolve(module)
  }

  const scriptEl = document.createElement('script')
  scriptEl.src = path
  scriptEl.dataset.importedBy = 'dynamic import'
  document.head.appendChild(scriptEl)

  // TODO: deal with errors when downloading the file
  return new Promise((resolve, reject) => scriptEl.onload = () => fileDownloaded(resolve))
}

export class TaskAbortion extends Error {
  constructor(task) {
    super(`A task of type ${task?.constructor?.name} was cancelled`)
  }
}
