import { ModelProxy, GeneratorProxy } from './model-proxy.js'

export class RemoteModel extends ModelProxy {
  constructor() {
    super()
  }
  
  get RestCallTask() {
    return class RestCallTask extends AsyncTask {
      #controller
      
      constructor() {
        super()
      }
      
      async execute([endpoint]) {
        this.#controller = new AbortController()
        const response = fetch(endpoint, {
          signal: this.#controller.signal
        })
        return response.json()
      }
      
      async cancel() {
        this.#controller.abort()
      }
    }
  }
}

class RemoteGenerator extends GeneratorProxy {
  constructor(sourceDomains, targetDomains) {
    super()
    this.sourceDomains = sourceDomains
    this.targetDomains = targetDomains
  }

  async generate(input) {

  }
}
