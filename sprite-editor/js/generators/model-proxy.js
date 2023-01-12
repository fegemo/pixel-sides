export class ModelProxy {
  _loaded
  _generators = {}
  
  constructor() {
  }
  
  async initialize() {
    throw new Error('Abstract "initialize" method called in "ModelProxy"')
  }
  
  selectGenerator() {
    throw new Error('Abstract "selectGenerator" method called in "ModelProxy"')
  }
  
  get loaded() {
    return this._loaded
  }
}

export class GeneratorProxy {
  constructor() {
  }

  createGenerationTask() {
    throw new Error('Abstract "createGenerationTask" method called in "GeneratorProxy"')
  }
}
