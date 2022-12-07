export default class Observable {
  #value
  
  constructor(value) {
    this.#value = value
    this.listeners = []
  }
  
  get() {
    return this.#value
  }
  
  set(updated) {
    if (updated === this.#value) {
      return
    }
    
    const previousValue = this.#value
    this.#value = updated
    for (let listener of this.listeners) {
      listener(updated, previousValue)
    }
  }
  
  addListener(listener) {
    this.listeners.push(listener)
  }
  
  removeListener(listener) {
    const index = this.listeners.indexOf(listener)
    if (index >= 0) {
      this.listeners.splice(index, 1)
    }
  }
}