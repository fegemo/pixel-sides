import Canvas from './canvas.js'
import { Pencil, Bucket, Eraser, EyeDropper, ColorPicker } from './tools.js'
import generators from './generators.js'
import Observable from './observable.js'

class Editor extends EventTarget {
  #zoom
  #mousePosition
  #primaryColor = new Observable()
  #secondaryColor = new Observable()
  #executedCommands = []
  #undoneCommands = []
  
  constructor(containerEl, canvasEl, tools, canvasSize) {
    super()
    this.containerEl = containerEl
    this.canvas = new Canvas(canvasEl, this, canvasSize)
    this.tools = tools
    this.canvasSize = canvasSize
    this.#executedCommands = []
    this.#undoneCommands = []

    tools.forEach(t => t.attachToEditor(this))
    this.mouseStatsElements = {
      containerEl: containerEl.querySelector('#mouse-stats'),
      xEl: containerEl.querySelector('#mouse-stats #x-mouse'),
      yEl: containerEl.querySelector('#mouse-stats #y-mouse'),
      zoomEl: containerEl.querySelector('#mouse-stats #zoom')
    }
    
    this.zoom = 1

    this.containerEl.ownerDocument.defaultView.addEventListener('keydown', this.keyboardMultiplexer.bind(this))
  }
  
  updateMouseStats(info) {
    if (info.x !== undefined) {
      this.mouseStatsElements.xEl.innerHTML = info.x
    }
    if (info.y !== undefined) {
      this.mouseStatsElements.yEl.innerHTML = info.y
    }
    if (info.zoom !== undefined) {
      this.mouseStatsElements.zoomEl.innerHTML = info.zoom
    }
  }

  recordCommand(command) {
    this.#executedCommands.push(command)
    this.#undoneCommands.length = 0
  }

  undo() {
    const undone = this.#executedCommands.pop()
    if (undone) {
      this.#undoneCommands.push(undone)
      this.replayCommands()
    }
  }

  redo() {
    const redone = this.#undoneCommands.pop()
    if (redone) {
      this.#executedCommands.push(redone)
      this.replayCommands()
    }
  }

  replayCommands() {
    this.canvas.clear()
    for (let command of this.#executedCommands) {
      command.execute(this)
    }
  }
  
  keyboardMultiplexer(e) {
    // Ctrl+z
    if (e.type === 'keydown' && e.ctrlKey && e.key === 'z') {
      e.preventDefault()
      this.undo()
    }
    // Ctrl+y
    else if (e.type === 'keydown' && e.ctrlKey && e.key === 'y') {
      e.preventDefault()
      this.redo()
    }
  }

  getActiveTool(toolGroup) {
    let toolsToConsider = this.tools || []
    if (toolGroup) {
      toolsToConsider = toolsToConsider.filter(tool => tool.exclusionGroup === toolGroup)
    }
    return toolsToConsider.find(tool => tool.active)
  }

  get zoom() {
    return this.#zoom
  }
  
  set zoom(value) {
    this.#zoom = value
    this.canvas.el.style.transform = `scale(${value})`
    this.updateMouseStats({ zoom: value.toFixed(2) })
  }
  
  get mousePosition() {
    return this.#mousePosition
  }
  
  set mousePosition(pos) {
    this.#mousePosition = pos
    this.updateMouseStats(pos)
  }
  
  get primaryColor() {
    return this.#primaryColor
  }
  
  set primaryColor(value) {
    this.#primaryColor.set(value)
  }
  
  get primaryColorAsInt() {
    return Editor.hexToRGB(this.#primaryColor.get())
  }
  
  get secondaryColor() {
    return this.#secondaryColor
  }
  
  set secondaryColor(value) {
    this.#secondaryColor.set(value)
  }

  get secondaryColorAsInt() {
    return Editor.hexToRGB(this.#secondaryColor.get())
  }

  static hexToRGB(hex) {
    hex = hex.replace(/^#?([a-f\d])([a-f\d])([a-f\d])$/i, (m, r, g, b) => '#' + r + r + g + g + b + b)
    return hex.substring(1).match(/.{2}/g)
      .map(x => parseInt(x, 16))
  }
}

const editor = new Editor(
  document.querySelector('#main-canvas-section'),
  document.querySelector('#main-canvas'),
  [
    new Pencil(document.querySelectorAll('#pencil-tool')),
    new Bucket(document.querySelectorAll('#bucket-tool')),
    new Eraser(document.querySelectorAll('#eraser-tool')),
    new EyeDropper(document.querySelectorAll('#eye-dropper-tool')),
    new ColorPicker('Primary Color', document.querySelectorAll('#primary-color'), '#7890e8'),
    new ColorPicker('Secondary Color', document.querySelectorAll('#secondary-color'), '#ffffff'),
  ],
  [64, 64]
  )
  
  