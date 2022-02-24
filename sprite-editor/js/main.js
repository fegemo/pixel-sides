import Canvas from './canvas.js'
import { Pencil, Bucket, Ellipsis, ColorPicker } from './tools.js'
import commands from './commands.js'
import generators from './generators.js'

class Editor {
  #zoom
  #mousePosition
  #primaryColor
  #secondaryColor

  constructor(containerEl, canvasEl, tools, canvasSize) {
    this.containerEl = containerEl
    this.canvas = new Canvas(canvasEl, this, canvasSize)
    this.tools = tools
    this.canvasSize = canvasSize
    
    tools.forEach(t => t.attachToEditor(this))
    this.mouseStatsElements = {
      containerEl: containerEl.querySelector('#mouse-stats'),
      xEl: containerEl.querySelector('#mouse-stats #x-mouse'),
      yEl: containerEl.querySelector('#mouse-stats #y-mouse'),
      zoomEl: containerEl.querySelector('#mouse-stats #zoom')
    }

    this.zoom = 1
  }

  updateMouseStats(info) {
    if (info.x) {
      this.mouseStatsElements.xEl.innerHTML = info.x
    }
    if (info.y) {
      this.mouseStatsElements.yEl.innerHTML = info.y
    }
    if (info.zoom) {
      this.mouseStatsElements.zoomEl.innerHTML = info.zoom
    }
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
    this.#primaryColor = value
  }

  get secondaryColor() {
    return this.#secondaryColor
  }

  set secondaryColor(value) {
    this.#secondaryColor = value
  }
}

const editor = new Editor(
  document.querySelector('#main-canvas-section'),
  document.querySelector('#main-canvas'),
  [
    new Pencil(document.querySelectorAll('#pencil-tool')),
    new Bucket(document.querySelectorAll('#bucket-tool')),
    new Ellipsis(document.querySelectorAll('#ellipsis-tool')),
    new ColorPicker('Primary Color', document.querySelectorAll('#primary-color'), '#7890e8'),
    new ColorPicker('Secondary Color', document.querySelectorAll('#secondary-color'), '#ffffff'),
  ],
  [64, 64]
)

