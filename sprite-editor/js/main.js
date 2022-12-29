import Canvas from './canvas.js'
import { Pencil, Bucket, Eraser, Line, Rectangle, EyeDropper, ColorPicker } from './tools.js'
import { MultiCanvasPlugin, DomainTransferPlugin } from './plugins.js'
import { Observable } from './observable.js'

class Editor extends EventTarget {
  #zoom
  #mousePosition
  #primaryColor = new Observable()
  #secondaryColor = new Observable()
  #canvasSize = new Observable()
  #executedCommands = []
  #undoneCommands = []
  #setupCommands = []
  plugins = {}
  
  constructor(containerEl, canvasEl, tools, canvasSize) {
    super()
    this.containerEl = containerEl
    this.#canvasSize.set(canvasSize)
    this.tools = tools
    this.canvas = new Canvas(canvasEl, this, this.#canvasSize)
    this.#setupCommands = []
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

  executeCommand(command) {
    command.execute(this)
    this.#notifyCanvasChange()
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
    this.#setupCommands.forEach(this.executeCommand.bind(this))
    this.#executedCommands.forEach(this.executeCommand.bind(this))
  }

  addSetupCommand(command) {
    this.#setupCommands.push(command)
  }

  #notifyCanvasChange() {
    this.dispatchEvent(
      new CustomEvent('canvaschange', { detail: { canvas: this.canvas }})
      )
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

  register(plugin) {
    const { name } = plugin
    this.plugins[name] = plugin
  }

  async install() {
    const pluginsInOrder = []

    // check for plugin dependencies, adding them to a list in a proper order to be installed
    for (let name of Object.keys(this.plugins)) {
      const plugin = this.plugins[name]
      let deps = plugin.dependencies
      if (deps && !Array.isArray(deps)) {
        deps = [deps]
      }
      
      for (let dependencyName of deps) {
        if (!(dependencyName in this.plugins)) {
          throw new Error(`Plugin ${plugin.name} required ${dependencyName}, but it was not registered before installation`)
        }
      }

      // inserts at the left, but it'll move to the right until all dependencies are to the left
      pluginsInOrder.unshift(plugin)
    }

    // preinstall hook - can do it as all dependencies have been satisfied
    for (let name of Object.keys(this.plugins)) {
      this.plugins[name].preInstall(this)
    }

    // puts plugins in correct order for the install hook
    for (let name of Object.keys(this.plugins)) {
      // move to the right until all dependencies are to the left
      const plugin = this.plugins[name]
      const index = pluginsInOrder.indexOf(plugin)
      let lastDependencyIndex = index
      for (let dependencyName of plugin.dependencies) {
        const dependencyIndex = pluginsInOrder.indexOf(this.plugins[dependencyName])
        lastDependencyIndex = Math.max(lastDependencyIndex, dependencyIndex)
      }

      pluginsInOrder.splice(index, 1)
      pluginsInOrder.splice(lastDependencyIndex, 0, plugin)
      // TODO: in case of cycles this fails... should do topological sorting
    }

    for (let plugin of pluginsInOrder) {
      await plugin.readyToInstall
      await plugin.install(this)
    }
  }

  get canvasSize() {
    return this.#canvasSize
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
    new Line(document.querySelectorAll('#line-tool')),
    new Rectangle(document.querySelectorAll('#rectangle-tool')),
    new EyeDropper(document.querySelectorAll('#eye-dropper-tool')),
    new ColorPicker('Primary Color', document.querySelectorAll('#primary-color'), '#7890e8'),
    new ColorPicker('Secondary Color', document.querySelectorAll('#secondary-color'), '#ffffff'),
  ],
  [64, 64]
  )
  
editor.register(new MultiCanvasPlugin(
  ['front', 'right', 'back', 'left'],
  ['front', 'right', 'back', 'left'],
  document.querySelector('#colateral-section'),
  'multi-canvas.css'))
editor.register(new DomainTransferPlugin(
  'pix2pix',
  ['front', 'right', 'back', 'left'],
  document.querySelector('#colateral-section'),
  4,
  'ai.css'))
editor.install()
