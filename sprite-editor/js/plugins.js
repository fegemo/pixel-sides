import Observable from './observable.js'
import { Command } from './commands.js'

class Plugin {
  #cssPath
  #readyForInstallation = []

  constructor(name, dependencies = [], cssPath = null) {
    this.name = name
    this.dependencies = dependencies
    this.#cssPath = cssPath
  }
  
  preInstall(editor) {
    if (this.#cssPath) {
      this.loadCss()
    }
  }

  install(editor) {
    throw new Error('Abstract method "install" called on class Plugin.')
  }

  loadCss() {
    const linkEl = document.createElement('link')
    linkEl.rel = 'stylesheet'
    linkEl.href = this.#cssPath
    linkEl.dataset.addedBy = this.name

    const lastCssLinkEl = Array.from(document.head.querySelectorAll('link[rel="stylesheet"]')).at(-1)
    lastCssLinkEl.insertAdjacentElement("afterend", linkEl)
    
    const cssLoaded = new Promise((resolve, reject) => {
      linkEl.onload = () => resolve()
    })

    this.#readyForInstallation.push(cssLoaded)
  }

  get readyToInstall() {
    return Promise.all(this.#readyForInstallation)
  }
}

export class MultiCanvasPlugin extends Plugin {
  #editor
  #containerEl
  #activeCanvas = new Observable()

  constructor(canvasIds, labels, el, cssPath) {
    super('multi-canvas', [], cssPath)
    this.#containerEl = el
    this.canvases = []
    
    for (let c = 0; c < canvasIds.length; c++) {
      const id = canvasIds[c]
      const name = labels[c]
      this.canvases.push(new MultiCanvasPlugin.Canvas(id, name))
    }
  }

  #createDom(editor) {
    this.#containerEl.innerHTML = ''
    this.#containerEl.style.setProperty('--multi-canvas-length', this.canvases.length)

    for (let canvas of this.canvases) {
      const index = this.canvases.indexOf(canvas)
      const canvasEl = canvas.createDom(index, editor.canvasSize)
      this.#containerEl.appendChild(canvasEl)
    }
  }

  install(editor) {
    this.#editor = editor

    // create/connect to ui
    this.#createDom(editor)

    // allow setting which canvas is active, and observes the model
    this.#containerEl.addEventListener('click', this.#canvasClicked.bind(this))
    this.#activeCanvas.addListener(this.setActiveCanvas.bind(this))
    
    // binds canvas content to the main
    editor.addEventListener('canvaschange', this.updateActiveCanvas.bind(this))
    
    // activates first canvas through a setup command and adds clear setup for correct ctrl+z/y
    const command = new MultiCanvasPlugin.CanvasChangeCommand(this, this.canvases[0])
    editor.addSetupCommand(command)
    editor.executeCommand(command)
    this.canvases.forEach(canvas => editor.addSetupCommand(new MultiCanvasPlugin.ClearCanvasCommand(canvas)))

    // monkeypatches the editor so when it replays commands, it clears all canvases
    // (instead of only the main one)
    const officialReplayCommands = editor.replayCommands
    editor.replayCommands = () => {
      this.canvases.forEach(canvas => canvas.clear())
      officialReplayCommands.apply(editor)
    }
  }

  #canvasClicked(e) {
    const clickedContainerEl = e.target.closest('.multi-canvas-container')
    if (!clickedContainerEl) {
      return
    }
    const clickedCanvasId = clickedContainerEl.querySelector('canvas').id
    const clickedCanvas = this.canvases.find(canvas => clickedCanvasId.endsWith(canvas.id))

    // creates a command so it gets recorded to the history
    // and ctrl+z properly knows on which canvas to undo
    const command = new MultiCanvasPlugin.CanvasChangeCommand(this, clickedCanvas)
    this.#editor.executeCommand(command)
    this.#editor.recordCommand(command)
  }

  setActiveCanvas(canvas, previous) {    
    // 1. visualmente desativa o anterior e ativa o novo
    if (previous) {
      previous.deactivate()
    }
    canvas.activate()

    // 2. definir conteúdo do main-canvas como o deste
    this.#editor.canvas.clear()
    this.#editor.canvas.restore(canvas.el)
  }

  updateActiveCanvas(e) {
    this.#activeCanvas.get().updateContent(e.detail.canvas.el)
  }

  static get Canvas() {
    return class Canvas {
      #id
      #name
      #containerEl
      #canvasEl
      #ctx

      constructor(id, name) {
        this.#id = id
        this.#name = name
      }

      createDom(index, canvasSize) {
        const size = canvasSize.get()
        const template = `
          <div class="multi-canvas-container">
            <span class="canvas-label">${this.name}</span>
            <canvas id="multi-canvas-${this.id}" class="multi-canvas-${index + 1} other-canvas checkerboard-background" width="${size[0]}" height="${size[1]}"></canvas>
          </div>
        `
        this.#containerEl = document.createRange().createContextualFragment(template).firstElementChild
        this.#canvasEl = this.#containerEl.querySelector('canvas')
        this.#ctx = this.#canvasEl.getContext('2d')

        canvasSize.addListener(size => {
          this.#canvasEl.width = size[0]
          this.#canvasEl.height = size[1]
        })

        return this.#containerEl
      }

      clear() {
        this.#ctx.clearRect(0, 0, this.#canvasEl.width, this.#canvasEl.height)
      }

      updateContent(el) {
        this.clear()
        this.#ctx.drawImage(el, 0, 0)
      }
      
      activate() {
        this.#containerEl.classList.add('active-canvas')
      }
      
      deactivate() {
        this.#containerEl.classList.remove('active-canvas')
      }

      get el() {
        return this.#canvasEl
      }

      get name() {
        return this.#name
      }

      get id() {
        return this.#id
      }
    }
  }

  static get CanvasChangeCommand() {
    return class CanvasChangeCommand extends Command {
      #multiCanvasPlugin

      constructor(multiCanvasPlugin, targetCanvas) {
        super('activate canvas', { targetCanvas })
        this.#multiCanvasPlugin = multiCanvasPlugin
      }

      execute(editor) {
        const targetCanvas = this.params.targetCanvas
        this.#multiCanvasPlugin.#activeCanvas.set(targetCanvas)
      }
    }
  }

  static get ClearCanvasCommand() {
    return class ClearCanvasCommand extends Command {
      constructor(canvas) {
        super('clear canvas', { canvas })
      }

      execute(editor) {
        this.params.canvas.clear()
      }
    }
  }
}
