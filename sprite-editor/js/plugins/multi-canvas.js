import { Plugin } from './plugin.js'
import { Command } from '../commands.js'
import { Observable } from '../observable.js'

export class MultiCanvasPlugin extends Plugin {
  #editor
  #el
  #containerEl
  #activeCanvas = new Observable()
  static individualCanvasClass = 'other-canvas'
  static canvasContainerClass = 'js-canvas-container'

  constructor(canvasIds, labels, el, cssPath) {
    super('multi-canvas-plugin', [], cssPath)
    this.#containerEl = el
    this.canvases = []

    for (let c = 0; c < canvasIds.length; c++) {
      const id = canvasIds[c]
      const name = labels[c]
      const domain = id
      this.canvases.push(new MultiCanvasPlugin.Canvas(id, name, domain))
    }
  }

  #createDom(editor) {
    this.#containerEl.style.setProperty('--multi-canvas-length', this.canvases.length)

    this.#el = document.createRange().createContextualFragment(`<div id="multi-canvas-container"></div>`).firstElementChild
    this.#containerEl.appendChild(this.#el)

    for (let canvas of this.canvases) {
      const index = this.canvases.indexOf(canvas)
      const canvasEl = canvas.createDom(index, editor.canvasSize)
      this.#el.appendChild(canvasEl)
    }
  }

  install(editor) {
    this.#editor = editor

    // create/connect to ui
    this.#createDom(editor)

    // allow setting which canvas is active, and observes the model
    this.#containerEl.addEventListener('click', this.#canvasClicked.bind(this))
    this.#activeCanvas.addListener(this.setActiveCanvas.bind(this))

    // 2-way data binding for canvases:
    // binds canvas content to the main canvas: when the main canvas changes, update the active other canvas
    editor.addEventListener('canvaschange', this.updateActiveCanvas.bind(this))
    // when the active other canvas is changed, update the main canvas
    editor.addEventListener('canvaschange', this.updateMainCanvas.bind(this))


    // activates first canvas through a setup command and adds clear setup for correct ctrl+z/y
    const command = new MultiCanvasPlugin.CanvasChangeCommand(this, this.canvases[0])
    editor.addSetupCommand(command)
    editor.executeCommand(command)
    this.canvases.forEach(canvas => editor.addSetupCommand(new MultiCanvasPlugin.ClearCanvasCommand(canvas)))
  }

  requestAsideCanvas(id, prefix, domain) {
    const newCanvas = new MultiCanvasPlugin.Canvas(id, '', domain, prefix)
    this.canvases.push(newCanvas)
    return newCanvas
  }
  
  #canvasClicked(e) {
    let canvasEl = e.target.closest('.' + MultiCanvasPlugin.individualCanvasClass) || 
      e.target.closest('.' + MultiCanvasPlugin.canvasContainerClass)?.querySelector('.' + MultiCanvasPlugin.individualCanvasClass)
    if (!canvasEl) return
    if (canvasEl.classList.contains('ai-dragging')) return

    const canvas = this.canvases.find(canvas => canvas.elementId === canvasEl.id)

    // creates a command so it gets recorded to the history
    // and ctrl+z properly knows on which canvas to undo
    const command = new MultiCanvasPlugin.CanvasChangeCommand(this, canvas)
    this.#editor.executeCommand(command)
    this.#editor.recordCommand(command)
  }

  setActiveCanvas(canvas, previous) {
    // 1. visually deactivates the previous canvas and activates the new one
    if (previous) {
      previous.deactivate()
    }
    canvas.activate()
    
    // 2. set the main-canvas content as this other-canvas content
    this.#editor.canvas.clear()
    this.#editor.canvas.restore(canvas.el)
  }

  updateActiveCanvas(e) {
    // check if it's the main-canvas that has been changed
    if (e.detail.canvas === this.#editor.canvas) {
      // the main-canvas is tainted: let's update the active one
      this.#activeCanvas.get().updateContent(e.detail.canvas.el)
    }
  }
  
  updateMainCanvas(e) {
    // some other-canvas is tainted: if it's the active one, update the main-canvas
    const activeCanvas = this.#activeCanvas.get()
    if (activeCanvas === e.detail.canvas) {
      this.#editor.canvas.clear()
      this.#editor.canvas.restore(e.detail.canvas.el)
    }
  }

  get activeCanvas() {
    return this.#activeCanvas
  }

  get containerEl() {
    return this.#containerEl
  }

  static get Canvas() {
    return class Canvas {
      #id
      #name
      #domain
      #containerEl
      #canvasEl
      #ctx
      #prefix
      #canvasSize

      constructor(id, name, domain, prefix = 'multi-canvas') {
        this.#id = id
        this.#name = name
        this.#domain = domain
        this.#prefix = prefix
        this.updateSize = this.updateSize.bind(this)
      }

      createDom(index, canvasSize) {
        this.#canvasSize = canvasSize
        const size = canvasSize.get()
        const template = `
          <div class="${MultiCanvasPlugin.canvasContainerClass} ${this.#prefix}-container">
            ${this.name ? '<span class="canvas-label">' + this.name + '</span>' : ''}
            <canvas id="${this.elementId}" class="${this.#prefix}-${index + 1} other-canvas checkerboard-background" width="${size[0]}" height="${size[1]}"></canvas>
          </div>
          `
        this.#containerEl = document.createRange().createContextualFragment(template).firstElementChild
        this.#canvasEl = this.#containerEl.querySelector('canvas')
        this.#ctx = this.#canvasEl.getContext('2d')

        canvasSize.addListener(this.updateSize)

        return this.#containerEl
      }

      updateSize(size) {
        this.#canvasEl.width = size[0]
        this.#canvasEl.height = size[1]
      }

      clear() {
        this.#ctx.clearRect(0, 0, this.#canvasEl.width, this.#canvasEl.height)
      }

      updateContent(content) {
        this.clear()

        if (content instanceof HTMLCanvasElement) {
          this.#ctx.drawImage(content, 0, 0)
        } else {
          const { pixels, width, height } = content
          const imageData = new ImageData(pixels, width, height)
          this.#ctx.putImageData(imageData, 0, 0)
        }
      }

      activate() {
        this.#containerEl.classList.add('active-canvas')
      }

      deactivate() {
        this.#containerEl.classList.remove('active-canvas')
      }

      remove() {
        this.#canvasSize.addListener(this.updateSize)
        this.#containerEl.remove()
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

      get domain() {
        return this.#domain
      }

      get elementId() {
        return this.#prefix + '-' + this.id
      }
    }
  }

  static get CanvasChangeCommand() {
    return class CanvasChangeCommand extends Command {
      #multiCanvasPlugin

      constructor(multiCanvasPlugin, targetCanvas) {
        super('activate canvas', { targetCanvas }, false)
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
