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
  #el
  #containerEl
  #activeCanvas = new Observable()

  constructor(canvasIds, labels, el, cssPath) {
    super('multi-canvas-plugin', [], cssPath)
    this.#containerEl = el
    this.canvases = []
    
    for (let c = 0; c < canvasIds.length; c++) {
      const id = canvasIds[c]
      const name = labels[c]
      this.canvases.push(new MultiCanvasPlugin.Canvas(id, name))
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
    
    // binds canvas content to the main
    editor.addEventListener('canvaschange', this.updateActiveCanvas.bind(this))
    
    // activates first canvas through a setup command and adds clear setup for correct ctrl+z/y
    const command = new MultiCanvasPlugin.CanvasChangeCommand(this, this.canvases[0])
    editor.addSetupCommand(command)
    editor.executeCommand(command)
    this.canvases.forEach(canvas => editor.addSetupCommand(new MultiCanvasPlugin.ClearCanvasCommand(canvas)))
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
      #prefix
      #canvasSize

      constructor(id, name, prefix='multi-canvas') {
        this.#id = id
        this.#name = name
        this.#prefix = prefix
        this.updateSize = this.updateSize.bind(this)
      }

      createDom(index, canvasSize) {
        this.#canvasSize = canvasSize
        const size = canvasSize.get()
        const template = `
          <div class="${this.#prefix}-container">
            ${this.name ? '<span class="canvas-label">' + this.name + '</span>' : ''}
            <canvas id="${this.#prefix}-${this.id}" class="${this.#prefix}-${index + 1} other-canvas checkerboard-background" width="${size[0]}" height="${size[1]}"></canvas>
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

export class Pix2PixPlugin extends Plugin {
  #canvasIds
  #previewEl
  #containerEl
  #views
  #canvases
  #numberOfSuggestions

  constructor(canvasIds, el, numberOfSuggestions, cssPath) {
    super('pix2pix-plugin', ['multi-canvas-plugin'], cssPath)
    this.#canvasIds = canvasIds
    this.#containerEl = el
    this.#numberOfSuggestions = new Observable(numberOfSuggestions)
  }

  updateNumberOfSuggestions(number) {
    this.#previewEl.style.setProperty('--ai-number-of-suggestions', number)
  }

  #createDom(editor) {
    // creates the ai colateral bar (to the left of the multi-canvas)
    this.#previewEl = document.createRange().createContextualFragment(`<div id="ai-section"></div>`).firstElementChild
    this.#containerEl.appendChild(this.#previewEl)

    // creates views for each multi-canvas canvas
    // a view represents a domain (e.g., front, right, left or back), its elements (canvas etc.), and the
    // suggestion(s) from the model
    this.#views = this.#canvases.map(canvas => new Pix2PixPlugin.View(canvas, new Pix2PixPlugin.ProgressBar(canvas.name, 0), this.#previewEl, this.#numberOfSuggestions))
    this.#views.forEach(view => view.createDom(editor))
  }

  install(editor) {
    // connect to the canvases of the multi-canvas-plugin
    // TODO: fazer algo sobre "conectar" os canvasIds recebidos? Apenas uma validação? Nem isso?
    this.#canvases = editor.plugins['multi-canvas-plugin'].canvases
    this.#createDom(editor)
    this.#numberOfSuggestions.addListener(this.updateNumberOfSuggestions.bind(this))
    this.updateNumberOfSuggestions(this.#numberOfSuggestions.get())
  }

  static get View() {
    return class View {
      #numberOfSuggestions
      #editor
      #containerEl
      #multiPreviewEl

      constructor(canvas, progressBar, containerEl, numberOfSuggestions) {
        this.canvas = canvas
        this.progressBar = progressBar
        this.#containerEl = containerEl
        this.#numberOfSuggestions = numberOfSuggestions
        this.suggestionCanvases = []
      }
      
      createDom(editor) {
        this.#editor = editor

        const template = `<div class="ai-preview-multi-container"></div>`
        this.#multiPreviewEl = document.createRange().createContextualFragment(template).firstElementChild
        this.#containerEl.appendChild(this.#multiPreviewEl)

        // creates 1 canvas per suggestion
        this.#numberOfSuggestions.addListener(this.#createSuggestionCanvases.bind(this))
        this.#createSuggestionCanvases(this.#numberOfSuggestions.get(), 0)
        
        // progress bar
        const pgbEl = this.progressBar.createDom()
        this.canvas.el.insertAdjacentElement('afterend', pgbEl)
      }
      
      #createSuggestionCanvases(number, previous) {
        const toCreate = number - previous
        
        if (toCreate > 0) {
          for (let c = 0; c < toCreate; c++) {
            this.suggestionCanvases.push(new MultiCanvasPlugin.Canvas('suggestion-' + c, null, 'ai-preview'))
          }
          this.suggestionCanvases.forEach((sc, c) => {
            this.#multiPreviewEl.appendChild(
              sc.createDom(c, this.#editor.canvasSize)
            )
          })
        } else if (toCreate < 0) {
          const indexFromWhichToRemove = this.suggestionCanvases.length + toCreate
          const deleted = this.suggestionCanvases.splice(indexFromWhichToRemove, -1*toCreate)
          deleted.forEach(del => del.remove())
        }
      }
    }
  }

  static get ProgressBar() {
    return class ProgressBar {
      #name
      #value
      #el

      constructor(name, value = 0) {
        this.#name = name
        this.#value = value
      }

      createDom() {
        const template = `<div class="ai-progress-bar vertical" aria-role="progressbar" aria-label="Progress of ${this.#name}"></div>`
        this.#el = document.createRange().createContextualFragment(template).firstElementChild
        return this.#el
      }

      get value() {
        return this.#value
      }

      set value(percentage) {
        this.#value = percentage
        this.#el.style.setProperty('--progress-bar-value', Math.min(1, Math.max(0, this.#value)))
      }
    }
  }
}
