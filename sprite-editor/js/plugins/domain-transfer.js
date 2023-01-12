import { Plugin } from './plugin.js'
import { selectModel } from '../generators/model.js'
import { DOMAINS } from '../generators/config.js'
import { Observable, ComputedProgressObservable } from '../observable.js'
import { range } from '../functional-util.js'
import { Command } from '../commands.js'

// TODO?: extract common behavior to an abstract DomainTransferPlugin
// and create SingleInputDomainTransferPlugin and MultiInputDomain...
import { allowDrag } from '../drag-and-drop.js'

export class DomainTransferPlugin extends Plugin {
  #canvasIds
  #editor
  #previewEl
  #containerEl
  #views
  #canvases
  #numberOfSuggestions
  #aiArchitecture

  constructor(aiArchitecture, canvasIds, el, numberOfSuggestions, cssPath) {
    super(`domain-transfer-${aiArchitecture}-plugin`, ['multi-canvas-plugin'], cssPath)
    this.#aiArchitecture = aiArchitecture
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
    this.#views = this.#canvases.map(canvas => new DomainTransferPlugin.View(canvas, new DomainTransferPlugin.ProgressBar(canvas.name, 0), this.#previewEl, this.#numberOfSuggestions))
    this.#views.forEach(view => view.createDom(editor, this))

    const draggable = allowDrag('.other-canvas', {
      draggedData: (e) => {
        const draggedEl = e.target
        const view = this.#views.find(view => view.containsElement(draggedEl))
        const canvas = view.canvasFrom(draggedEl)

        return {
          source: canvas,
          sourceDomain: view.domain,
          sourceView: view
        }
      }
    })

    // copy the canvas onto the target dropzone
    draggable.dropOn('.multi-canvas-container', (e, draggedData, droppedData) => {
      const { source } = draggedData
      const { target } = droppedData
      
      if (source === target) {
        e.rejectAction()
        return
      }

      const loadImageCommand = new LoadImageCommand(source.el, target)

      editor.executeCommand(loadImageCommand)
      editor.recordCommand(loadImageCommand)
    }, {
      accept: '.other-canvas',
      actionDescription: 'Copy image here',
      droppedData: (e) => {
        const canvasId = e.target.closest('.js-canvas-container')?.querySelector('.other-canvas')?.id
        const view = this.#views.find(view => view.canvas.elementId === canvasId)

        return {
          target: view.canvas,
          targetDomain: view.domain,
          targetView: view
        }
      }
    })

    // generate images from a source domain (dragged other-canvas) in a target (dropzone)
    draggable.dropOn('.ai-preview-multi-container', (e, draggedData, droppedData) => {
      const { source } = draggedData
      const { target, targetView } = droppedData

      this.generate(source, target, targetView)
    }, {
      accept: '.other-canvas',
      actionDescription: (e, draggedData, droppedData) => {
        const { source } = draggedData
        const { target } = droppedData
        return `Generate ${target.name} from ${source.name}`
      },
      droppedData: (e) => {
        const aiPreviewDropzone = e.target
        const view = this.#views.find(view => view.containsElement(aiPreviewDropzone))
        return {
          target: view.canvas,
          targetDomain: view.domain,
          targetView: view
        }
      }
    })
  }

  async install(editor) {
    this.#editor = editor
    // connect to the canvases of the multi-canvas-plugin
    // TODO: fazer algo sobre "conectar" os canvasIds recebidos? Apenas uma validação? Nem isso?
    this.#canvases = editor.plugins['multi-canvas-plugin'].canvases
    this.#createDom(editor)
    this.#numberOfSuggestions.addListener(this.updateNumberOfSuggestions.bind(this))
    this.updateNumberOfSuggestions(this.#numberOfSuggestions.get())

    this.model = selectModel(this.#aiArchitecture)
    console.time('initialize-model')
    const tasks = await this.model.initialize()

    // Bind tasks to progress bars:
    //   each task that has target as a domain will contribute
    //   to the loading progress of the view of that domain
    //   ...in case target==='any', all any tasks contribute to all
    //   views
    for (let view of this.#views) {
      const tasksForDomain = tasks.filter(t => t.targetDomain === view.domain || t.targetDomain === DOMAINS.any)
      view.progressBar.watchProgress(tasksForDomain.map(t => t.progress))
    }
    await Promise.allSettled(tasks.map(t => t.done))
    console.timeEnd('initialize-model')
  }

  generator(view, editor) {
    return async () => {
      const activeCanvas = editor.plugins['multi-canvas-plugin'].activeCanvas.get()

      const sourceDomain = activeCanvas.id
      const targetDomain = view.canvas.id
      const sourceCanvasEl = activeCanvas.el

      const numberOfSuggestions = this.#numberOfSuggestions.get()
      const generator = this.model.selectGenerator(sourceDomain, targetDomain)
      const tasks = Array.from(range(numberOfSuggestions)).map(() => generator.createGenerationTask(sourceDomain, targetDomain))
      view.progressBar.watchProgress(tasks.map(t => t.progress))

      for (let s of range(numberOfSuggestions)) {
        const targetCanvas = view.suggestionCanvases[s]
        const generatedImage = await tasks[s].run(sourceCanvasEl)
        targetCanvas.updateContent(generatedImage)
      }
    }
  }

  async generate(source, target, targetView) {
    const sourceDomain = source.domain
    const targetDomain = target.domain
    console.log(`Generating ${targetDomain} from ${sourceDomain}`)

    const numberOfSuggestions = this.#numberOfSuggestions.get()
    const generator = this.model.selectGenerator(sourceDomain, targetDomain)
    const tasks = Array.from(range(numberOfSuggestions)).map(() => generator.createGenerationTask(sourceDomain, targetDomain))
    targetView.progressBar.watchProgress(tasks.map(t => t.progress))

    for (let s of range(numberOfSuggestions)) {
      const targetCanvas = targetView.suggestionCanvases[s]
      const generatedImage = await tasks[s].run(source.el)

      const loadImageCommand = new LoadImageCommand(generatedImage, targetCanvas)
      this.#editor.executeCommand(loadImageCommand)
      // SHOULD NOT record the command, so it does not enter the ctrl+z/y stacks
      // editor.recordCommand(loadImageCommand)
    }
  }

  static get View() {
    // a View represents a domain, has a multi-canvas-plugin canvas
    // and one or many suggestion canvases
    // it creates DOM for the suggestion canvases
    return class View {
      #numberOfSuggestions
      #editor
      #containerEl
      #multiPreviewEl
      progressBar

      constructor(canvas, progressBar, containerEl, numberOfSuggestions) {
        this.canvas = canvas
        this.progressBar = progressBar
        this.#containerEl = containerEl
        this.#numberOfSuggestions = numberOfSuggestions
        this.suggestionCanvases = []
      }

      createDom(editor, plugin) {
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
            const multiCanvasPlugin = this.#editor.plugins['multi-canvas-plugin']
            this.suggestionCanvases.push(multiCanvasPlugin.requestAsideCanvas(`${this.domain}-suggestion-${c + 1}`, 'ai-preview', this.domain))
          }
          this.suggestionCanvases.forEach((sc, c) => {
            this.#multiPreviewEl.insertBefore(
              sc.createDom(c, this.#editor.canvasSize),
              this.#multiPreviewEl.querySelector('button.ai-generate')
            )
          })

        } else if (toCreate < 0) {
          const indexFromWhichToRemove = this.suggestionCanvases.length + toCreate
          const deleted = this.suggestionCanvases.splice(indexFromWhichToRemove, -1 * toCreate)

          deleted.forEach(del => del.remove())
        }
      }

      canvasFrom(element) {
        // returns the canvas which the element represents
        const viewCanvases = [this.canvas, ...this.suggestionCanvases]
        for (let canvas of viewCanvases) {
          if (canvas.el.closest('.js-canvas-container')?.contains(element)) {
            return canvas
          }
        }
      }

      containsElement(element) {
        // returns whether the view contains some element
        let contains = false
        contains ||= this.canvas.el === element
        contains ||= !!this.suggestionCanvases.find(sc => sc.el === element)
        contains ||= this.#multiPreviewEl.contains(element)
        contains ||= this.canvas.el.closest('.js-canvas-container').contains(element)
        return contains
      }

      get domain() {
        return this.canvas.id
      }
    }
  }

  static get ProgressBar() {
    return class ProgressBar {
      #name
      #value
      #el
      #computedProgress

      constructor(name, value = 0) {
        this.#name = name
        this.#value = value
      }

      createDom() {
        const template = `<div class="ai-progress-bar vertical transition" aria-role="progressbar" aria-label="Progress of ${this.#name}"></div>`
        this.#el = document.createRange().createContextualFragment(template).firstElementChild
        return this.#el
      }

      watchProgress(observableProgresses) {
        // reset the progress value, as we're starting the task
        this.value = 0

        // a callback to update and stop watching when finished
        const updateAndStop = (value => {
          this.value = value
          if (this.value >= 1) {
            this.#computedProgress.removeListener(updateAndStop)
          }
        })

        // start watching
        this.#computedProgress = new ComputedProgressObservable(observableProgresses)
        this.#computedProgress.addListener(updateAndStop)
      }

      get value() {
        return this.#value
      }

      set value(percentage) {
        this.#value = percentage
        this.#el.classList.toggle('transition', percentage !== 0)
        setImmediate(() => this.#el.style.setProperty('--progress-bar-value', Math.min(1, Math.max(0, this.#value))))
      }
    }
  }
}

class LoadImageCommand extends Command {
  constructor(image, canvas) {
    super('load-image', { image, canvas }, false)
  }

  execute(editor) {
    const { image, canvas } = this.params
    canvas.updateContent(image)
    editor.dispatchEvent(new CustomEvent('canvaschange', { detail: { canvas } }))
  }
}
