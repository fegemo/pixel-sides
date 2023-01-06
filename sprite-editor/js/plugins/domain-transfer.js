import { Plugin } from './plugin.js'
import { selectModel } from '../generators/model.js'
import { DOMAINS } from '../generators/config.js'
import { Observable, ComputedProgressObservable } from '../observable.js'
import { range } from '../py-util.js'

// TODO?: extract common behavior to an abstract DomainTransferPlugin
// and create SingleInputDomainTransferPlugin and MultiInputDomain...
export class DomainTransferPlugin extends Plugin {
  #canvasIds
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
    this.#views.forEach(view => view.createDom(editor))

    // TODO: (temporary?) "generate" button
    this.#views.forEach(view => view.generateButton.addEventListener('click', this.generator(view, editor).bind(this)))
  }

  async install(editor) {
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

  static get View() {
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

      createDom(editor) {
        this.#editor = editor

        // TODO: (temporary?) "generate" button
        const template = `
          <div class="ai-preview-multi-container">
            <button class="ai-generate">Generate</button>
          </div>`
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
            this.suggestionCanvases.push(this.#editor.plugins['multi-canvas-plugin'].requestAsideCanvas(`${this.domain}-suggestion-${c+1}`, 'ai-preview'))
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

      get generateButton() {
        return this.#multiPreviewEl.querySelector('button.ai-generate')
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
        // reset
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
