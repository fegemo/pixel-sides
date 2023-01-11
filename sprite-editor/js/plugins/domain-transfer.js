import { Plugin } from './plugin.js'
import { selectModel } from '../generators/model.js'
import { DOMAINS } from '../generators/config.js'
import { Observable, ComputedProgressObservable } from '../observable.js'
import { range } from '../py-util.js'
import { Command } from '../commands.js'
// import 'https://cdn.interactjs.io/v1.10.16/auto-start/index.js'
// import 'https://cdn.interactjs.io/v1.10.16/actions/drag/index.js'
// import 'https://cdn.interactjs.io/v1.10.16/actions/resize/index.js'
// import 'https://cdn.interactjs.io/v1.10.16/modifiers/index.js'
// import 'https://cdn.interactjs.io/v1.10.16/dev-tools/index.js'
import interact from 'https://cdn.interactjs.io/v1.10.14/interactjs/index.js'
// import interact from 'https://cdn.interactjs.io/v1.9.20/interactjs/index.js'
// TODO?: extract common behavior to an abstract DomainTransferPlugin
// and create SingleInputDomainTransferPlugin and MultiInputDomain...
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

    // allow dragging multi-canvas views to the suggestions to ask for image generation:
    // dragged view: source domain // which dropzone means the target domain
    this.#containerEl.querySelectorAll('.multi-canvas-container').forEach(el => {
      const dropzoneEl = document.createElement('div')
      dropzoneEl.classList.add('ai-dropzone')
      dropzoneEl.dataset.dropActionLabel = 'Copy image here'
      el.appendChild(dropzoneEl)
    })
    interact('.multi-canvas-container .other-canvas')
      .draggable({
        manualStart: true,
        listeners: {
          move(e) {
            const draggedEl = e.target
            const x = (parseFloat(e.target.dataset.x) || 0) + e.dx
            const y = (parseFloat(e.target.dataset.y) || 0) + e.dy

            draggedEl.style.translate = `${x}px ${y}px`
            draggedEl.dataset.x = x
            draggedEl.dataset.y = y
          },
          end(e) {
            const draggedEl = e.target
            const droppedOnDropzone = !!e.dropzone
            draggedEl.addEventListener('transitionend', () => draggedEl.remove(), { once: true })

            if (droppedOnDropzone) {
              draggedEl.classList.add('ai-dropped-on-dropzone')
              setImmediate(() => {
                draggedEl.style.scale = 0
                draggedEl.style.opacity = 0
              })
            } else {
              draggedEl.classList.add('ai-dropped-out-of-zone')
              setImmediate(() => {
                draggedEl.style.scale = 1
                draggedEl.style.translate = 0
                draggedEl.style.opacity = 1
                draggedEl.style.left = 0
                draggedEl.style.top = 0
              })
            }
          }
        }
      })
      .on('move', (e) => {
        if (e.interaction.pointerIsDown && !e.interaction.interacting()) {
          const original = e.currentTarget
          const clone = original.cloneNode(true)
          clone.classList.add('ai-dragging')
          clone.style.width = original.offsetWidth + 'px'
          clone.style.height = original.offsetHeight + 'px'
          let shiftX = e.offsetX
          let shiftY = e.offsetY
          clone.style.left = (-original.offsetWidth / 2 + shiftX)+ 'px'
          clone.style.top = (-original.offsetHeight / 2 + shiftY)+ 'px'

          clone.style.transformOrigin = 'center center'
          if (original instanceof HTMLCanvasElement) {
            const ctx = clone.getContext('2d')
            ctx.drawImage(original, 0, 0)
          }
          setImmediate(() => clone.style.scale = 0.5)

          original.closest('.multi-canvas-container').appendChild(clone)
          e.interaction.start({ name: 'drag' }, e.interactable, clone)
        }
      })
    interact('.multi-canvas-container .ai-dropzone')
      .dropzone({
        accept: '.ai-preview-multi-container .other-canvas',
        ondrop: (e) => {
          const target = this.#canvases.find(canvas => canvas.elementId === e.target.closest('.multi-canvas-container').querySelector('.other-canvas').id)
          const sourceCanvasEl = e.relatedTarget
          const loadImageCommand = new LoadImageCommand(sourceCanvasEl, target)
          
          editor.executeCommand(loadImageCommand)
          editor.recordCommand(loadImageCommand)
          
          e.target.classList.remove('ai-droppable')
          e.target.classList.remove('ai-dragged-hover')
        },
        ondropactivate(e) {
          e.target.classList.add('ai-droppable')
        },
        ondropdeactivate(e) {
          e.target.classList.remove('ai-droppable')
        },
        ondragenter(e) {
          e.target.classList.add('ai-dragged-hover')
        },
        ondragleave(e) {
          e.target.classList.remove('ai-dragged-hover')
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
    const sourceDomain = source.id
    const targetDomain = target.id
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

        // set up each suggestion area as a dropzone
        // the dropzone has a target domain and the dragged element represents the source
        const dropzoneEl = document.createElement('div')
        dropzoneEl.classList.add('ai-dropzone')
        this.#multiPreviewEl.appendChild(dropzoneEl)
        interact(dropzoneEl).dropzone({
          accept: '.multi-canvas-container .other-canvas',
          ondrop: (e) => {
            const source = plugin.#canvases.find(canvas => canvas.elementId === e.relatedTarget.id)
            const target = this.canvas
            plugin.generate(source, target, this)

            e.target.classList.remove('ai-droppable')
            e.target.classList.remove('ai-dragged-hover')
          },
          ondropactivate: (e) => {
            e.target.classList.add('ai-droppable')

            const source = plugin.#canvases.find(canvas => canvas.elementId === e.relatedTarget.id)
            const target = this.canvas
            e.target.dataset.dropActionLabel = `Generate ${target.name} from ${source.name}`
          },
          ondropdeactivate(e) {
            e.target.classList.remove('ai-droppable')
          },
          ondragenter(e) {
            e.target.classList.add('ai-dragged-hover')
          },
          ondragleave(e) {
            e.target.classList.remove('ai-dragged-hover')
          }
        })

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
            this.suggestionCanvases.push(multiCanvasPlugin.requestAsideCanvas(`${this.domain}-suggestion-${c + 1}`, 'ai-preview'))
          }
          this.suggestionCanvases.forEach((sc, c) => {
            this.#multiPreviewEl.insertBefore(
              sc.createDom(c, this.#editor.canvasSize),
              this.#multiPreviewEl.querySelector('button.ai-generate')
            )
          })

          // TODO?: set up suggestions as draggables (here? cant we delegate?)
          for (let canvas of this.suggestionCanvases) {
            interact(canvas.el).draggable({
              accept: '.ai-preview-container .other-canvas',
              manualStart: true,
              listeners: {
                move(e) {
                  const x = (parseFloat(e.target.dataset.x) || 0) + e.dx
                  const y = (parseFloat(e.target.dataset.y) || 0) + e.dy

                  const draggedEl = e.target
                  draggedEl.style.translate = `${x}px ${y}px`
                  draggedEl.dataset.x = x
                  draggedEl.dataset.y = y
                },
                end(e) {
                  const draggedEl = e.target
                  const droppedOnDropzone = !!e.dropzone
                  draggedEl.addEventListener('transitionend', () => draggedEl.remove(), { once: true })

                  if (droppedOnDropzone) {
                    draggedEl.classList.add('ai-dropped-on-dropzone')
                    setImmediate(() => {
                      draggedEl.style.scale = 0
                      draggedEl.style.opacity = 0
                    })
                  } else {
                    draggedEl.classList.add('ai-dropped-out-of-zone')
                    setImmediate(() => {
                      draggedEl.style.scale = 1
                      draggedEl.style.translate = 0
                      draggedEl.style.opacity = 1
                    })
                  }
                }
              }
            })
            .on('move', (e) => {
              if (e.interaction.pointerIsDown && !e.interaction.interacting()) {
                const original = e.currentTarget
                const clone = original.cloneNode(true)
                clone.classList.add('ai-dragging')
                clone.style.width = original.offsetWidth + 'px'
                clone.style.height = original.offsetHeight + 'px'
                clone.style.left = 0
                clone.style.top = 0
                clone.style.transformOrigin = 'center center'
                if (original instanceof HTMLCanvasElement) {
                  const ctx = clone.getContext('2d')
                  ctx.drawImage(original, 0, 0)
                }
                setImmediate(() => clone.style.scale = 0.5)

                original.closest('.js-canvas-container').appendChild(clone)
                e.interaction.start({ name: 'drag' }, e.interactable, clone)
              }                  
            })
          }

        } else if (toCreate < 0) {
          const indexFromWhichToRemove = this.suggestionCanvases.length + toCreate
          const deleted = this.suggestionCanvases.splice(indexFromWhichToRemove, -1 * toCreate)
          // TODO: remove interact.js handlers of the suggestion elements being removed
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
