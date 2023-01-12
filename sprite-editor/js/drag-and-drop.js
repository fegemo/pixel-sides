import interact from 'https://cdn.interactjs.io/v1.10.14/interactjs/index.js'
import { callOrReturn } from './functional-util.js'

class DragDropObject {
  constructor(element, options = {}) {
    this.draggableObject = interact(element)
      .draggable({
        manualStart: true,

        listeners: {
          start(e) {
            e.interactable.data = callOrReturn(options, 'draggedData', null, [e])
          },

          // sets the object x,y when the mouse moves
          move(e) {
            const draggedEl = e.target
            const x = (parseFloat(draggedEl.dataset.x) || 0) + e.dx
            const y = (parseFloat(draggedEl.dataset.y) || 0) + e.dy

            draggedEl.style.translate = `${x}px ${y}px`
            draggedEl.dataset.x = x
            draggedEl.dataset.y = y
          },

          // when dragging ends, we either send the dragged
          // element back to its original location or make it vanish
          // and in either case, the object is consumed (removed from DOM)
          end(e) {
            const draggedEl = e.target
            const droppedOnDropzone = !!e.dropzone
            const dropRejected = e.rejected

            draggedEl.classList.add('ai-dropped')
            draggedEl.addEventListener('transitionend', () => draggedEl.remove(), { once: true })
            
            const dragResultClass = droppedOnDropzone && !dropRejected ? 'ai-dropped-vanish' : 'ai-dropped-return'
            setImmediate(() => draggedEl.classList.add(dragResultClass))
          }
        }
      })

    // we want to drag a clone of the object, so we need to intercept
    // the 'move' event, clone the initially dragged element and
    // then manually start the interaction process,
    // per  https://interactjs.io/docs/faq/#clone-target-draggable
    this.draggableObject
      .on('move', (e) => {
        const justStartedDragging = e.interaction.pointerIsDown && !e.interaction.interacting()
        if (justStartedDragging) {
          // create and configure the cloned element
          const original = e.currentTarget
          const clone = original.cloneNode(true)
          original.insertAdjacentElement('afterend', clone)

          clone.classList.add('ai-dragging')
          clone.style.width = original.offsetWidth + 'px'
          clone.style.height = original.offsetHeight + 'px'
          clone.style.transformOrigin = 'center center'
          // puts the center of the clone at the mouse pointer
          clone.style.left = (-original.offsetWidth / 2 + e.offsetX) + 'px'
          clone.style.top = (-original.offsetHeight / 2 + e.offsetY) + 'px'
          
          setImmediate(() => clone.style.scale = 0.5)

          if (original instanceof HTMLCanvasElement) {
            // we cloned the canvas, but we need to redraw at the cloned
            const ctx = clone.getContext('2d')
            ctx.drawImage(original, 0, 0)
          }

          // start the interact.js drag action, but with the clone
          e.interaction.start({ name: 'drag' }, e.interactable, clone)
        }
      })
  }

  dropOn(selector, callback, options = {}) {
    // add a dropzone element
    const targets = document.querySelectorAll(selector)
    targets.forEach(el => {
      const dropzone = document.createElement('div')
      dropzone.classList.add('ai-dropzone')
      el.appendChild(dropzone)
    })

    // configure dropzone using interact.js
    interact(`${selector} .ai-dropzone`)
      .dropzone({
        accept: options?.accept,
        ondrop: (e) => {
          const draggedData = e.draggable.data
          const droppedData = callOrReturn(options, 'droppedData', null, [e])
          
          // calls the function registered for the DROP action
          // and allows it to reject the action
          e.rejectAction = () => e.dragEvent.rejected = true
          callback(e, draggedData, droppedData)
          
          e.target.classList.remove('ai-droppable')
          e.target.classList.remove('ai-dragged-hover')
        },
        ondropactivate: (e) => {
          const draggedData = e.draggable?.data
          const droppedData = callOrReturn(options, 'droppedData', null, [e])
          
          const actionLabel = callOrReturn(options, 'actionDescription', '', [e, draggedData, droppedData])
          e.target.dataset.dropActionLabel = actionLabel
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
        },
      })

    // allow chaining multiple .dropOn
    return this
  }
}

export function allowDrag(element, options) {
  return new DragDropObject(element, options)
}
