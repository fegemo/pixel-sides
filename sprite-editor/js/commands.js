import { lineEFLA, equalPosition } from './algorithms.js'

export class Command {
  constructor(name, params) {
    this.name = name
    this.params = params
  }

  execute(editor) {
    throw new Error('Abstract "execute" method on Command was called.')
  }
}

export class PencilCommand extends Command {
  constructor(color, paintedPositions) {
    super('pencil', { color, paintedPositions })
    this.previousPosition = this.params.paintedPositions[0]
  }

  configure(editor) {
    editor.canvas.ctx.save()
    editor.canvas.ctx.fillStyle = this.params.color
  }
  
  deconfigure(editor) {
    editor.canvas.ctx.restore()
  }

  execute(editor) {
    this.configure(editor)
    this.previousPosition = this.params.paintedPositions[0]
    for (let position of this.params.paintedPositions) {
      this.#iterateConfigured(editor, position)
    }
    this.deconfigure(editor)
  }

  #iterateConfigured(editor, position) {
    if (equalPosition(position, this.previousPosition)) {
      editor.canvas.ctx.fillRect(position.x, position.y, 1, 1)
    } else {
      lineEFLA((x, y) => editor.canvas.ctx.fillRect(x, y, 1, 1), this.previousPosition, position)
    }
    this.previousPosition = position
  }

  iterate(editor, position) {
    this.configure(editor)
    this.#iterateConfigured(editor, position)    
    this.deconfigure(editor)
  }

  logPosition(position) {
    if (equalPosition(this.previousPosition, position)) {
      return
    }
    this.params.paintedPositions.push(position)
  }
}

export class PenCommand extends Command {
  constructor(color, paintedPositions) {
    super('pencil', { color, paintedPositions })
  }

  execute(editor) {
    const ctx = editor.canvas.ctx
    ctx.save()
    ctx.fillStyle = this.params.color
    ctx.lineJoin = 'miter'
    ctx.miterLimit = 1
    ctx.lineCap = 'butt'
    ctx.lineWidth = 1
    ctx.imageSmoothingEnabled = false
    ctx.beginPath()
    const { x, y } = this.params.paintedPositions[0]
    ctx.moveTo(x, y)
    ctx.stroke()
    for (const {x, y} of this.params.paintedPositions) {
      this.iterate(editor, {x, y})
    }
    ctx.closePath()
    ctx.restore()
    console.log("Finished drawing cold")
  }

  iterate(editor, position) {
    editor.canvas.ctx.lineTo(position.x, position.y)
    editor.canvas.ctx.stroke()
  }
}

export class LineCommand extends Command {
  constructor(color, startPosition, endPosition) {
    super('line', { color, startPosition, endPosition })
  }

  execute(editor) {
    editor.canvas.ctx.save()

    editor.canvas.ctx.strokeStyle = this.params.color

    const { x: startX, y: startY } = this.params.startPosition
    editor.canvas.ctx.moveTo(startX, startY)

    const { x: targetX, y: targetY } = this.params.endPosition
    editor.canvas.ctx.lineTo(targetX, targetY)
    editor.canvas.ctx.restore()
  }
}
