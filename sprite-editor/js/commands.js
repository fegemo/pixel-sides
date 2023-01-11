import { lineEFLA, equalPosition, floodFill } from './algorithms.js'

export class Command {
  constructor(name, params, taintsCanvas = true) {
    this.name = name
    this.params = params
    this.taintsCanvas = taintsCanvas
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

export class EraserCommand extends PencilCommand {
  constructor(color, erasedPositions) {
    super(color, erasedPositions)
    this.name = 'eraser'
  }

  configure(editor) {
    super.configure(editor)
    editor.canvas.ctx.fillStyle = this.params.color
    editor.canvas.ctx.globalCompositeOperation = 'destination-out'
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

class TwoPointPolygonCommand extends Command {
  constructor(color, startPosition, endPosition) {
    super('generic two line', { color, startPosition, endPosition })
  }

  configure(editor) {
    editor.canvas.ctx.save()
    editor.canvas.ctx.fillStyle = this.params.color
  }

  deconfigure(editor) {
    editor.canvas.ctx.restore()
  }

  execute(editor) {

  }

  updatePosition(position) {
    this.params.endPosition = position
  }
}

export class LineCommand extends TwoPointPolygonCommand {
  constructor(color, start, end) {
    super(color, start, end)
    this.name = 'line'
  }

  execute(editor) {
    this.configure(editor)
    const startPosition = this.params.startPosition
    const endPosition = this.params.endPosition
    lineEFLA((x, y) => editor.canvas.ctx.fillRect(x, y, 1, 1), startPosition, endPosition)
    this.deconfigure(editor)
  }
}

export class RectangleCommand extends TwoPointPolygonCommand {
  constructor(color, start, end) {
    super(color, start, end)
    this.name = 'rectangle'
  }

  execute(editor) {
    this.configure(editor)
    const startPosition = this.params.startPosition
    const endPosition = this.params.endPosition
    const ctx = editor.canvas.ctx
    ctx.fillRect(startPosition.x, startPosition.y, endPosition.x - startPosition.x, endPosition.y - startPosition.y)
    this.deconfigure(editor)
  }
}

export class BucketCommand extends Command {
  constructor(color, position) {
    super('bucket', { color, position })
  }

  configure(editor) {
    editor.canvas.ctx.save()
    editor.canvas.ctx.fillStyle = this.params.color
  }

  deconfigure(editor) {
    editor.canvas.ctx.restore()
  }

  execute(editor) {
    const ctx = editor.canvas.ctx
    this.configure(editor)
    floodFill(
      ctx.getImageData(0, 0, editor.canvas.width, editor.canvas.height),
      this.params.position,
      ({x, y}) => ctx.fillRect(x, y, 1, 1)
    )
    this.deconfigure(editor)
  }
}
