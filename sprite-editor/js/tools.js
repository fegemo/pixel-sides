import { PencilCommand } from "./commands.js"

class Tool {
  constructor(name, exclusionGroup, triggerElements, shortcut) {
    this.name = name
    this.exclusionGroup = exclusionGroup
    this.active = false
    this.els = triggerElements
    // this.shortcut = shortcut

    triggerElements.forEach(el => el.addEventListener('click', this.activate.bind(this)))
    if (shortcut) {
      document.addEventListener('keyup', e => {
        if (e.key.toLowerCase() === shortcut.toLowerCase()) {
          this.activate()
        }
      })
    }
  }

  attachToEditor(editor) {
    this.editor = editor
    if (!this.editor.tools) {
      this.editor.tools = []
    }
    if (!this.editor.tools.includes(this)) {
      this.editor.tools.append(this)
    }
  }

  activate() {
    if (!this.editor) {
      throw new Error(`Activated a tool (${this.name}) which was not attached to any editor`)
    }

    // 1. deactivate previously activated from the same exclusion group
    const activeToolsFromSameGroup = this.editor.tools.filter(t => t.exclusionGroup === this.exclusionGroup && t.active)
    activeToolsFromSameGroup.forEach(t => t.deactivate())

    // 2. call the underlying activation specific to the tool
    this.active = true
    this.els.forEach(el => el.classList.add('active-tool'))
    this.activated()
  }

  deactivate() {
    if (!this.editor) {
      throw new Error(`Deactivated a tool (${this.name}) which was not attached to any editor`)
    }
    this.active = false
    this.els.forEach(el => el.classList.remove('active-tool'))
    this.deactivated()
  }

  activated() {
    throw new Error('Called abstract method "activated" of the Tool')
  }

  deactivated() {
    throw new Error('Called abstract method "deactivated" of the Tool')
  }

}

export class Pencil extends Tool {
  constructor(elements) {
    super('Pencil', 'regular-tools', elements, 'P')
    this.draw = this.draw.bind(this)
  }

  draw(e) {
    switch (e.type) {
      case 'mousedown':
        if (this.activelyDrawing) {
          return
        }
        this.savedCanvas = this.editor.canvas.save()
        const color = e.button == 0 ? this.editor.primaryColor : this.editor.secondaryColor
        this.command = new PencilCommand(color, [this.editor.mousePosition])
        
        this.activelyDrawing = true
        break

      case 'mouseup':
        if (this.activelyDrawing) {
          this.editor.canvas.restore(this.savedCanvas)
          this.command.execute(this.editor)
          this.editor.recordCommand(this.command)
  
          this.activelyDrawing = false
        }
        break

      case 'mousemove':
        if (this.activelyDrawing) {
          const position = this.editor.mousePosition

          this.command.logPosition(position)
          this.command.iterate(this.editor, position)
        }
        break
    }
  }

  activated() {
    ['mousedown', 'mousemove', 'mouseup'].forEach(type =>
      this.editor.canvas.el.addEventListener(type, this.draw)
    )
  }

  deactivated() {
    ['mousedown', 'mousemove', 'mouseup'].forEach(type =>
      this.editor.canvas.el.removeEventListener(type, this.draw)
    )
  }
}
export class Pen extends Tool {
  constructor(elements) {
    super('Pen', 'regular-tools', elements, 'E')
    this.draw = this.draw.bind(this)
    this.mousedownPosition = null
    this.positionsToPaint = null
  }

  draw(e) {
    switch (e.type) {
      case 'mousedown':
        this.savedCanvas = this.editor.canvas.ctx.getImageData(0, 0, this.editor.canvas.width, this.editor.canvas.height)
        this.mousedownPosition = this.editor.mousePosition
        this.positionsToPaint = [this.mousedownPosition]
        break

      case 'mouseup':
        this.editor.canvas.ctx.putImageData(this.savedCanvas, 0, 0)
        const color = e.button == 0 ? this.editor.primaryColor : this.editor.secondaryColor
        const command = new PenCommand(color, this.positionsToPaint)
        command.execute(this.editor)

        this.mousedownPosition = null
        this.positionsToPaint = null
        break

      case 'mousemove':
        if (Array.isArray(this.positionsToPaint)) {
          const { x, y } = this.editor.mousePosition
          const { x: lastX, y: lastY } = this.positionsToPaint.at(-1)
          if (x !== lastX || y !== lastY) {
            this.positionsToPaint.push({ x, y })
          }
        }

        if (this.mousedownPosition) {
          const { x, y } = this.editor.mousePosition
          this.editor.canvas.ctx.fillStyle = this.editor.primaryColor
          this.editor.canvas.ctx.fillRect(x, y, 1, 1);
        }
        break
    }
  }

  activated() {
    this.editor.canvas.el.addEventListener('mousedown', this.draw)
    this.editor.canvas.el.addEventListener('mousemove', this.draw)
    this.editor.canvas.el.addEventListener('mouseup', this.draw)
  }

  deactivated() {
    this.editor.canvas.el.removeEventListener('mousedown', this.draw)
    this.editor.canvas.el.removeEventListener('mousemove', this.draw)
    this.editor.canvas.el.removeEventListener('mouseup', this.draw)
  }

  get command() {

  }
}

export class Bucket extends Tool {
  constructor(elements) {
    super('Bucket', 'regular-tools', elements, 'B')
  }

  activated() {

  }

  deactivated() {

  }
}

export class Ellipsis extends Tool {
  constructor(elements) {
    super('Bucket', 'regular-tools', elements, 'E')
  }

  activated() {

  }

  deactivated() {

  }
}

export class ColorPicker extends Tool {
  constructor(name, elements, defaultColor) {
    super(name, 'color-picker', elements)
    this.inputs = Array.from(elements).map(el => el.querySelector('input[type="color"]')).filter(el => !!el)
    this.bindColor = this.bindColor.bind(this)

    const primaryOrSecondary = ['primary', 'secondary'].find(type => name.toLowerCase().includes(type))
    this.specialColorSlot = primaryOrSecondary // can also be undefined
    this.defaultColor = defaultColor
  }

  activated() {
    this.inputs.forEach(el => el.addEventListener('input', this.bindColor))
  }

  deactivated() {
    this.inputs.forEach(el => el.removeEventListener('input', this.bindColor))
  }

  attachToEditor(editor) {
    super.attachToEditor(editor)

    this.activated()
    this.inputs[0].value = this.defaultColor
    this.inputs[0].dispatchEvent(new Event('input'));
    this.deactivated()
  }

  bindColor(e) {
    // gets the color selected by the user
    const chosenColor = e.currentTarget.value

    // sets the swatch bg color accordingly
    this.inputs.forEach(el => {
      el.closest('.swatch').style.backgroundColor = chosenColor
      if (el !== e.currentTarget) {
        el.value = chosenColor
      }
    })

    // tells the editor a primary or secondary color was selected
    if (this.specialColorSlot) {
      this.editor[this.specialColorSlot + 'Color'] = chosenColor
    }
  }
}
