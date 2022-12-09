import { BucketCommand, EraserCommand, LineCommand, RectangleCommand, PencilCommand } from "./commands.js"

class Tool {
  #shouldDisableMenu

  constructor(name, exclusionGroup, triggerElements, shortcut, shouldDisableMenu = true) {
    this.name = name
    this.exclusionGroup = exclusionGroup
    this.active = false
    this.els = triggerElements
    this.shortcut = shortcut
    this.#shouldDisableMenu = shouldDisableMenu
  }

  attachToEditor(editor) {
    this.editor = editor
    if (!this.editor.tools) {
      this.editor.tools = []
    }
    if (!this.editor.tools.includes(this)) {
      this.editor.tools.append(this)
    }

    this.els.forEach(el => el.addEventListener('click', this.activate.bind(this)))
    if (this.shortcut) {
      this.editor.containerEl.ownerDocument.addEventListener('keyup', e => {
        if (e.key.toLowerCase() === this.shortcut.toLowerCase()) {
          this.activate()
        }
      })
    }
  }

  activate() {
    if (!this.editor) {
      throw new Error(`Activated a tool (${this.name}) which was not attached to any editor`)
    }

    // call preActivation hook
    this.preActivation()

    // 1. deactivate previously activated from the same exclusion group
    const activeToolsFromSameGroup = this.editor.tools.filter(t => t.exclusionGroup === this.exclusionGroup && t.active)
    activeToolsFromSameGroup.forEach(t => t.deactivate())

    // 2. call the underlying activation specific to the tool
    this.active = true
    this.els.forEach(el => el.classList.add('active-tool'))

    if (this.#shouldDisableMenu) {
      this.editor.canvas.el.addEventListener('contextmenu', this.#disableContextMenu)
    }

    this.activated()
  }

  deactivate() {
    if (!this.editor) {
      throw new Error(`Deactivated a tool (${this.name}) which was not attached to any editor`)
    }
    if (this.#shouldDisableMenu) {
      this.editor.canvas.el.removeEventListener('contextmenu', this.#disableContextMenu)
    }
    this.active = false
    this.els.forEach(el => el.classList.remove('active-tool'))
    this.deactivated()
  }

  preActivation() {
    // allow tools to override
  }

  activated() {
    throw new Error('Called abstract method "activated" of the Tool')
  }

  deactivated() {
    throw new Error('Called abstract method "deactivated" of the Tool')
  }

  #disableContextMenu(e) {
    e.preventDefault()
    return false
  }
}

export class Pencil extends Tool {
  constructor(elements) {
    super('Pencil', 'regular-tools', elements, 'P')
    this.draw = this.draw.bind(this)
  }

  draw(e) {
    // consider only left/right buttons
    if (e.button !== 0 && e.button !== 2) {
      return
    }

    switch (e.type) {
      case 'mousedown':
        if (this.activelyDrawing) {
          return
        }
        this.savedCanvas = this.editor.canvas.save()
        this.command = this.commandBuilder(e)
        
        this.activelyDrawing = true
        break

      case 'mouseout':
      case 'mouseup':
        if (this.activelyDrawing) {
          this.editor.canvas.restore(this.savedCanvas)
          this.editor.executeCommand(this.command)
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

  commandBuilder(e) {
    const color = e.button == 0 ? this.editor.primaryColor.get() : this.editor.secondaryColor.get()
    return new PencilCommand(color, [this.editor.mousePosition])
  }

  activated() {
    ['mousedown', 'mousemove', 'mouseup', 'mouseout'].forEach(type =>
      this.editor.canvas.el.addEventListener(type, this.draw)
    )
  }

  deactivated() {
    ['mousedown', 'mousemove', 'mouseup', 'mouseout'].forEach(type =>
      this.editor.canvas.el.removeEventListener(type, this.draw)
    )
  }
}

export class Eraser extends Pencil {
  #eraserColor = '#000F'

  constructor(elements) {
    super(elements)
    this.name = 'Eraser'
    this.shortcut = 'E'
  }

  commandBuilder() {
    return new EraserCommand(this.#eraserColor, [this.editor.mousePosition])
  }
}

export class Bucket extends Tool {
  constructor(elements) {
    super('Bucket', 'regular-tools', elements, 'B')
    this.draw = this.draw.bind(this)
  }

  draw(e) {
    const color = e.button === 0 ? this.editor.primaryColor.get() : this.editor.secondaryColor.get()
    const command = new BucketCommand(color, this.editor.mousePosition)
    this.editor.executeCommand(command)
    this.editor.recordCommand(command)
  }

  activated() {
    this.editor.canvas.el.addEventListener('mouseup', this.draw)
  }

  deactivated() {
    this.editor.canvas.el.removeEventListener('mouseup', this.draw)
  }
}

class TwoPointPolygon extends Tool {
  constructor(elements) {
    super('Two Point Polygon', 'regular-tools', elements)
    this.draw = this.draw.bind(this)
  }

  draw(e) {
    // consider only left/right buttons
    if (e.button !== 0 && e.button !== 2) {
      return
    }

    switch (e.type) {
      case 'mousedown':
        if (this.activelyDrawing) {
          return
        }
        this.savedCanvas = this.editor.canvas.save()
        this.command = this.commandBuilder(e)
        this.activelyDrawing = true
        break

      case 'mousemove':
        if (this.activelyDrawing) {
          const position = this.editor.mousePosition
          this.editor.canvas.restore(this.savedCanvas)
          this.command.updatePosition(position)
          this.editor.executeCommand(this.command)
        }
        break
      case 'mouseup':
        if (this.activelyDrawing) {
          this.editor.canvas.restore(this.savedCanvas)
          this.editor.executeCommand(this.command)
          this.editor.recordCommand(this.command)

          this.activelyDrawing = false
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

export class Line extends TwoPointPolygon {
  constructor(elements) {
    super(elements)
    this.name = 'Line'
    this.shortcut = 'L'
  }

  commandBuilder(e) {
    const primaryOrSecondary = e.button === 0 ? 'primary' : 'secondary'
    return new LineCommand(
      this.editor[primaryOrSecondary + 'Color'].get(),
      this.editor.mousePosition,
      this.editor.mousePosition
    )
  }
}

export class Rectangle extends TwoPointPolygon {
  constructor(elements) {
    super(elements)
    this.name = 'Rectangle'
    this.shortcut = 'R'
  }

  commandBuilder(e) {
    const primaryOrSecondary = e.button === 0 ? 'primary' : 'secondary'
    return new RectangleCommand(
      this.editor[primaryOrSecondary + 'Color'].get(),
      this.editor.mousePosition,
      this.editor.mousePosition
    )
  }
}

export class EyeDropper extends Tool {
  constructor(elements) {
    super('Eye Dropper', 'regular-tools', elements, 'D')
    this.pickColor = this.pickColor.bind(this)
  }

  pickColor(e) {
    // consider only left/right buttons
    if (e.button !== 0 && e.button !== 2) {
      return
    }

    switch (e.type) {
      case 'mousedown':
        this.primaryOrSecondary = e.button === 0 ? 'primary' : 'secondary'
        this.savedColor = this.editor[this.primaryOrSecondary + 'Color'].get()
        this.picking = true
        // fallthrough to 'mousemove'

      case 'mousemove':
        if (this.picking) {
          const { x, y } = this.editor.mousePosition
          const [r, g, b, a] = this.editor.canvas.ctx.getImageData(x, y, 1, 1).data
          this.editor[this.primaryOrSecondary + 'Color'].set(`rgba(${r}, ${g}, ${b}, ${a})`)
        }
        break

      case 'mouseup':
        this.picking = false
        this.currentPixelColors = null
        if (this.savedTool) {
          this.savedTool.activate()
        }
        break

      case 'mouseout':
        if (this.picking) {
          this.picking = false
          this.editor[this.primaryOrSecondary + 'Color'].set(this.savedColor)
        }
    }
  }

  preActivation() {
    // saves the previous tool to reactivate it upon color selection
    this.savedTool = this.editor.getActiveTool()
  }

  activated() {
    ['mousedown', 'mousemove', 'mouseup', 'mouseout'].forEach(type =>
      this.editor.canvas.el.addEventListener(type, this.pickColor)
    )
  }

  deactivated() {
    ['mousedown', 'mousemove', 'mouseup', 'mouseout'].forEach(type =>
      this.editor.canvas.el.removeEventListener(type, this.pickColor)
    )
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

    // binds the color swatches to the value of the editor prop
    // (so it updates when an outside tool (such as eye dropper)
    //  changes the primary/secondary color outside here)
    if (this.specialColorSlot) {
      this.editor[this.specialColorSlot + 'Color'].addListener((value) => {
        // sets the swatch bg color accordingly
        this.inputs.forEach(el => el.closest('.swatch').style.backgroundColor = value)
      })

    }

    this.activated()
    this.inputs[0].value = this.defaultColor
    this.inputs[0].dispatchEvent(new Event('input'));
    this.deactivated()
  }

  bindColor(e) {
    // gets the color selected by the user
    const chosenColor = e.currentTarget.value

    // tells the editor a primary or secondary color was selected
    if (this.specialColorSlot) {
      this.editor[this.specialColorSlot + 'Color'].set(chosenColor)
    }
  }
}
