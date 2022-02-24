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
  }

  activated() {

  }

  deactivated() {

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
    
    this.activated()
    this.inputs[0].value = defaultColor
    this.inputs[0].dispatchEvent(new Event('input'));
    this.deactivated()
  }

  activated() {
    this.inputs.forEach(el => el.addEventListener('input', this.bindColor))
  }

  deactivated() {
    this.inputs.forEach(el => el.removeEventListener('input', this.bindColor))
  }

  bindColor(e) {
    this.inputs.forEach(el => {
      const chosenColor = e.currentTarget.value
      el.closest('.swatch').style.backgroundColor = chosenColor 
      if (el !== e.currentTarget) {
        el.value = chosenColor
      }
    })
  }
}