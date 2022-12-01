export default class Canvas {
  constructor(el, editor, size) {
    this.el = el
    this.editor = editor
    this.ctx = el.getContext('2d')
    
    this.el.addEventListener('mousemove', this.showMouseStats.bind(this))
    this.editor.containerEl.addEventListener('wheel', this.zoomInOrOut.bind(this))
    if (size) {
      setTimeout(() => this.resizeToFixedAndZoom.bind(this)(size), 0)
    } else {
      this.resizeToCover()
    }
  }

  showMouseStats(e) {
    const rect = e.currentTarget.getBoundingClientRect(),
      offsetX = Math.floor((e.clientX - rect.left) / this.editor.zoom),
      offsetY = Math.floor((e.clientY - rect.top) / this.editor.zoom);

    this.editor.mousePosition = {
      // x: e.offsetX,
      // y: e.offsetY
      x: offsetX,
      y: offsetY
    }
  }

  resizeToFixedAndZoom(size) {
    this.el.width = size[0]
    this.el.height = size[1]

    const minFromWidthOrHeight = Math.min(this.editor.containerEl.offsetWidth, this.editor.containerEl.offsetHeight)
    this.editor.zoom = (0.8 * minFromWidthOrHeight) / Math.min(...size)
  }

  resizeToCover() {
    const minFromWidthOrHeight = Math.min(this.editor.containerEl.offsetWidth, this.editor.containerEl.offsetHeight)
    this.el.width = 0.8 * minFromWidthOrHeight
    this.el.height = this.el.width
  }

  zoomInOrOut(e) {
    e.preventDefault()
    let scale = this.editor.zoom
    scale += e.deltaY * -0.005
    scale = Math.min(Math.max(.125, scale), 20);

    this.editor.zoom = scale
  }

}


// resize upon resize
// show mouse (x,y) on editor
