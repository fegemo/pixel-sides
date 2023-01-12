export class Plugin {
  #cssPath
  #readyForInstallation = []

  constructor(name, dependencies = [], cssPath = null) {
    this.name = name
    this.dependencies = dependencies
    this.#cssPath = cssPath
  }

  preInstall(editor) {
    if (this.#cssPath) {
      this.loadCss()
    }
  }

  install(editor) {
    throw new Error('Abstract method "install" called on class Plugin.')
  }

  loadCss() {
    const linkEl = document.createElement('link')
    linkEl.rel = 'stylesheet'
    linkEl.href = this.#cssPath
    linkEl.dataset.addedBy = this.name

    const lastCssLinkEl = Array.from(document.head.querySelectorAll('link[rel="stylesheet"]')).at(-1)
    lastCssLinkEl.insertAdjacentElement("afterend", linkEl)

    const cssLoaded = new Promise((resolve, reject) => {
      linkEl.onload = () => resolve()
    })

    this.#readyForInstallation.push(cssLoaded)
  }

  get readyToInstall() {
    return Promise.all(this.#readyForInstallation)
  }
}
