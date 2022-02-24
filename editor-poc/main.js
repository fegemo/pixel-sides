import generator from './generator.js'

let models = null
const progress = document.querySelector('#model-load-progress')
const source = document.querySelector('#input-image')
const targets = document.querySelectorAll('[id^="output-image-"]')

async function loadModels() {
    if (!models) {
        models = await generator.initialize(p => progress.value = p)
    }
}

document.querySelector('#load-model').addEventListener('click', loadModels)


document.querySelector('#generate').addEventListener('click', async () => {
    if (!models) {
        await loadModels()
    }

    await Promise.all([models.f2r(source, targets[0]), models.f2b(source, targets[1]),  models.f2l(source, targets[2])])
})

document.querySelectorAll('#input-file')[0].addEventListener('input', e => {
    const ctx = source.getContext('2d')
    
    const filePath = e.currentTarget.files[0]
    const image = new Image()
    image.addEventListener('load', e => {
        ctx.drawImage(image, 0, 0)
    })
    
    ctx.clearRect(0, 0, source.width, source.height);
    if (filePath) {
        image.src = URL.createObjectURL(filePath)
        targets.forEach(target => target.getContext('2d').clearRect(0, 0, target.width, target.height))
    }
})