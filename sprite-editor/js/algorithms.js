// EFLA algorithm from: https://stackoverflow.com/a/40888742/1783793
export function lineEFLA(setPixel, {x: x1, y: y1}, {x: x2, y: y2}) {

    var dlt, mul,
        sl = y2 - y1,
        ll = x2 - x1,
        yl = false,
        lls = ll >> 31,
        sls = sl >> 31,
        i

    if ((sl ^ sls) - sls > (ll ^ lls) - lls) {
        sl ^= ll
        ll ^= sl
        sl ^= ll
        yl = true
    }

    dlt = ll < 0 ? -1 : 1
    mul = (ll === 0) ? sl : sl / ll

    if (yl) {
        x1 += 0.5
        for (i = 0; i !== ll; i += dlt)
            setPixel((x1 + i * mul) | 0, y1 + i)
    }
    else {
        y1 += 0.5
        for (i = 0; i !== ll; i += dlt)
            setPixel(x1 + i, (y1 + i * mul) | 0)
    }
    setPixel(x2, y2)
}

export function manhattanDistance({x: x1, y: y1}, {x: x2, y: y2}) {
    return Math.abs(x2 - x1) + Math.abs(y2 - y1)
}

export function equalPosition({ x: x1, y: y1 }, { x: x2, y: y2 }) {
    return x1 === x2 && y1 === y2
}

export function equalColor(color1, color2) {
    return Array.isArray(color1) && Array.isArray(color2) && color1.every((value, i) => value === color2[i])
}

export function floodFill(imageData, sourcePosition, setPixel) {
    const posToIndex = (x, y) => y * imageData.width * 4 + x * 4
    const getColor = pixelIndex => {
        const color = Array.from(imageData.data.slice(pixelIndex, pixelIndex + 4))
        return color.length ? color : [0, 0, 0, 0]
    }
    const sourceColor = getColor(posToIndex(sourcePosition.x, sourcePosition.y))
    
    const visited = []
    const hasVisited = (vx, vy) => visited.some(({x, y}) => vx === x && vy === y)

    function floodFillRecursive(x, y) {
        if (x < 0 || x > imageData.width - 1) return
        if (y < 0 || y > imageData.height - 1) return

        const currentColor = getColor(posToIndex(x, y))
        if (!equalColor(currentColor, sourceColor)) return

        setPixel({x, y})
        visited.push({x, y})

        if (!hasVisited(x, y - 1)) floodFillRecursive(x, y - 1)
        if (!hasVisited(x + 1, y)) floodFillRecursive(x + 1, y)
        if (!hasVisited(x, y + 1)) floodFillRecursive(x, y + 1)
        if (!hasVisited(x - 1, y)) floodFillRecursive(x - 1, y)
    }

    floodFillRecursive(sourcePosition.x, sourcePosition.y)
}
