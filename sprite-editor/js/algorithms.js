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
}

export function manhattanDistance({x: x1, y: y1}, {x: x2, y: y2}) {
    return Math.abs(x2 - x1) + Math.abs(y2 - y1)
}

export function equalPosition({ x: x1, y: y1 }, { x: x2, y: y2 }) {
    return x1 === x2 && y1 === y2
}
