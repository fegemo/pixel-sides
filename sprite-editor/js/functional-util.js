export function* range(start, end, step = 1) {
  if (end === undefined) {
    end = start
    start = 0
  }

  for (let i = start; i < end; i += step) {
    yield i
  }
}

export function callOrReturn(object, property, defaultValue = null, params = []) {
  const fnOrvalue = object[property]
  if (fnOrvalue) {
    if (fnOrvalue instanceof Function) {
      return fnOrvalue(...params)
    } else {
      return fnOrvalue
    }
  }
  
  return defaultValue
}
