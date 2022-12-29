export function* range(start, end, step = 1) {
  if (end === undefined) {
    end = start
    start = 0
  }

  for (let i = start; i < end; i += step) {
    yield i
  }
}
