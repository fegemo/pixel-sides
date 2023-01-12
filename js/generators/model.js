import { LocalModel } from "./local-model.js"
import { RemoteModel } from "./remote-model.js"

export function selectModel(architecture) {
  // TODO check if user has hardware capabilities to run in the client
  // ....
  return new LocalModel(architecture)
}
