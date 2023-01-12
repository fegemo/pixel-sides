export const DOMAINS = {
  front: 'front',
  left: 'left',
  back: 'back',
  right: 'right',

  any: 'any',
  many: 'many'
}

export const config = {
  pix2pix: {
    inputs: ['sourceImage'],
    checkpoints: [
      {
        file: 'models/pix2pix/front-to-left/model.json',
        source: DOMAINS.front,
        target: DOMAINS.left
      },
      {
        file: 'models/pix2pix/front-to-back/model.json',
        source: DOMAINS.front,
        target: DOMAINS.back
      },
      {
        file: 'models/pix2pix/front-to-right/model.json',
        source: DOMAINS.front,
        target: DOMAINS.right
      },
      {
        file: 'models/pix2pix/back-to-left/model.json',
        source: DOMAINS.back,
        target: DOMAINS.left
      },
      {
        file: 'models/pix2pix/back-to-front/model.json',
        source: DOMAINS.back,
        target: DOMAINS.front
      },
      {
        file: 'models/pix2pix/back-to-right/model.json',
        source: DOMAINS.back,
        target: DOMAINS.right
      },
      {
        file: 'models/pix2pix/right-to-left/model.json',
        source: DOMAINS.right,
        target: DOMAINS.left
      },
      {
        file: 'models/pix2pix/right-to-front/model.json',
        source: DOMAINS.right,
        target: DOMAINS.front
      },
      {
        file: 'models/pix2pix/right-to-back/model.json',
        source: DOMAINS.right,
        target: DOMAINS.back
      },
      {
        file: 'models/pix2pix/left-to-right/model.json',
        source: DOMAINS.left,
        target: DOMAINS.right
      },
      {
        file: 'models/pix2pix/left-to-front/model.json',
        source: DOMAINS.left,
        target: DOMAINS.front
      },
      {
        file: 'models/pix2pix/left-to-back/model.json',
        source: DOMAINS.left,
        target: DOMAINS.back
      },
    ],
    endpoint: '/api/pix2pix/{source}/2/{target}'
  },
  stargan: {
    inputs: ['sourceImage', 'targetDomain'],
    checkpoints: [
      {
        file: 'models/stargan/model.json',
        source: DOMAINS.any,
        target: DOMAINS.any
      }
    ],
    endpoint: '/api/stargan/2/{target}'
  },
  collagan: {
    checkpoints: [
      {
        file: 'models/collagan/model.json',
        source: DOMAINS.many,
        target: DOMAINS.any
      }      
    ],
    endpoint: '/api/collagan/2/{target}'
  }
}
