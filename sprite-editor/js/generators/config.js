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
    checkpoints: [
      {
        file: '/models/pix2pix/front2left/model.json',
        source: DOMAINS.front,
        target: DOMAINS.left
      },
      {
        file: '/models/pix2pix/front2back/model.json',
        source: DOMAINS.front,
        target: DOMAINS.back
      },
      {
        file: '/models/pix2pix/front2right/model.json',
        source: DOMAINS.front,
        target: DOMAINS.right
      },
      {
        file: '/models/pix2pix/back2left/model.json',
        source: DOMAINS.back,
        target: DOMAINS.left
      },
      {
        file: '/models/pix2pix/back2front/model.json',
        source: DOMAINS.back,
        target: DOMAINS.front
      },
      {
        file: '/models/pix2pix/back2right/model.json',
        source: DOMAINS.back,
        target: DOMAINS.right
      },
      {
        file: '/models/pix2pix/right2left/model.json',
        source: DOMAINS.right,
        target: DOMAINS.left
      },
      {
        file: '/models/pix2pix/right2front/model.json',
        source: DOMAINS.right,
        target: DOMAINS.front
      },
      {
        file: '/models/pix2pix/right2back/model.json',
        source: DOMAINS.right,
        target: DOMAINS.back
      },
      {
        file: '/models/pix2pix/left2right/model.json',
        source: DOMAINS.left,
        target: DOMAINS.right
      },
      {
        file: '/models/pix2pix/left2front/model.json',
        source: DOMAINS.left,
        target: DOMAINS.front
      },
      {
        file: '/models/pix2pix/left2back/model.json',
        source: DOMAINS.left,
        target: DOMAINS.back
      },
    ],
    endpoint: '/api/pix2pix/{source}/2/{target}'
  },
  stargan: {
    checkpoints: [
      {
        file: '/models/stargan/model.json',
        source: DOMAINS.any,
        target: DOMAINS.any
      }
    ],
    endpoint: '/api/stargan/2/{target}'
  },
  collagan: {
    checkpoints: [
      {
        file: '/models/collagan/model.json',
        source: DOMAINS.many,
        target: DOMAINS.any
      }      
    ],
    endpoint: '/api/collagan/2/{target}'
  }
}
