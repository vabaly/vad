// @ts-ignore
import { log } from "./logging"

export type ONNXRuntimeAPI = any
export type ModelFetcher = () => Promise<ArrayBuffer>

export interface SpeechProbabilities {
  notSpeech: number
  isSpeech: number
}

export interface Model {
  reset_state: () => void
  process: (arr: Float32Array) => Promise<SpeechProbabilities>
}

export class Silero {
  _session
  _h
  _c
  _sr

  constructor(
    private ort: ONNXRuntimeAPI,
    private modelFetcher: ModelFetcher
  ) {}

  static new = async (ort: ONNXRuntimeAPI, modelFetcher: ModelFetcher) => {
    const model = new Silero(ort, modelFetcher)
    await model.init()
    return model
  }

  init = async () => {
    log.debug("initializing vad")
    const modelArrayBuffer = await this.modelFetcher()
    // 使用模型数据创建一个 推理会话对象（InferenceSession 对象）,这个对象包含了模型的结构和参数，并提供了一个接口，
    // 这个接口接受输入，获得输出
    // 所谓推理就是将将数据输入到模型后，在模型中进行计算处理的过程
    this._session = await this.ort.InferenceSession.create(modelArrayBuffer)
    // 创建一个张量（Tensor），张量和向量不同
    // 一个一维数组可以表示向量，也叫一维张量
    // 一个二维数组可以表示矩阵，也叫二位张量
    // 因此，向量只是张量的一部分
    // 下面这句话创建了一个存储数据为 int64 的长 16000 的向量（一维张量）
    // 在这里，第二个参数表示张量的形状，具体来说是这种形式：[第一维数据个数, 第二维数据个数, ...]
    // 这里缺少初始值,所以创建出来的 Tensor 里面的数据都是 undefined
    this._sr = new this.ort.Tensor("int64", [16000n])
    this.reset_state()
    log.debug("vad is initialized")
  }

  reset_state = () => {
    const zeroes = Array(2 * 64).fill(0)
    // 在这里，第三个参数表示张量形状，是一个三维张量
    // zeroes 这个一维数组就是上面三维张量的初始值
    // 具体来说，这个三维张量的总元素数量为 2 * 1 * 64 = 128
    // 因此我们需要创建一个长度为 128 的一维数组 zeroes
    // 在这个数组中，前 64 个元素对应第一个 1x64 的矩阵，后 64 个元素对应第二个 1x64 的矩阵
    // 因此这个一维数组可以被理解为将这两个矩阵按行排列在一起的结果
    this._h = new this.ort.Tensor("float32", zeroes, [2, 1, 64])
    this._c = new this.ort.Tensor("float32", zeroes, [2, 1, 64])
  }

  process = async (audioFrame: Float32Array): Promise<SpeechProbabilities> => {
    const t = new this.ort.Tensor("float32", audioFrame, [1, audioFrame.length])
    const inputs = {
      input: t,
      h: this._h,
      c: this._c,
      sr: this._sr,
    }
    const out = await this._session.run(inputs)
    this._h = out.hn
    this._c = out.cn
    const [isSpeech] = out.output.data
    const notSpeech = 1 - isSpeech
    return { notSpeech, isSpeech }
  }
}
