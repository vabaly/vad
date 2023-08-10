// ONNX 深度学习模型在浏览器上的运行时
import * as ort from "onnxruntime-web"
import {
  log,
  Message,
  Silero,
  SpeechProbabilities,
  defaultFrameProcessorOptions,
  FrameProcessor,
  FrameProcessorOptions,
  validateOptions,
} from "./_common"
import { modelFetcher } from "./model-fetcher"
import { assetPath } from "./asset-path"

interface RealTimeVADCallbacks {
  /** Callback to run after each frame. The size (number of samples) of a frame is given by `frameSamples`. */
  onFrameProcessed: (probabilities: SpeechProbabilities) => any

  /** Callback to run if speech start was detected but `onSpeechEnd` will not be run because the
   * audio segment is smaller than `minSpeechFrames`.
   */
  onVADMisfire: () => any

  /** Callback to run when speech start is detected */
  onSpeechStart: () => any

  /**
   * Callback to run when speech end is detected.
   * Takes as arg a Float32Array of audio samples between -1 and 1, sample rate 16000.
   * This will not run if the audio segment is smaller than `minSpeechFrames`.
   */
  onSpeechEnd: (audio: Float32Array) => any
}

/**
 * Customizable audio constraints for the VAD.
 * Excludes certain constraints that are set for the user by default.
 */
type AudioConstraints = Omit<
  MediaTrackConstraints,
  "channelCount" | "echoCancellation" | "autoGainControl" | "noiseSuppression"
>

interface RealTimeVADOptionsWithoutStream
  extends FrameProcessorOptions,
    RealTimeVADCallbacks {
  additionalAudioConstraints?: AudioConstraints
  workletURL: string
  stream: undefined
}

interface RealTimeVADOptionsWithStream
  extends FrameProcessorOptions,
    RealTimeVADCallbacks {
  stream: MediaStream
  workletURL: string
}

// 1.1 实时 VAD 的选项
export type RealTimeVADOptions =
  | RealTimeVADOptionsWithStream
  | RealTimeVADOptionsWithoutStream

const _getWorkletURL = () => {
  return assetPath("vad.worklet.bundle.min.js")
}

export const defaultRealTimeVADOptions: RealTimeVADOptions = {
  ...defaultFrameProcessorOptions,
  onFrameProcessed: (probabilities) => {},
  onVADMisfire: () => {
    log.debug("VAD misfire")
  },
  onSpeechStart: () => {
    log.debug("Detected speech start")
  },
  onSpeechEnd: () => {
    log.debug("Detected speech end")
  },
  workletURL: _getWorkletURL(),
  stream: undefined,
}

// 3. MicVAD 类
export class MicVAD {
  // @ts-ignore
  audioContext: AudioContext
  // @ts-ignore
  stream: MediaStream
  // @ts-ignore
  audioNodeVAD: AudioNodeVAD
  listening = false

  // 3.1 静态方法，实例化 MicVAD 类并且调用 init 方法，返回实例
  static async new(options: Partial<RealTimeVADOptions> = {}) {
    const vad = new MicVAD({ ...defaultRealTimeVADOptions, ...options })
    await vad.init()
    return vad
  }

  constructor(public options: RealTimeVADOptions) {
    // 3.2 这里仅仅是校验一下参数，没有别的作用，所以初始化的作用就是创建了一个实例而已，有一些属性被声明或赋予默认值
    validateOptions(options)
  }

  init = async () => {
    if (this.options.stream === undefined)
    // 3.3 可以自己传一个 MediaStream 进来，不然就是会自动创建，这里会调起麦克风
      this.stream = await navigator.mediaDevices.getUserMedia({
        // 3.4 getUserMedia 的参数就是对媒体流的一些 “约束”，
        // “约束” 其实就是设置，比如下面的 audio 就是对音频输入的一些约束或者说设置
        audio: {
          // 3.5 可以自己传一些设置参数进来，但是下面的属性不会被覆盖
          ...this.options.additionalAudioConstraints,
          // 单通道
          channelCount: 1,
          // 回声消除
          echoCancellation: true,
          // 自动增益,说话过弱或过强时就会自动调整接收水平
          autoGainControl: true,
          // 降噪
          noiseSuppression: true,
        },
      })
    else this.stream = this.options.stream

    // 3.5 音频处理上下文环境
    this.audioContext = new AudioContext()
    // 3.6 MediaStreamAudioSourceNode 类的构造函数接受两个参数:
    // 第一个参数是 AudioContext 对象，表示在哪个音频上下文中创建该节点；
    // 第二个参数是一个配置对象，其中 mediaStream 属性指定了要作为音频源的媒体流对象。
    // 因此下面的代码就是在 audioContext 中创建一个 MediaStream 类型的输入节点
    const source = new MediaStreamAudioSourceNode(this.audioContext, {
      mediaStream: this.stream,
    })

    this.audioNodeVAD = await AudioNodeVAD.new(this.audioContext, this.options)
    this.audioNodeVAD.receive(source)
  }

  pause = () => {
    this.audioNodeVAD.pause()
    this.listening = false
  }

  start = () => {
    this.audioNodeVAD.start()
    this.listening = true
  }
}

// 4. 自己定义的 AudioNodeVAD 类
export class AudioNodeVAD {
  // @ts-ignore
  frameProcessor: FrameProcessor
  // @ts-ignore
  entryNode: AudioNode

  // 4.1 创建 AudioNodeVAD 实例并初始化的静态方法
  static async new(
    ctx: AudioContext,
    options: Partial<RealTimeVADOptions> = {}
  ) {
    const vad = new AudioNodeVAD(ctx, {
      ...defaultRealTimeVADOptions,
      ...options,
    })
    await vad.init()
    return vad
  }

  // 4.2 也是仅仅实例化
  // 在 JavaScript 中，使用 public 关键字可以定义类的属性和构造函数参数。
  // 在这里，ctx 和 options 都是类的属性，它们作为构造函数的参数传入，可以在类的其他方法中使用
  // 因此, 学会一招同时声明类的属性和构造函数参数的方法
  constructor(public ctx: AudioContext, public options: RealTimeVADOptions) {
    validateOptions(options)
  }

  pause = () => {
    this.frameProcessor.pause()
  }

  start = () => {
    this.frameProcessor.resume()
  }

  receive = (node: AudioNode) => {
    node.connect(this.entryNode)
  }

  processFrame = async (frame: Float32Array) => {
    const { probs, msg, audio } = await this.frameProcessor.process(frame)
    if (probs !== undefined) {
      this.options.onFrameProcessed(probs)
    }
    switch (msg) {
      case Message.SpeechStart:
        this.options.onSpeechStart()
        break

      case Message.VADMisfire:
        this.options.onVADMisfire()
        break

      case Message.SpeechEnd:
        // @ts-ignore
        this.options.onSpeechEnd(audio)
        break

      default:
        break
    }
  }

  init = async () => {
    // 4.3 加载一个 worklet 模块, workletURL 所在目录默认是页面 JS 脚本所在目录下
    await this.ctx.audioWorklet.addModule(this.options.workletURL)
    const vadNode = new AudioWorkletNode(this.ctx, "vad-helper-worklet", {
      processorOptions: {
        frameSamples: this.options.frameSamples,
      },
    })
    this.entryNode = vadNode

    const model = await Silero.new(ort, modelFetcher)

    this.frameProcessor = new FrameProcessor(model.process, model.reset_state, {
      frameSamples: this.options.frameSamples,
      positiveSpeechThreshold: this.options.positiveSpeechThreshold,
      negativeSpeechThreshold: this.options.negativeSpeechThreshold,
      redemptionFrames: this.options.redemptionFrames,
      preSpeechPadFrames: this.options.preSpeechPadFrames,
      minSpeechFrames: this.options.minSpeechFrames,
    })

    vadNode.port.onmessage = async (ev: MessageEvent) => {
      switch (ev.data?.message) {
        case Message.AudioFrame:
          const buffer: ArrayBuffer = ev.data.data
          const frame = new Float32Array(buffer)
          await this.processFrame(frame)
          break

        default:
          break
      }
    }
  }
}
