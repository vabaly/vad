// 5. 这个就是 Worklet 音频处理文件, 会被 TSC 转换成 JS
// 然后再被 webpack 打包成一个独立的 JS 文件
// 代码里面使用了一些 AudioWorklet 的全局变量,那么这个全局变量都有哪些呢?
// 可以看 AudioWorkletGlobalScope 的定义:https://developer.mozilla.org/en-US/docs/Web/API/AudioWorkletGlobalScope#instance_properties

import { Message, log, Resampler } from "./_common"

interface WorkletOptions {
  frameSamples: number
}

class Processor extends AudioWorkletProcessor {
  // @ts-ignore
  resampler: Resampler
  _initialized = false
  options: WorkletOptions

  constructor(options) {
    super()
    // 5.1 传进来的配置属性
    this.options = options.processorOptions as WorkletOptions
    this.init()
  }
  init = async () => {
    log.debug("initializing worklet")
    // 5.2 这里单纯的实例化一个采样器的实例,什么也没做
    this.resampler = new Resampler({
      // 输入音频的采样率（即原始采样率）,这个值和浏览器的设定有关系
      // 在我的 Edge 浏览器中是 48000
      nativeSampleRate: sampleRate,
      // 目标采样率, 每秒采 16000 个点, 即 16KHz
      targetSampleRate: 16000,
      // 目标帧大小（即期望每个输出帧包含的采样点数）
      // 当确定一帧包含的采样点数以及采样率之后,就能算出一帧的长度(单位是时间)
      // frame 就是帧的意思
      // frameLength = frameSamples / sampleRate
      // 默认是 frameLength = 1536 个 / (16000 个 / 秒) ≈ 0.096 秒
      targetFrameSize: this.options.frameSamples,
    })
    this._initialized = true
    log.debug("initialized worklet")
  }

  // 输入的音频处理
  process(
    inputs: Float32Array[][],
    outputs: Float32Array[][],
    parameters: Record<string, Float32Array>
  ): boolean {
    // @ts-ignore
    // 因为是一个输入来源,一个单通道的音频数据,所以直接取出来
    const arr = inputs[0][0]

    if (this._initialized && arr instanceof Float32Array) {
      const frames = this.resampler.process(arr)
      for (const frame of frames) {
        this.port.postMessage(
          { message: Message.AudioFrame, data: frame.buffer },
          [frame.buffer]
        )
      }
    }
    return true
  }
}

registerProcessor("vad-helper-worklet", Processor)
