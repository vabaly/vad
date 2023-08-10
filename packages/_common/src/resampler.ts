import { log } from "./logging"

interface ResamplerOptions {
  nativeSampleRate: number
  targetSampleRate: number
  targetFrameSize: number
}

// 6. 用在 audioWorklet 中的类
export class Resampler {
  inputBuffer: Array<number>

  constructor(public options: ResamplerOptions) {
    // 一般浏览器默认的采样率不会低于 16000
    // 不过好像在非实时语音的情况下, nativeSampleRate 好像是用户传的参数
    if (options.nativeSampleRate < 16000) {
      log.error(
        "nativeSampleRate is too low. Should have 16000 = targetSampleRate <= nativeSampleRate"
      )
    }
    this.inputBuffer = []
  }

  // 6.1 处理音频数据
  process = (audioFrame: Float32Array): Float32Array[] => {
    const outputFrames: Array<Float32Array> = []

    // 6.2 audioFrame 数组中每一项都是一个 -1 到 1 的数字
    // 这个数字代表采样时刻声音大小和强度(即音频信号幅度)
    // 这些音频声音大小和强度最终要输入到语音识别模型中才能得到具体的话
    // 因此,语音识别模型才是最重要的,因为每个人对同一个字说出来的音频信号都不太一样,
    // 同一个人说同一个字也可能不一样,所以训练模型的数据越多越好
    for (const sample of audioFrame) {
      // inputBuffer 就是 audioFrame 一个副本的感觉
      this.inputBuffer.push(sample)
    }

    while (
      // 采样点数量 / 原始采样率 = 整体的秒数
      // 整体的秒数 * 目标采样率 = 目标采样点数量
      (this.inputBuffer.length * this.options.targetSampleRate) /
        this.options.nativeSampleRate >
        // 当目标采样点数量大于一帧的数量时,就会继续这个循环
      this.options.targetFrameSize
    ) {
      const outputFrame = new Float32Array(this.options.targetFrameSize)
      let outputIndex = 0
      let inputIndex = 0
      // 当 outputIndex 小于一帧的采样点数量,循环继续
      while (outputIndex < this.options.targetFrameSize) {
        let sum = 0
        let num = 0
        while (
          inputIndex <
          Math.min(
            // 总采样点数量
            this.inputBuffer.length,
            // 原始采样率 / 目标采样率 = 同一段时间原始采样数量和目标采样数量的比值
            // 比值可能小于 1,可能大于 1,这里还是大于 1 的
            ((outputIndex + 1) * this.options.nativeSampleRate) /
              this.options.targetSampleRate
          )
        ) {
          sum += this.inputBuffer[inputIndex] as number
          num++
          inputIndex++
        }
        // outputFrame 的采样点等于几个原始采样点的平均值
        outputFrame[outputIndex] = sum / num
        
        outputIndex++
      }
      this.inputBuffer = this.inputBuffer.slice(inputIndex)
      outputFrames.push(outputFrame)
    }
    return outputFrames
  }
}
