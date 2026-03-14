import type { DetectionResult, ModelSize, WorkerResponse } from "./types";

export type ProgressCallback = (stage: string, message: string) => void;

export class InferenceClient {
  private worker: Worker | null = null;

  async initialize(
    modelSize: ModelSize,
    onProgress?: ProgressCallback,
  ): Promise<number> {
    return new Promise((resolve, reject) => {
      // Terminate existing worker if switching model sizes
      this.worker?.terminate();
      this.worker = new Worker(
        new URL("../workers/inference.worker.ts", import.meta.url),
      );

      onProgress?.("weights", `Fetching yolo26${modelSize} weights...`);

      fetch(`/weights/yolo26${modelSize}.safetensors`)
        .then((resp) => {
          if (!resp.ok)
            throw new Error(`HTTP ${resp.status}: ${resp.statusText}`);
          return resp.arrayBuffer();
        })
        .then((weightsBuffer) => {
          if (!this.worker) throw new Error("Worker terminated");

          this.worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
            const msg = event.data;
            if (msg.type === "init:progress") {
              onProgress?.(msg.stage, msg.message);
            } else if (msg.type === "init:done") {
              resolve(msg.modelSizeMB);
            } else if (msg.type === "init:error") {
              reject(new Error(msg.error));
            }
          };

          this.worker.onerror = (err) => reject(err);

          // Transfer the ArrayBuffer to the worker (zero-copy)
          this.worker.postMessage({ type: "init", weightsBuffer, modelSize }, [
            weightsBuffer,
          ]);
        })
        .catch(reject);
    });
  }

  async detect(
    pixels: Uint8Array,
    width: number,
    height: number,
    confidenceThreshold: number,
  ): Promise<DetectionResult> {
    return new Promise((resolve, reject) => {
      if (!this.worker) {
        reject(new Error("Worker not initialized"));
        return;
      }

      this.worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
        const msg = event.data;
        if (msg.type === "detect:done") {
          resolve(JSON.parse(msg.resultJson) as DetectionResult);
        } else if (msg.type === "detect:error") {
          reject(new Error(msg.error));
        }
      };

      // Transfer the pixel buffer to the worker (zero-copy)
      this.worker.postMessage(
        {
          type: "detect",
          pixels,
          width,
          height,
          confidenceThreshold,
        },
        [pixels.buffer],
      );
    });
  }

  terminate(): void {
    this.worker?.terminate();
    this.worker = null;
  }
}
