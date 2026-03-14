import type { DetectionResult, ModelName, WorkerResponse } from "./types";

export type ProgressCallback = (stage: string, message: string) => void;

export class InferenceClient {
  private worker: Worker | null = null;

  async initialize(
    modelName: ModelName,
    onProgress?: ProgressCallback
  ): Promise<number> {
    // Terminate previous worker if re-initializing with a different model
    this.terminate();

    return new Promise((resolve, reject) => {
      this.worker = new Worker(
        new URL("../workers/inference.worker.ts", import.meta.url)
      );

      onProgress?.("weights", `Fetching ${modelName} weights...`);

      fetch(`/weights/${modelName}.safetensors`)
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

          this.worker.postMessage(
            { type: "init", weightsBuffer, modelName },
            [weightsBuffer]
          );
        })
        .catch(reject);
    });
  }

  async detect(
    pixels: Uint8Array,
    width: number,
    height: number,
    confidenceThreshold: number
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

      this.worker.postMessage(
        {
          type: "detect",
          pixels,
          width,
          height,
          confidenceThreshold,
        },
        [pixels.buffer]
      );
    });
  }

  terminate(): void {
    this.worker?.terminate();
    this.worker = null;
  }
}
