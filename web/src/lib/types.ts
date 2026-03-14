export interface Detection {
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
  class_id: number;
  class_name: string;
}

export interface DetectionResult {
  detections: Detection[];
  inference_time_ms: number;
  image_width: number;
  image_height: number;
}

export type ModelStatus =
  | "idle"
  | "loading-wasm"
  | "loading-weights"
  | "initializing-model"
  | "ready"
  | "detecting"
  | "error";

export type WorkerRequest =
  | { type: "init"; weightsBuffer: ArrayBuffer }
  | {
      type: "detect";
      pixels: Uint8Array;
      width: number;
      height: number;
      confidenceThreshold: number;
    };

export type WorkerResponse =
  | { type: "init:progress"; stage: string; message: string }
  | { type: "init:done"; modelSizeMB: number }
  | { type: "init:error"; error: string }
  | { type: "detect:done"; resultJson: string }
  | { type: "detect:error"; error: string };
