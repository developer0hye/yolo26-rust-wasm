"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { StatusBanner } from "@/components/status-banner";
import { ImageDropzone } from "@/components/image-dropzone";
import { ConfidenceSlider } from "@/components/confidence-slider";
import { DetectionCanvas } from "@/components/detection-canvas";
import { DetectionList } from "@/components/detection-list";
import { InferenceClient } from "@/lib/worker-client";
import type { DetectionResult, ModelName, ModelStatus } from "@/lib/types";

const MODEL_OPTIONS: ModelName[] = [
  "yolo26n",
  "yolo26s",
  "yolo26m",
  "yolo26l",
  "yolo26x",
];

export default function Home() {
  const [selectedModel, setSelectedModel] = useState<ModelName>("yolo26n");
  const [modelStatus, setModelStatus] = useState<ModelStatus>("idle");
  const [statusMessage, setStatusMessage] = useState("Loading...");
  const [currentImage, setCurrentImage] = useState<HTMLImageElement | null>(
    null
  );
  const [cachedResult, setCachedResult] = useState<DetectionResult | null>(
    null
  );
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.25);
  const clientRef = useRef<InferenceClient | null>(null);

  const loadModel = useCallback((modelName: ModelName) => {
    const client = clientRef.current ?? new InferenceClient();
    clientRef.current = client;

    setModelStatus("idle");
    setStatusMessage(`Loading ${modelName}...`);

    client
      .initialize(modelName, (stage, message) => {
        setStatusMessage(message);
        if (stage === "weights") setModelStatus("loading-weights");
        else if (stage === "wasm") setModelStatus("loading-wasm");
        else if (stage === "model") setModelStatus("initializing-model");
      })
      .then((sizeMB) => {
        setModelStatus("ready");
        setStatusMessage(
          `${modelName} loaded (${sizeMB} MB). Select or drop an image to detect objects.`
        );
      })
      .catch((err) => {
        setModelStatus("error");
        setStatusMessage(`Failed to load model: ${err}`);
      });
  }, []);

  useEffect(() => {
    loadModel(selectedModel);
    return () => clientRef.current?.terminate();
    // Only run on mount — model switching is handled by handleModelChange
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleModelChange = useCallback(
    (modelName: ModelName) => {
      setSelectedModel(modelName);
      setCachedResult(null);
      loadModel(modelName);
    },
    [loadModel]
  );

  const handleImageSelected = useCallback(
    async (file: File) => {
      if (modelStatus !== "ready" || !clientRef.current) return;

      // createImageBitmap always bakes in EXIF rotation, eliminating
      // orientation mismatch between preprocessing and display canvases.
      const bitmap: ImageBitmap = await createImageBitmap(file);
      let w: number = bitmap.width;
      let h: number = bitmap.height;
      const MAX = 4096;
      if (w > MAX || h > MAX) {
        const scale: number = MAX / Math.max(w, h);
        w = Math.round(w * scale);
        h = Math.round(h * scale);
      }

      const offscreen: HTMLCanvasElement = document.createElement("canvas");
      offscreen.width = w;
      offscreen.height = h;
      const ctx: CanvasRenderingContext2D | null = offscreen.getContext("2d");
      if (!ctx) {
        bitmap.close();
        return;
      }
      ctx.drawImage(bitmap, 0, 0, w, h);
      bitmap.close();

      const imageData: ImageData = ctx.getImageData(0, 0, w, h);
      const pixels: Uint8Array = new Uint8Array(imageData.data.buffer);

      // Create EXIF-free display image from the normalized canvas pixels
      const displayImg: HTMLImageElement = await new Promise((resolve) => {
        offscreen.toBlob((blob) => {
          const img: HTMLImageElement = new Image();
          img.onload = () => {
            URL.revokeObjectURL(img.src);
            resolve(img);
          };
          img.src = URL.createObjectURL(blob!);
        }, "image/jpeg", 0.92);
      });

      setCurrentImage(displayImg);
      setModelStatus("detecting");
      setStatusMessage("Detecting objects...");

      clientRef.current!
        .detect(pixels, w, h, 0.0)
        .then((result: DetectionResult) => {
          setCachedResult(result);
          const count: number = result.detections.filter(
            (d) => d.confidence >= confidenceThreshold
          ).length;
          setModelStatus("ready");
          setStatusMessage(
            `${count} object${count !== 1 ? "s" : ""} detected in ${result.inference_time_ms.toFixed(0)}ms`
          );
        })
        .catch((err: unknown) => {
          setModelStatus("error");
          setStatusMessage(`Detection failed: ${err}`);
        });
    },
    [modelStatus, confidenceThreshold]
  );

  const handleThresholdChange = useCallback(
    (value: number) => {
      setConfidenceThreshold(value);
      if (cachedResult) {
        const count = cachedResult.detections.filter(
          (d) => d.confidence >= value
        ).length;
        setStatusMessage(
          `${count} object${count !== 1 ? "s" : ""} detected in ${cachedResult.inference_time_ms.toFixed(0)}ms`
        );
      }
    },
    [cachedResult]
  );

  const isModelBusy = modelStatus !== "ready";

  return (
    <main className="mx-auto flex h-[100dvh] max-w-6xl flex-col px-6 py-6 sm:px-10 sm:py-10">
      {/* Header */}
      <div className="shrink-0">
        <h1 className="text-[28px] font-semibold leading-tight text-[#0e1726] sm:text-[36px]">
          YOLO26 <span className="text-[#1a4a8f]">Rust WASM</span>
        </h1>
        <p className="mt-1 text-[15px] text-[#6b7a8d]">
          Browser-based object detection powered by Rust and WebAssembly
        </p>
      </div>

      {/* Controls bar */}
      <div className="mt-5 flex h-10 shrink-0 items-center gap-4">
        <ModelSelector
          value={selectedModel}
          onChange={handleModelChange}
          isDisabled={
            modelStatus !== "ready" && modelStatus !== "error"
          }
        />
        <StatusBanner status={modelStatus} message={statusMessage} />
        <ImageDropzone
          onImageSelected={handleImageSelected}
          isDisabled={isModelBusy}
        />
        <ConfidenceSlider
          value={confidenceThreshold}
          onChange={handleThresholdChange}
        />
      </div>

      {/* Content — fills remaining viewport height */}
      <div className="mt-5 grid min-h-0 flex-1 grid-cols-1 grid-rows-[1fr] gap-5 lg:grid-cols-[1fr_320px]">
        <div className="min-h-0 overflow-hidden">
          {currentImage ? (
            <DetectionCanvas
              image={currentImage}
              detections={cachedResult?.detections ?? []}
              confidenceThreshold={confidenceThreshold}
            />
          ) : (
            <ImageDropzoneArea
              onImageSelected={handleImageSelected}
              isDisabled={isModelBusy}
            />
          )}
        </div>

        <DetectionList
          detections={cachedResult?.detections ?? []}
          confidenceThreshold={confidenceThreshold}
          inferenceTimeMs={cachedResult?.inference_time_ms ?? 0}
        />
      </div>
    </main>
  );
}

function ModelSelector({
  value,
  onChange,
  isDisabled,
}: {
  value: ModelName;
  onChange: (model: ModelName) => void;
  isDisabled: boolean;
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value as ModelName)}
      disabled={isDisabled}
      className="h-full rounded-lg border border-[#e2e6eb] bg-white px-3 text-[14px] font-medium text-[#0e1726] transition-colors hover:border-[#1a4a8f] focus:border-[#1a4a8f] focus:outline-none disabled:cursor-not-allowed disabled:opacity-50"
    >
      {MODEL_OPTIONS.map((m) => (
        <option key={m} value={m}>
          {m.toUpperCase()}
        </option>
      ))}
    </select>
  );
}

function ImageDropzoneArea({
  onImageSelected,
  isDisabled,
}: {
  onImageSelected: (file: File) => void;
  isDisabled: boolean;
}) {
  const [isDragover, setIsDragover] = useState(false);

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        if (!isDisabled) setIsDragover(true);
      }}
      onDragLeave={() => setIsDragover(false)}
      onDrop={(e) => {
        e.preventDefault();
        setIsDragover(false);
        if (isDisabled) return;
        const file = e.dataTransfer.files[0];
        if (file?.type.startsWith("image/")) onImageSelected(file);
      }}
      className={`flex h-full min-h-[280px] items-center justify-center rounded-2xl border-2 border-dashed text-[15px] transition-colors ${
        isDragover
          ? "border-[#1a4a8f] bg-[#1a4a8f]/5 text-[#1a4a8f]"
          : "border-[#e2e6eb] text-[#6b7a8d]"
      }`}
    >
      Drop an image here to get started
    </div>
  );
}
