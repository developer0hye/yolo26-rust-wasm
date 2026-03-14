"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { renderDetections } from "@/lib/canvas-renderer";
import type { Detection } from "@/lib/types";

interface DetectionCanvasProps {
  image: HTMLImageElement | null;
  detections: Detection[];
  confidenceThreshold: number;
}

export function DetectionCanvas({
  image,
  detections,
  confidenceThreshold,
}: DetectionCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);
  const [modalSrc, setModalSrc] = useState<string>("");

  // Render at display size (fitted to container) with ResizeObserver
  useEffect(() => {
    if (!canvasRef.current || !containerRef.current || !image) return;

    const render = (): void => {
      const container: HTMLDivElement | null = containerRef.current;
      const canvas: HTMLCanvasElement | null = canvasRef.current;
      if (!container || !canvas) return;

      const containerW: number = container.clientWidth;
      const containerH: number = container.clientHeight;
      if (containerW <= 0 || containerH <= 0) return;

      const imgW: number = image.naturalWidth;
      const imgH: number = image.naturalHeight;

      // Fit image within container, maintain aspect ratio, don't upscale
      const scale: number = Math.min(containerW / imgW, containerH / imgH, 1);
      const displayW: number = Math.round(imgW * scale);
      const displayH: number = Math.round(imgH * scale);

      renderDetections(
        canvas,
        image,
        detections,
        confidenceThreshold,
        displayW,
        displayH,
      );
    };

    render();

    const observer: ResizeObserver = new ResizeObserver(render);
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, [image, detections, confidenceThreshold]);

  // Full-resolution render for the modal
  const handleCanvasClick = useCallback((): void => {
    if (!image) return;
    const tempCanvas: HTMLCanvasElement = document.createElement("canvas");
    renderDetections(tempCanvas, image, detections, confidenceThreshold);
    setModalSrc(tempCanvas.toDataURL("image/png"));
    setIsModalOpen(true);
  }, [image, detections, confidenceThreshold]);

  // Close modal on Escape
  useEffect(() => {
    if (!isModalOpen) return;
    const handleEsc = (e: KeyboardEvent): void => {
      if (e.key === "Escape") setIsModalOpen(false);
    };
    window.addEventListener("keydown", handleEsc);
    return () => window.removeEventListener("keydown", handleEsc);
  }, [isModalOpen]);

  if (!image) return null;

  return (
    <>
      <div
        ref={containerRef}
        className="group relative flex h-full cursor-pointer items-center justify-center overflow-hidden rounded-2xl border border-[#e2e6eb] bg-[#f4f6f8]"
        onClick={handleCanvasClick}
      >
        <canvas ref={canvasRef} className="block" />
        <span className="pointer-events-none absolute bottom-3 right-3 rounded-md bg-black/50 px-2.5 py-1 text-[12px] font-medium text-white opacity-0 transition-opacity group-hover:opacity-100">
          Click for full resolution
        </span>
      </div>

      {isModalOpen && (
        <div
          className="fixed inset-0 z-50 overflow-auto bg-black/70 p-8 backdrop-blur-sm"
          onClick={() => setIsModalOpen(false)}
        >
          <img
            src={modalSrc}
            alt="Full resolution detection result"
            className="mx-auto max-w-none"
            onClick={(e) => e.stopPropagation()}
          />
        </div>
      )}
    </>
  );
}
