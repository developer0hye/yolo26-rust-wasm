import { classColor, classColorWithAlpha } from "./colors";
import type { Detection } from "./types";

/**
 * Draw image and detection overlays onto a canvas.
 *
 * When displayWidth/displayHeight are provided, the canvas renders at that
 * CSS size (with DPR scaling for Retina sharpness) and detection coordinates
 * are mapped from the original image space to the display space.
 * When omitted, the canvas renders at the image's native resolution.
 */
export function renderDetections(
  canvas: HTMLCanvasElement,
  image: HTMLImageElement,
  detections: Detection[],
  threshold: number,
  displayWidth?: number,
  displayHeight?: number,
): number {
  const ctx: CanvasRenderingContext2D | null = canvas.getContext("2d");
  if (!ctx) return 0;

  const imgW: number = image.naturalWidth;
  const imgH: number = image.naturalHeight;
  const isScaled: boolean =
    displayWidth !== undefined && displayHeight !== undefined;

  const drawW: number = displayWidth ?? imgW;
  const drawH: number = displayHeight ?? imgH;

  if (isScaled) {
    // Retina: internal resolution = display size × DPR
    const dpr: number = window.devicePixelRatio || 1;
    canvas.width = Math.round(drawW * dpr);
    canvas.height = Math.round(drawH * dpr);
    canvas.style.width = `${drawW}px`;
    canvas.style.height = `${drawH}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  } else {
    canvas.width = drawW;
    canvas.height = drawH;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }

  ctx.drawImage(image, 0, 0, drawW, drawH);

  // Map original image coords → draw coords
  const sx: number = drawW / imgW;
  const sy: number = drawH / imgH;

  const filtered: Detection[] = detections.filter(
    (d) => d.confidence >= threshold,
  );

  for (const d of filtered) {
    const x: number = d.x * sx;
    const y: number = d.y * sy;
    const w: number = d.width * sx;
    const h: number = d.height * sy;

    const color: string = classColor(d.class_id);
    const fillColor: string = classColorWithAlpha(d.class_id, 0.15);

    ctx.fillStyle = fillColor;
    ctx.fillRect(x, y, w, h);

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);

    // Label
    const label: string = `${d.class_name} ${(d.confidence * 100).toFixed(0)}%`;
    ctx.font = "bold 14px 'Geist Sans', -apple-system, sans-serif";
    const tw: number = ctx.measureText(label).width;
    const labelW: number = tw + 10;
    const labelH: number = 20;

    let ly: number;
    if (y - labelH - 2 >= 0) {
      ly = y - 6;
    } else {
      ly = y + labelH - 1;
    }

    let lx: number = x;
    if (lx + labelW > drawW) {
      lx = drawW - labelW;
    }
    if (lx < 0) lx = 0;

    ctx.fillStyle = color;
    ctx.fillRect(lx, ly - 15, labelW, labelH);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, lx + 5, ly);
  }

  return filtered.length;
}
