import { classColor } from "@/lib/colors";
import type { Detection } from "@/lib/types";

interface DetectionListProps {
  detections: Detection[];
  confidenceThreshold: number;
  inferenceTimeMs: number;
}

export function DetectionList({
  detections,
  confidenceThreshold,
  inferenceTimeMs,
}: DetectionListProps) {
  const filtered = detections.filter(
    (d) => d.confidence >= confidenceThreshold
  );
  const hasDetections = detections.length > 0;

  return (
    <div className="flex h-full flex-col overflow-hidden rounded-2xl border border-[#e2e6eb] bg-white">
      <div className="shrink-0 border-b border-[#e2e6eb] px-4 py-2.5">
        {hasDetections ? (
          <>
            <span className="text-[14px] font-semibold text-[#0e1726]">
              {filtered.length} object{filtered.length !== 1 ? "s" : ""}
            </span>
            <span className="ml-1.5 text-[12px] text-[#6b7a8d]/60">
              {inferenceTimeMs.toFixed(0)}ms
            </span>
          </>
        ) : (
          <span className="text-[14px] font-semibold text-[#6b7a8d]/60">
            Detections
          </span>
        )}
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
        {hasDetections ? (
          filtered.map((d, i) => (
            <div
              key={i}
              className="border-b border-[#e2e6eb]/60 px-4 py-2 last:border-b-0"
            >
              <div className="flex flex-wrap items-center gap-x-2.5 gap-y-0.5">
                <span
                  className="h-2.5 w-2.5 shrink-0 rounded-sm"
                  style={{ backgroundColor: classColor(d.class_id) }}
                />
                <span className="text-[14px] font-medium text-[#0e1726]">
                  {d.class_name}
                </span>
                <span className="text-[14px] tabular-nums text-[#6b7a8d]">
                  {(d.confidence * 100).toFixed(1)}%
                </span>
              </div>
              <div className="mt-0.5 pl-5 text-[11px] tabular-nums text-[#6b7a8d]/70">
                ({Math.round(d.x)}, {Math.round(d.y)}) {Math.round(d.width)}×{Math.round(d.height)}
              </div>
            </div>
          ))
        ) : (
          <div className="flex h-full items-center justify-center px-4 py-8 text-[13px] text-[#6b7a8d]/60">
            Select an image to detect objects
          </div>
        )}
      </div>
    </div>
  );
}
