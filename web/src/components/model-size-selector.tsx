import type { ModelSize } from "@/lib/types";
import { MODEL_SIZE_LABELS } from "@/lib/types";

interface ModelSizeSelectorProps {
  value: ModelSize;
  onChange: (size: ModelSize) => void;
  isDisabled: boolean;
}

const MODEL_SIZES: ModelSize[] = ["n", "s", "m", "l", "x"];

export function ModelSizeSelector({
  value,
  onChange,
  isDisabled,
}: ModelSizeSelectorProps) {
  return (
    <div className="flex items-center gap-3">
      <label
        htmlFor="model-size"
        className="text-[14px] font-medium text-[#0e1726]/70"
      >
        Model
      </label>
      <select
        id="model-size"
        value={value}
        onChange={(e) => onChange(e.target.value as ModelSize)}
        disabled={isDisabled}
        className="h-8 rounded-md border border-[#e2e6eb] bg-white px-2 text-[14px] text-[#0e1726] disabled:cursor-not-allowed disabled:opacity-50"
      >
        {MODEL_SIZES.map((size) => (
          <option key={size} value={size}>
            {MODEL_SIZE_LABELS[size]}
          </option>
        ))}
      </select>
    </div>
  );
}
