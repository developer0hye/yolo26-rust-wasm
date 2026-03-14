interface ConfidenceSliderProps {
  value: number;
  onChange: (value: number) => void;
}

export function ConfidenceSlider({ value, onChange }: ConfidenceSliderProps) {
  return (
    <div className="flex items-center gap-3">
      <label
        htmlFor="conf-slider"
        className="text-[14px] font-medium text-[#0e1726]/70"
      >
        Confidence
      </label>
      <input
        id="conf-slider"
        type="range"
        min="0.05"
        max="1.0"
        step="0.05"
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="h-1.5 w-40 cursor-pointer appearance-none rounded-full bg-[#e2e6eb] accent-[#1a4a8f]"
      />
      <span className="min-w-[2.5rem] text-[14px] tabular-nums text-[#6b7a8d]">
        {value.toFixed(2)}
      </span>
    </div>
  );
}
