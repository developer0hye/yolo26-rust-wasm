"use client";

import { useCallback, useRef } from "react";

interface ImageDropzoneProps {
  onImageSelected: (file: File) => void;
  isDisabled: boolean;
}

export function ImageDropzone({
  onImageSelected,
  isDisabled,
}: ImageDropzoneProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    (file: File) => {
      if (!file.type.startsWith("image/")) return;
      onImageSelected(file);
    },
    [onImageSelected]
  );

  return (
    <label
      className={`inline-flex shrink-0 cursor-pointer items-center rounded-full px-5 py-2 text-[14px] font-semibold transition-colors ${
        isDisabled
          ? "cursor-not-allowed bg-[#e2e6eb] text-[#6b7a8d]/50"
          : "bg-[#1a4a8f] text-white hover:bg-[#15407d]"
      }`}
    >
      Select Image
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        disabled={isDisabled}
        className="sr-only"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleFile(file);
        }}
      />
    </label>
  );
}
