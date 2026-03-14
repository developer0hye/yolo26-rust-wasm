import type { ModelStatus } from "@/lib/types";

interface StatusBannerProps {
  status: ModelStatus;
  message: string;
}

export function StatusBanner({ status, message }: StatusBannerProps) {
  const isError = status === "error";
  const isReady = status === "ready";

  return (
    <div
      className={`min-w-0 flex-1 truncate rounded-lg px-4 py-2 text-[13px] font-medium transition-colors ${
        isError
          ? "bg-red-500/10 text-red-600"
          : isReady
            ? "bg-[#1a4a8f]/8 text-[#1a4a8f]"
            : "bg-[#f4f6f8] text-[#6b7a8d]"
      }`}
      title={message}
    >
      {message}
    </div>
  );
}
