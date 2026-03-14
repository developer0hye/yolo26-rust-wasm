export function classColor(classId: number): string {
  const hue = (classId * 137.5) % 360;
  return `hsl(${hue}, 70%, 50%)`;
}

export function classColorWithAlpha(classId: number, alpha: number): string {
  const hue = (classId * 137.5) % 360;
  return `hsla(${hue}, 70%, 50%, ${alpha})`;
}
