export function frameAt(frames, time) {
  if (!frames?.length) return null;
  let lo = 0,
    hi = frames.length - 1;
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    if (frames[mid].time <= time) lo = mid;
    else hi = mid - 1;
  }
  return frames[lo];
}

export function visibleDetections(detections, confidence, classes) {
  return (detections || []).filter(
    (d) => d.confidence >= confidence && classes.includes(d.class_id),
  );
}

export function formatTime(seconds = 0) {
  const value = Math.max(0, Math.floor(Number.isFinite(seconds) ? seconds : 0));
  return `${String(Math.floor(value / 60)).padStart(2, "0")}:${String(value % 60).padStart(2, "0")}`;
}

export function resultsCsv(frames) {
  const rows = ["time_seconds,frame,class,confidence,x1,y1,x2,y2"];
  for (const frame of frames) {
    for (const d of frame.detections) {
      rows.push(
        [
          frame.time ?? 0,
          frame.frame ?? 0,
          d.class_name,
          d.confidence,
          ...d.box,
        ].join(","),
      );
    }
  }
  return rows.join("\r\n");
}

export function saveBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
