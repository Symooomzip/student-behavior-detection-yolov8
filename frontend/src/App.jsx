import { useEffect, useMemo, useRef, useState } from "react";
import { json, request, uploadImage } from "./api";
import {
  formatTime,
  frameAt,
  resultsCsv,
  saveBlob,
  visibleDetections,
} from "./results";

const REPOSITORY =
  "https://github.com/Symooomzip/student-behavior-detection-yolov8";
const MODES = [
  ["image", "Image"],
  ["video", "Video"],
  ["camera", "Camera"],
];
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

export default function App() {
  const [meta, setMeta] = useState(null);
  const [mode, setMode] = useState("image");
  const [page, setPage] = useState("analysis");
  const [file, setFile] = useState(null);
  const [source, setSource] = useState("");
  const [dimensions, setDimensions] = useState(null);
  const [result, setResult] = useState(null);
  const [videoResult, setVideoResult] = useState(null);
  const [job, setJob] = useState(null);
  const [busy, setBusy] = useState(false);
  const [phase, setPhase] = useState("");
  const [error, setError] = useState("");
  const [confidence, setConfidence] = useState(0.25);
  const [iou, setIou] = useState(0.45);
  const [sampleFps, setSampleFps] = useState(4);
  const [activeClasses, setActiveClasses] = useState([]);
  const [annotations, setAnnotations] = useState(true);
  const [selected, setSelected] = useState(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraFrame, setCameraFrame] = useState("");
  const [cameraMs, setCameraMs] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [mediaError, setMediaError] = useState(false);
  const input = useRef(null),
    image = useRef(null),
    video = useRef(null),
    webcam = useRef(null);
  const sequence = useRef(0),
    controller = useRef(null),
    jobId = useRef(null),
    stream = useRef(null);
  const cameraRunning = useRef(false),
    settings = useRef({ iou });
  settings.current = { iou };

  useEffect(() => {
    let alive = true;
    json("/api/meta")
      .then((value) => {
        if (!alive) return;
        setMeta(value);
        setConfidence(value.confidence);
        setIou(value.iou);
        setActiveClasses(value.classes.map((c) => c.id));
      })
      .catch((e) => {
        if (alive) setError(e.message);
      });
    return () => {
      alive = false;
    };
  }, []);

  useEffect(
    () => () => {
      if (source.startsWith("blob:")) URL.revokeObjectURL(source);
    },
    [source],
  );
  useEffect(
    () => () => {
      sequence.current++;
      controller.current?.abort();
      cameraRunning.current = false;
      stream.current?.getTracks().forEach((track) => track.stop());
      if (jobId.current)
        fetch(`/api/videos/${jobId.current}`, {
          method: "DELETE",
          keepalive: true,
        }).catch(() => {});
    },
    [],
  );

  // Sync overlays with the actual presented video frame, including while seeking.
  useEffect(() => {
    const element = video.current;
    if (!element || !videoResult) return;
    let cancelled = false,
      handle;
    const tick = () => {
      if (cancelled) return;
      setCurrentTime(element.currentTime);
      handle = requestAnimationFrame(tick);
    };
    handle = requestAnimationFrame(tick);
    return () => {
      cancelled = true;
      cancelAnimationFrame(handle);
    };
  }, [videoResult, source, page]);

  const current =
    mode === "video" ? frameAt(videoResult?.frames, currentTime) : result;
  const detections = useMemo(
    () => visibleDetections(current?.detections, confidence, activeClasses),
    [current, confidence, activeClasses],
  );
  const chosen = detections.find((d) => d.id === selected) || null;
  const label = (id) =>
    meta?.classes.find((c) => c.id === id)?.label || "Behavior";
  const counts =
    meta?.classes.map((c) => ({
      ...c,
      count: (current?.detections || []).filter(
        (d) => d.class_id === c.id && d.confidence >= confidence,
      ).length,
    })) || [];
  const canAnalyze = !!meta?.model_available && (mode === "camera" || !!file);
  const hasMedia = mode === "camera" ? !!cameraFrame : !!source;
  const liveSource = mode === "camera" ? cameraFrame : source;
  const size = current || dimensions;
  const isFinished = !!current || !!videoResult;

  function stopCamera() {
    cameraRunning.current = false;
    stream.current?.getTracks().forEach((track) => track.stop());
    stream.current = null;
    if (webcam.current) webcam.current.srcObject = null;
    setCameraActive(false);
  }

  function interrupt() {
    sequence.current++;
    controller.current?.abort();
    controller.current = null;
    stopCamera();
    if (jobId.current) {
      fetch(`/api/videos/${jobId.current}`, { method: "DELETE" }).catch(
        () => {},
      );
      jobId.current = null;
    }
    setBusy(false);
    setPhase("");
  }

  function changeMode(value) {
    if (value === mode) return;
    interrupt();
    setMode(value);
    setFile(null);
    setSource("");
    setResult(null);
    setJob(null);
    setVideoResult(null);
    setCameraFrame("");
    setCameraMs(null);
    setDimensions(null);
    setError("");
    setCurrentTime(0);
    setDuration(0);
    setSelected(null);
    setMediaError(false);
    setPlaying(false);
  }

  function loadFile(value) {
    if (!value) return;
    const imageMode = mode === "image";
    const supported = imageMode
      ? /\.(jpe?g|png|webp|bmp)$/i
      : /\.(mp4|mov|avi|webm|mkv|m4v)$/i;
    if (!supported.test(value.name)) {
      setError(
        imageMode
          ? "Choose a JPG, PNG, WebP, or BMP image."
          : "Choose an MP4, MOV, AVI, WebM, or MKV video.",
      );
      return;
    }
    if (value.size > (imageMode ? 25 : 200) * 1024 * 1024) {
      setError(`The file exceeds the ${imageMode ? 25 : 200} MB limit.`);
      return;
    }
    interrupt();
    setFile(value);
    setSource(URL.createObjectURL(value));
    setResult(null);
    setVideoResult(null);
    setJob(null);
    setError("");
    setSelected(null);
    setDimensions(null);
    setCurrentTime(0);
    setDuration(0);
    setPlaying(false);
    setMediaError(false);
  }

  async function loadExample() {
    const id = sequence.current;
    try {
      const response = await request("/api/example");
      const blob = await response.blob();
      if (id === sequence.current)
        loadFile(new File([blob], "image001.jpg", { type: "image/jpeg" }));
    } catch (e) {
      if (id === sequence.current) setError(e.message);
    }
  }

  async function analyze() {
    if (!file || busy) return;
    interrupt();
    const id = sequence.current;
    const abort = new AbortController();
    controller.current = abort;
    setBusy(true);
    setError("");
    setResult(null);
    setVideoResult(null);
    setJob(null);
    setSelected(null);
    setPhase(mode === "video" ? "Uploading video…" : "Analyzing image…");
    try {
      if (mode === "image") {
        const value = await uploadImage(file, iou, abort.signal);
        if (id !== sequence.current) return;
        setResult(value);
        setSelected(value.detections[0]?.id ?? null);
      } else {
        const body = new FormData();
        body.append("file", file);
        let status = await json(
          `/api/videos?sample_fps=${sampleFps}&iou=${iou}&confidence=${confidence}`,
          { method: "POST", body, signal: abort.signal },
        );
        if (id !== sequence.current) {
          fetch(`/api/videos/${status.id}`, { method: "DELETE" }).catch(
            () => {},
          );
          return;
        }
        jobId.current = status.id;
        setJob(status);
        while (!["complete", "error", "cancelled"].includes(status.state)) {
          await sleep(600);
          if (id !== sequence.current) return;
          status = await json(`/api/videos/${status.id}`, {
            signal: abort.signal,
          });
          if (id !== sequence.current) return;
          setJob(status);
          setPhase(
            status.state === "queued"
              ? "Waiting for the model…"
              : "Analyzing video and preparing the MP4…",
          );
        }
        if (status.state === "error")
          throw new Error(status.error || "Processing failed.");
        if (status.state === "cancelled") {
          setPhase("Analysis canceled.");
          return;
        }
        const value = await json(`/api/videos/${status.id}/results`, {
          signal: abort.signal,
        });
        if (id !== sequence.current) return;
        setVideoResult(value);
        setSource(`/api/videos/${status.id}/preview.mp4`);
        setCurrentTime(0);
        setDuration(value.duration);
        setMediaError(false);
        setSelected(value.frames[0]?.detections[0]?.id ?? null);
        jobId.current = null;
      }
      setPhase("Analysis complete.");
    } catch (e) {
      if (e.name !== "AbortError" && id === sequence.current) {
        setError(e.message);
        setPhase("Analysis failed.");
      }
    } finally {
      if (id === sequence.current) {
        setBusy(false);
        controller.current = null;
      }
    }
  }

  async function startCamera() {
    interrupt();
    const id = sequence.current;
    setError("");
    setResult(null);
    setCameraFrame("");
    setPhase("Starting camera…");
    setBusy(true);
    try {
      if (!navigator.mediaDevices?.getUserMedia)
        throw new Error(
          "The camera requires localhost or an HTTPS connection.",
        );
      const media = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 960 }, height: { ideal: 540 } },
        audio: false,
      });
      if (id !== sequence.current) {
        media.getTracks().forEach((t) => t.stop());
        return;
      }
      stream.current = media;
      webcam.current.srcObject = media;
      await webcam.current.play();
      if (id !== sequence.current) return;
      cameraRunning.current = true;
      setCameraActive(true);
      setBusy(false);
      setPhase("Camera active");
      const canvas = document.createElement("canvas");
      const context = canvas.getContext("2d");
      while (cameraRunning.current && id === sequence.current) {
        const w = webcam.current.videoWidth,
          h = webcam.current.videoHeight;
        if (!w || !h) {
          await sleep(100);
          continue;
        }
        canvas.width = Math.min(960, w);
        canvas.height = Math.round((h * canvas.width) / w);
        context.drawImage(webcam.current, 0, 0, canvas.width, canvas.height);
        const blob = await new Promise((resolve) =>
          canvas.toBlob(resolve, "image/jpeg", 0.85),
        );
        const preview = canvas.toDataURL("image/jpeg", 0.85);
        if (!blob || id !== sequence.current) break;
        const abort = new AbortController();
        controller.current = abort;
        const start = performance.now();
        const value = await uploadImage(
          new File([blob], "camera.jpg", { type: "image/jpeg" }),
          settings.current.iou,
          abort.signal,
        );
        if (id !== sequence.current || !cameraRunning.current) break;
        setCameraFrame(preview);
        setResult(value);
        setCameraMs(Math.round(performance.now() - start));
        await sleep(80);
      }
    } catch (e) {
      if (id === sequence.current && e.name !== "AbortError") {
        setError(
          e.name === "NotAllowedError"
            ? "Camera access denied. Allow camera access in your browser and try again."
            : e.name === "NotFoundError"
              ? "No camera detected."
              : e.message,
        );
        stopCamera();
        setPhase("Camera stopped.");
      }
    } finally {
      if (id === sequence.current) setBusy(false);
    }
  }

  function exportData(format) {
    const frames = (videoResult?.frames || [result])
      .filter(Boolean)
      .map((f) => ({
        ...f,
        detections: visibleDetections(f.detections, confidence, activeClasses),
      }));
    const data = {
      model: meta.model,
      source: file?.name || "camera",
      confidence,
      classes: activeClasses,
      ...(videoResult
        ? { sample_fps: videoResult.sample_fps, duration: videoResult.duration }
        : {}),
      frames,
    };
    saveBlob(
      new Blob(
        [
          format === "json"
            ? JSON.stringify(data, null, 2)
            : resultsCsv(frames),
        ],
        {
          type:
            format === "json" ? "application/json" : "text/csv;charset=utf-8",
        },
      ),
      `detections.${format}`,
    );
  }

  async function exportImage() {
    try {
      const media = mode === "video" ? video.current : image.current;
      if (!media) return;
      const canvas = document.createElement("canvas");
      canvas.width = mode === "video" ? media.videoWidth : media.naturalWidth;
      canvas.height =
        mode === "video" ? media.videoHeight : media.naturalHeight;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(media, 0, 0);
      const font = Math.max(13, Math.round(canvas.width / 65));
      ctx.font = `${font}px Arial`;
      ctx.lineWidth = Math.max(2, canvas.width / 500);
      detections.forEach((d) => {
        const [x1, y1, x2, y2] = d.box.map(
          (v, i) => v * (i % 2 ? canvas.height : canvas.width),
        );
        ctx.strokeStyle = "#e5c779";
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        const text = `${label(d.class_id)} ${Math.round(d.confidence * 100)} %`;
        const width = ctx.measureText(text).width + 10,
          top = Math.max(0, y1 - font - 8);
        const left = Math.min(x1, Math.max(0, canvas.width - width));
        ctx.fillStyle = "#e5c779";
        ctx.fillRect(left, top, width, font + 8);
        ctx.fillStyle = "#232823";
        ctx.fillText(text, left + 5, top + font + 1);
      });
      canvas.toBlob((blob) => {
        if (blob) saveBlob(blob, "annotated-detection.png");
      }, "image/png");
    } catch {
      setError(
        "Could not export the image. Wait for the media to load and try again.",
      );
    }
  }

  function seek(value) {
    if (video.current) {
      video.current.currentTime = value;
      setCurrentTime(value);
    }
  }
  function navigate(value) {
    if (value === "about") {
      if (cameraRunning.current) interrupt();
      video.current?.pause();
    }
    setPage(value);
  }

  return (
    <div className="application">
      <a className="skip-link" href="#workspace">
        Skip to analysis
      </a>
      <header className="header">
        <div className="identity">
          <span className="brackets" aria-hidden="true">
            [ ]
          </span>
          <div>
            <h1>Student Behavior Detection</h1>
            <p>Computer vision project</p>
          </div>
        </div>
        <nav aria-label="Main navigation">
          <button
            aria-current={page === "analysis" ? "page" : undefined}
            onClick={() => navigate("analysis")}
          >
            Analysis
          </button>
          <button
            aria-current={page === "about" ? "page" : undefined}
            onClick={() => navigate("about")}
          >
            About
          </button>
          <a href={`${REPOSITORY}#readme`} target="_blank" rel="noreferrer">
            README ↗
          </a>
        </nav>
      </header>
      {page === "about" ? (
        <About meta={meta} onBack={() => navigate("analysis")} />
      ) : (
        <main id="workspace">
          <div className="toolbar">
            <div className="mode-tabs" aria-label="Media type">
              {MODES.map(([key, title]) => (
                <button
                  key={key}
                  aria-pressed={mode === key}
                  onClick={() => changeMode(key)}
                >
                  {title}
                </button>
              ))}
            </div>
            <span className="mono muted">
              {meta ? "Local processing / YOLOv8" : "Connecting to server…"}
            </span>
          </div>
          {error && (
            <div className="error" role="alert">
              <span>{error}</span>
              <button
                onClick={() => setError("")}
                aria-label="Dismiss error"
              >
                ×
              </button>
            </div>
          )}
          {meta && !meta.model_available && (
            <div className="notice">
              Model not found. Start the server with{" "}
              <code>--weights models/best_legacy_run.pt</code>.
            </div>
          )}
          <div className="work-grid">
            <section
              className="canvas"
              aria-label="Media preview"
              onDragOver={(e) => {
                e.preventDefault();
                if (mode !== "camera") setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragging(false);
                if (mode !== "camera") loadFile(e.dataTransfer.files[0]);
              }}
            >
              <div className="canvas-toolbar">
                <span className="mono filename">
                  {mode === "camera"
                    ? "Local camera"
                    : file?.name || "No media open"}
                </span>
                <label className="check">
                  <input
                    type="checkbox"
                    checked={annotations}
                    onChange={(e) => setAnnotations(e.target.checked)}
                  />
                  Annotations
                </label>
              </div>
              <div className={`stage ${dragging ? "is-dragging" : ""}`}>
                {hasMedia ? (
                  <div
                    className="media"
                    style={{
                      aspectRatio: size
                        ? `${size.width} / ${size.height}`
                        : undefined,
                    }}
                  >
                    {mode === "video" ? (
                      <video
                        ref={video}
                        src={source}
                        playsInline
                        preload="metadata"
                        onLoadedMetadata={(e) => {
                          const v = e.currentTarget;
                          setDuration(v.duration);
                          setDimensions({
                            width: v.videoWidth,
                            height: v.videoHeight,
                          });
                        }}
                        onTimeUpdate={(e) =>
                          setCurrentTime(e.currentTarget.currentTime)
                        }
                        onPlay={() => setPlaying(true)}
                        onPause={() => setPlaying(false)}
                        onEnded={() => setPlaying(false)}
                        onError={() => setMediaError(true)}
                        aria-label="Video preview"
                      />
                    ) : (
                      <img
                        ref={image}
                        src={liveSource}
                        alt={
                          mode === "camera"
                            ? "Latest analyzed camera frame"
                            : "Image loaded for behavior detection"
                        }
                        onLoad={(e) =>
                          setDimensions({
                            width: e.currentTarget.naturalWidth,
                            height: e.currentTarget.naturalHeight,
                          })
                        }
                        onError={() =>
                          setError(
                            "Could not display this image. Choose another file.",
                          )
                        }
                      />
                    )}
                    {annotations && (
                      <div className="overlays">
                        {detections.map((d) => (
                          <button
                            key={d.id}
                            className={`detection ${chosen?.id === d.id ? "selected" : ""}`}
                            style={{
                              left: `${d.box[0] * 100}%`,
                              top: `${d.box[1] * 100}%`,
                              width: `${(d.box[2] - d.box[0]) * 100}%`,
                              height: `${(d.box[3] - d.box[1]) * 100}%`,
                            }}
                            onClick={() => setSelected(d.id)}
                            aria-label={`${label(d.class_id)}, confidence ${Math.round(d.confidence * 100)} percent`}
                            aria-pressed={chosen?.id === d.id}
                          >
                            {chosen?.id === d.id && (
                              <span
                                className={d.box[0] > 0.65 ? "align-right" : ""}
                                style={
                                  d.box[1] < 0.08
                                    ? { top: 0, bottom: "auto" }
                                    : undefined
                                }
                              >
                                {label(d.class_id)} /{" "}
                                {Math.round(d.confidence * 100)}%
                              </span>
                            )}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="empty">
                    <span className="empty-mark" aria-hidden="true">
                      {mode === "video" ? "▷" : mode === "camera" ? "◉" : "+"}
                    </span>
                    <h2>
                      {mode === "camera"
                        ? "Use your camera"
                        : mode === "video"
                          ? "Open a video"
                          : "Open an image"}
                    </h2>
                    <p>
                      {mode === "camera"
                        ? "The camera starts when you choose Start camera."
                        : "Drag a file here or choose one from your computer."}
                    </p>
                    {mode !== "camera" && (
                      <button
                        className="button light"
                        onClick={() => input.current.click()}
                      >
                        {mode === "image" ? "Choose image" : "Choose video"}
                      </button>
                    )}
                    {mode === "image" && meta?.example_available && (
                      <button
                        className="text-button on-dark"
                        onClick={loadExample}
                      >
                        Try the sample image
                      </button>
                    )}
                    <small>
                      {mode === "image"
                        ? "JPG, PNG, WebP, BMP · 25 MB max"
                        : mode === "video"
                          ? "MP4, MOV, AVI, WebM, MKV · 200 MB · 10 minutes"
                          : "Camera frames are not saved"}
                    </small>
                  </div>
                )}
                {busy && (
                  <div className="processing" role="status">
                    <span>{phase}</span>
                    {job && (
                      <>
                        <progress
                          max="1"
                          value={job.progress}
                          aria-label="Video analysis progress"
                        />
                        <span className="mono">
                          {Math.round(job.progress * 100)} % ·{" "}
                          {job.processed_frames} / {job.total_frames || "—"}{" "}
                          frames
                        </span>
                      </>
                    )}
                    <button
                      className="text-button on-dark"
                      onClick={() => {
                        interrupt();
                        setPhase("Analysis canceled.");
                        setJob(null);
                      }}
                    >
                      Cancel
                    </button>
                  </div>
                )}
                {dragging && (
                  <div className="drop-message">
                    Drop the file to open it
                  </div>
                )}
              </div>
              {mediaError && mode === "video" && (
                <p className="media-hint">
                  Preview unavailable for this codec. Analyze the video to convert
                  it to MP4.
                </p>
              )}
              {mode === "video" && source && (
                <div className="player">
                  <button
                    onClick={() => {
                      if (playing) video.current?.pause();
                      else
                        video.current?.play().catch(() => setMediaError(true));
                    }}
                    disabled={mediaError}
                    aria-label={playing ? "Pause video" : "Play video"}
                  >
                    {playing ? "Pause" : "Play"}
                  </button>
                  <input
                    type="range"
                    aria-label="Video position"
                    min="0"
                    max={Number.isFinite(duration) ? duration : 0}
                    step="0.01"
                    value={currentTime}
                    onChange={(e) => seek(Number(e.target.value))}
                  />
                  <output>
                    {formatTime(currentTime)} / {formatTime(duration)}
                  </output>
                </div>
              )}
              <div className="canvas-footer">
                <span>
                  {size
                    ? `${size.width} × ${size.height} px`
                    : "Media preview"}
                </span>
                <span>
                  {cameraActive && cameraMs
                    ? `Round trip ${cameraMs} ms`
                    : current
                      ? `Inference ${current.inference_ms} ms`
                      : "Waiting for analysis"}
                </span>
              </div>
            </section>
            <aside className="inspector" aria-label="Settings and results">
              <section>
                <div className="section-heading">
                  <h2>Source</h2>
                  {mode !== "camera" && file && (
                    <button
                      className="text-button"
                      onClick={() => input.current.click()}
                    >
                      Replace
                    </button>
                  )}
                </div>
                <p className="filename source-name">
                  {mode === "camera"
                    ? "This device’s camera"
                    : file?.name || "No file"}
                </p>
                <p className="mono muted">
                  {mode === "camera"
                    ? cameraActive
                      ? "Session active"
                      : "Camera stopped"
                    : file
                      ? `${(file.size / 1024 / 1024).toFixed(2)} MB${videoResult ? ` · ${formatTime(videoResult.duration)}` : ""}`
                      : "Open a file to get started."}
                </p>
              </section>
              <section>
                <div className="section-heading">
                  <h2>Behaviors</h2>
                  <span className="mono" aria-live="polite">
                    {current ? detections.length : "—"}
                  </span>
                </div>
                <div className="class-list">
                  {counts.map((c) => (
                    <div className="class-row" key={c.id}>
                      <label>
                        <input
                          type="checkbox"
                          checked={activeClasses.includes(c.id)}
                          onChange={(e) =>
                            setActiveClasses((prev) =>
                              e.target.checked
                                ? [...prev, c.id]
                                : prev.filter((x) => x !== c.id),
                            )
                          }
                        />
                        {c.label}
                      </label>
                      <span className="mono">{current ? c.count : "—"}</span>
                    </div>
                  ))}
                </div>
                {mode === "video" && (
                  <p className="small muted top-space">
                    Counts show detections in the current frame.
                  </p>
                )}
              </section>
              <section>
                <div className="section-heading">
                  <label htmlFor="confidence">Minimum confidence</label>
                  <output className="mono">
                    {Math.round(confidence * 100)} %
                  </output>
                </div>
                <input
                  id="confidence"
                  type="range"
                  min="0.1"
                  max="0.95"
                  step="0.05"
                  value={confidence}
                  onChange={(e) => setConfidence(Number(e.target.value))}
                />
                <p className="mono muted">Filter the visible detections.</p>
              </section>
              <section className="selection">
                <h2>{chosen ? label(chosen.class_id) : "No selection"}</h2>
                <p className="mono muted">
                  {chosen
                    ? `Detection ${String(chosen.id + 1).padStart(2, "0")} · confidence ${Math.round(chosen.confidence * 100)} %`
                    : current
                      ? detections.length
                        ? "Select a box in the image."
                        : "No detections match these filters."
                      : "Results will appear here."}
                </p>
              </section>
              <details className="advanced">
                <summary>Advanced settings</summary>
                <label>
                  IoU threshold <output>{iou.toFixed(2)}</output>
                  <input
                    type="range"
                    min="0.1"
                    max="0.9"
                    step="0.05"
                    value={iou}
                    disabled={busy}
                    onChange={(e) => setIou(Number(e.target.value))}
                  />
                </label>
                {mode === "video" && (
                  <label>
                    Frames analyzed per second
                    <select
                      value={sampleFps}
                      disabled={busy}
                      onChange={(e) => setSampleFps(Number(e.target.value))}
                    >
                      <option value="2">2 — fast</option>
                      <option value="4">4 — balanced</option>
                      <option value="8">8 — more detail</option>
                    </select>
                  </label>
                )}
                <p className="small muted">
                  These settings apply to the next analysis.
                </p>
              </details>
              <div className="inspector-actions">
                {cameraActive ? (
                  <button
                    className="button primary"
                    onClick={() => {
                      interrupt();
                      setPhase("Camera stopped.");
                    }}
                  >
                    Stop camera
                  </button>
                ) : (
                  <button
                    className="button primary"
                    disabled={!canAnalyze || busy}
                    onClick={mode === "camera" ? startCamera : analyze}
                  >
                    {busy
                      ? "Processing…"
                      : mode === "image"
                        ? "Analyze image"
                        : mode === "video"
                          ? "Analyze video"
                          : "Start camera"}
                  </button>
                )}
                <p className="small muted" role="status">
                  {phase || "Processed on your computer"}
                </p>
              </div>
            </aside>
          </div>
          <section className="bottom-panel" aria-label="Results and exports">
            <div className="bottom-heading">
              <h2>
                {mode === "camera"
                  ? "Latest analyzed frame"
                  : "Current file"}
              </h2>
              {isFinished && (
                <div className="export-actions">
                  <button onClick={exportImage}>Annotated image ↓</button>
                  {videoResult && (
                    <a href={`/api/videos/${job.id}/annotated.mp4`} download>
                      Annotated video ↓
                    </a>
                  )}
                  <button onClick={() => exportData("csv")}>CSV ↓</button>
                  <button onClick={() => exportData("json")}>JSON ↓</button>
                </div>
              )}
            </div>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Media</th>
                    <th className="desktop-column">Dimensions</th>
                    <th>Status</th>
                    <th>Result</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>{mode === "camera" ? "Camera" : file?.name || "—"}</td>
                    <td className="desktop-column">
                      {size ? `${size.width} × ${size.height}` : "—"}
                    </td>
                    <td>
                      {busy
                        ? "In progress"
                        : isFinished
                          ? "Analyzed"
                          : file
                            ? "Ready"
                            : "—"}
                    </td>
                    <td>
                      {current
                        ? `${detections.length} detection${detections.length === 1 ? "" : "s"}`
                        : "—"}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
            {videoResult && (
              <p className="small muted export-note">
                Analyzed at {videoResult.sample_fps.toFixed(1)} frames/s in{" "}
                {videoResult.processing_seconds.toFixed(1)} s. Boxes stay visible until the
                next sampled frame. The MP4 has no audio and uses the{" "}
                {Math.round(videoResult.export_confidence * 100)}% confidence threshold
                set at analysis time. CSV and JSON exports use the current filters.
              </p>
            )}
          </section>
          <input
            ref={input}
            className="file-input"
            type="file"
            accept={
              mode === "image"
                ? ".jpg,.jpeg,.png,.webp,.bmp"
                : ".mp4,.mov,.avi,.webm,.mkv,.m4v"
            }
            onChange={(e) => {
              loadFile(e.target.files[0]);
              e.target.value = "";
            }}
            aria-label="Open file"
          />
        </main>
      )}
      <video
        ref={webcam}
        className="capture-video"
        autoPlay
        muted
        playsInline
        aria-hidden="true"
      />
      <footer className="footer">
        <span>Mohammed Fakir / Mundiapolis</span>
        <span>
          {meta?.legacy ? "Legacy model · " : ""}Review model predictions
        </span>
      </footer>
    </div>
  );
}

function About({ meta, onBack }) {
  return (
    <main className="about">
      <span className="mono muted">ABOUT / YOLOv8</span>
      <h2>Understanding the results</h2>
      <p>
        This university project detects five categories of visible classroom behavior.
        Review the model’s annotations in context.
      </p>
      <div className="about-columns">
        <section>
          <h3>How to use</h3>
          <ol>
            <li>Open an image or video, or start the camera.</li>
            <li>Run the analysis. The first run loads the model.</li>
            <li>
              Adjust the confidence threshold, hide classes, and select detection boxes.
            </li>
            <li>Export the annotations and results.</li>
          </ol>
        </section>
        <section>
          <h3>Loaded model</h3>
          <p>
            <code>{meta?.model || "Connecting"}</code>
          </p>
          <p>
            {meta?.legacy
              ? "Weights from the original training run. That experiment’s scores were affected by splitting the dataset after augmentation."
              : "See the README for training and evaluation details."}
          </p>
        </section>
      </div>
      <section>
        <h3>Detection limits</h3>
        <p>
          “Focused,” “Distracted,” and “Sleeping” are labels learned from visual cues.
          They do not measure a person’s mental state, learning, or ability.
          The model was trained on a single classroom setup and can make mistakes.
        </p>
      </section>
      <section>
        <h3>Media and local processing</h3>
        <p>
          The server discards images and camera frames after analysis. It stores up to
          eight video analyses locally during the server session and clears idle
          results after an hour on the next request. Video exports are encoded as
          MP4 without audio, with a maximum size of 1280 pixels.
        </p>
      </section>
      <div className="about-links">
        <button className="button primary" onClick={onBack}>
          Back to analysis
        </button>
        <a href={`${REPOSITORY}#readme`} target="_blank" rel="noreferrer">
          Read the documentation ↗
        </a>
      </div>
    </main>
  );
}
