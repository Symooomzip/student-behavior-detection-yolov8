import { test } from "node:test";
import assert from "node:assert/strict";
import {
  frameAt,
  visibleDetections,
  resultsCsv,
  formatTime,
} from "./results.js";

test("seeking selects the last sample at or before the playhead, including exact boundaries", () => {
  const frames = [{ time: 0 }, { time: 0.25 }, { time: 0.5 }];
  assert.equal(frameAt(frames, 0.249), frames[0]);
  assert.equal(frameAt(frames, 0.25), frames[1]);
  assert.equal(frameAt(frames, 100), frames[2]);
  assert.equal(frameAt([], 0), null);
});
test("threshold and class visibility combine, while a zero detection result stays valid", () => {
  const data = [
    { class_id: 0, confidence: 0.25 },
    { class_id: 1, confidence: 0.8 },
  ];
  assert.deepEqual(visibleDetections(data, 0.25, [0]), [data[0]]);
  assert.deepEqual(visibleDetections(data, 0.3, [0]), []);
  assert.deepEqual(visibleDetections(undefined, 0.1, [0]), []);
});
test("CSV includes sample timestamps and normalized box coordinates", () => {
  const csv = resultsCsv([
    {
      time: 2.5,
      frame: 50,
      detections: [
        { class_name: "phone", confidence: 0.9, box: [0.1, 0.2, 0.4, 0.8] },
      ],
    },
  ]);
  assert.ok(csv.endsWith("2.5,50,phone,0.9,0.1,0.2,0.4,0.8"));
  assert.equal(formatTime(Infinity), "00:00");
  assert.equal(formatTime(125), "02:05");
});
