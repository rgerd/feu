import { Location } from "./proto/proto.js";
import { Application, FederatedPointerEvent, Graphics } from "pixi.js";
import World from "./world";

const outputElement = document.createElement("pre");
document.body.appendChild(outputElement);

function print(output: string, concat: boolean = false) {
  outputElement.innerText = concat
    ? `${outputElement.innerText}\n${output}`
    : output;
}
print("Waiting...");

const socket = new WebSocket("ws://localhost:8765");

function onSocketOpen() {
  const message = Location.create({
    x: 40,
    y: 23,
  });
  const locationBuffer = Location.encode(message).finish();
  print(`Sending ${locationBuffer.length} bytes...`, true);
  socket.send(locationBuffer);
}

async function onSocketMessage(message: MessageEvent<Blob>) {
  const arrayBuffer = await message.data.arrayBuffer();
  const reader = new Uint8Array(arrayBuffer);
  const location = Location.decode(reader);
  print(JSON.stringify(location.toJSON()), true);
  setTimeout(() => {
    socket.send(Location.encode(location).finish());
  }, 1000);
}

const MIN_SQRD_DISTANCE = 40;
const MAX_POINTS = 100;
const points = new Array<{ x: number; y: number }>();
let lastPoint: { x: number; y: number } | null = null;
let pointStartIdx = 0;
let pointEndIdx = 0;

function onPointerMove(event: FederatedPointerEvent) {
  if (lastPoint) {
    const dx = lastPoint.x - event.clientX;
    const dy = lastPoint.y - event.clientY;
    if (dx * dx + dy * dy < MIN_SQRD_DISTANCE) {
      return;
    }
  }
  lastPoint = {
    x: event.clientX,
    y: event.clientY,
  };
  points[pointEndIdx] = lastPoint;
  pointEndIdx = (pointEndIdx + 1) % MAX_POINTS;
  if (pointEndIdx == pointStartIdx) pointStartIdx = (pointStartIdx + 1) % MAX_POINTS;
}

socket.addEventListener("open", onSocketOpen);
socket.addEventListener("message", onSocketMessage);
window.addEventListener("load", () => {
  new World();
});
