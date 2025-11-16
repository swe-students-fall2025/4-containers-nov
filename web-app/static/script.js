console.log("🟢 script.js loaded!");

let capturing = false;
let muted = false;

const toggleBtn = document.getElementById("toggle-btn");
const muteBtn = document.getElementById("mute-btn");
const gestureImg = document.getElementById("gesture-img");

// Gesture → image mapping
const gestureImages = {
  palm: "palm.png",
  fist: "fist.png",
  like: "like.png",
  stop: "stop.png",
  ok: "ok.png",
  one: "one.png",
  two_up: "two_up.png",
  three: "three.png",
};

const gestureAudios = {
  palm: "sponge_bob.mp3",
  fist: "bom.mp3",
  like: "rizz.mp3",
  stop: "error.mp3",
  ok: "playme.mp3",
  one: "android_beep.mp3",
  two_up: "uwu.mp3",
  three: "among_us.mp3",
};

let previousGesture = null;

toggleBtn.addEventListener("click", async () => {
  capturing = !capturing;

  await fetch("/api/control", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled: capturing }),
  });

  toggleBtn.textContent = capturing ? "Stop Capture" : "Start Capture";
});

muteBtn.addEventListener("click", () => {
  muted = !muted;
  muteBtn.textContent = muted ? "🔇" : "🔊";
});

let lastGestureTime = Date.now();

setInterval(async () => {
  console.log("polling...");

  if (!capturing) return;

  const res = await fetch("/api/latest");
  const data = await res.json();

  if (data.gesture) {
    const gesture = data.gesture;
    lastGestureTime = Date.now();

    // Update image
    gestureImg.src = `/static/images/${gestureImages[gesture]}`;

    // Play sound: only when gesture changes
    if (!muted && gesture !== previousGesture) {
      new Audio(`/static/audios/${gestureAudios[gesture]}`).play();
    }

    previousGesture = gesture;
  } else if (Date.now() - lastGestureTime > 3000) {
    // after 3 seconds of no change, show thinking
    gestureImg.src = "/static/images/thinking.png";
  }
}, 1000);
