console.log("🟢 script.js loaded!");

let capturing = false;
let muted = false;

const toggleBtn = document.getElementById("toggle-btn");
const muteBtn = document.getElementById("mute-btn");
const gestureImg = document.getElementById("gesture-img");

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
  if (!capturing) return;

  // ========== 更新 meme ==========
  const res = await fetch("/api/latest");
  const data = await res.json();

  if (data.gesture) {
    const gesture = data.gesture;
    lastGestureTime = Date.now();

    gestureImg.src = `/static/images/${gestureImages[gesture]}`;

    if (!muted && gesture !== previousGesture) {
      new Audio(`/static/audios/${gestureAudios[gesture]}`).play();
    }

    previousGesture = gesture;
  } else if (Date.now() - lastGestureTime > 3000) {
    gestureImg.src = "/static/images/thinking.png";
  }

  // ========== 更新 Latest Event ==========
  const res2 = await fetch("/api/latest_full");
  const latest = await res2.json();

  if (latest.exists) {
    document.getElementById("latest-gesture").textContent = latest.gesture;
    document.getElementById("latest-meta").textContent = `Confidence: ${
      latest.confidence?.toFixed(2) ?? "-"
    } • Hand: ${latest.handedness}`;
    document.getElementById("latest-time").textContent =
      latest.timestamp_display;
  }

  // ========== 更新 Total / Breakdown / Recent ==========
  const dashRes = await fetch("/api/dashboard");
  const dash = await dashRes.json();

  // total count
  const totalEl = document.getElementById("total-count");
  if (totalEl) totalEl.textContent = dash.total_count;

  // gesture breakdown
  const breakdownEl = document.getElementById("gesture-breakdown");
  if (breakdownEl) {
    breakdownEl.innerHTML = dash.gesture_stats
      .map(
        (g) => `
        <li class="gesture-list-item">
          <span class="gesture-name">${g._id || "Unknown"}</span>
          <span class="gesture-count">${g.count}×</span>
        </li>`
      )
      .join("");
  }

  // recent events
  const tableEl = document.getElementById("recent-events-body");
  if (tableEl) {
    tableEl.innerHTML = dash.recent
      .map(
        (e) => `
        <tr>
          <td>${e.timestamp_display}</td>
          <td>${e.gesture}</td>
          <td>${e.confidence ? e.confidence.toFixed(2) : "-"}</td>
          <td>${e.handedness}</td>
        </tr>`
      )
      .join("");
  }
}, 1000);
