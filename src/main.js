const USER_ID = "user_" + Math.random().toString(36).slice(2, 8);

const messagesEl = document.getElementById("messages");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("send-btn");
const boardEl = document.getElementById("board");

// event listeners 

sendBtn.addEventListener("click", send);
inputEl.addEventListener("keydown", (e) => { if (e.key === "Enter") send(); });

// send message 

async function send() {
  const text = inputEl.value.trim();
  if (!text) return;

  inputEl.value = "";
  addMsg(text, "you");
  setLoading(true);

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, user_id: USER_ID }),
    });
    const data = await res.json();
    addMsg(data.response, "ai", data.path);
    if (data.topic_cluster) addToBoard(text, data.topic_cluster, data.path);
    if (data.session_id) currentSessionId = data.session_id;
  } catch (err) {
    addMsg("Could not reach backend.", "ai");
    console.error(err);
  }

  setLoading(false);
}

// ── UI helpers ────────────────────────────────────────────────────────────────

function addMsg(text, role, path) {
  const el = document.createElement("div");
  el.className = "msg " + role;
  el.textContent = text;

  if (role === "ai" && path && path !== "intake") {
    const tag = document.createElement("div");
    tag.className = "tag" + (path === "faculty" ? " faculty" : "");
    tag.textContent = path === "faculty" ? "→ sent to faculty" : "→ AI answered";
    el.appendChild(tag);
  }

  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function addToBoard(question, cluster, path) {
  const empty = boardEl.querySelector(".board-empty");
  if (empty) empty.remove();

  let clusterEl = document.getElementById("cluster-" + cluster);
  if (!clusterEl) {
    clusterEl = document.createElement("div");
    clusterEl.className = "cluster";
    clusterEl.id = "cluster-" + cluster;
    clusterEl.innerHTML = `<div class="cluster-name">${cluster}</div>`;
    boardEl.appendChild(clusterEl);
  }

  const card = document.createElement("div");
  card.className = "q-card mine";
  const isFaculty = path === "faculty";
  card.innerHTML = `
    ${question}
    <div class="q-status ${isFaculty ? "faculty" : ""}">${isFaculty ? "→ faculty" : "waiting"}</div>
  `;
  clusterEl.appendChild(card);
}

function setLoading(on) {
  sendBtn.disabled = on;
  inputEl.disabled = on;
}

// poll for answered questions every 5 seconds
let currentSessionId = null;

setInterval(async () => {
  if (!currentSessionId) return;
  try {
    const res = await fetch(`/answered/${currentSessionId}`);
    const data = await res.json();
    data.questions.forEach(q => updateBoardStatus(q.question, q.answer));
  } catch {}
}, 5000);

function updateBoardStatus(question, answer) {
  // find card with matching question text and update status
  document.querySelectorAll(".q-card").forEach(card => {
    if (card.textContent.includes(question)) {
      const status = card.querySelector(".q-status");
      if (status && status.textContent === "waiting") {
        status.textContent = "answered";
        status.className = "q-status answered";
        card.title = answer; // show answer on hover
      }
    }
  });
}