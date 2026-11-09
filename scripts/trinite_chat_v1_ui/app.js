"use strict";
(() => {
  const messages = document.getElementById("messages");
  const form = document.getElementById("chat-form");
  const prompt = document.getElementById("prompt");
  const status = document.getElementById("status");
  const send = document.getElementById("send");
  const remember = document.getElementById("remember");
  const loadButton = document.getElementById("load-saved");
  const forgetButton = document.getElementById("forget-saved");
  const history = [];
  const encoder = new TextEncoder();
  const KEY = "trinite.chat.saved-session.v1";
  const MAX_MESSAGE_BYTES = 8192;
  const MAX_REQUEST_BYTES = 16384;
  const MAX_HISTORY = 16;
  const byteLength = (value) => encoder.encode(value).length;
  const payload = (message) => ({schema: "trinite.chat.request.v1", message, history});
  let sessionId = null;
  let busy = false;

  function savedId() {
    try {
      const value = window.localStorage.getItem(KEY);
      return typeof value === "string" && /^[0-9a-f]{32}$/.test(value) ? value : null;
    } catch (_) {
      return null;
    }
  }
  function canSaveToken() {
    try {
      const test = KEY + ".probe";
      window.localStorage.setItem(test, "1");
      window.localStorage.removeItem(test);
      return true;
    } catch (_) {
      return false;
    }
  }
  function rememberToken(value) {
    try {
      window.localStorage.setItem(KEY, value);
      return true;
    } catch (_) {
      return false;
    }
  }
  function clearToken() {
    try {
      window.localStorage.removeItem(KEY);
      return true;
    } catch (_) {
      return false;
    }
  }
  function line(role, content, classification, fingerprint) {
    const article = document.createElement("article");
    article.className = "message " + role;
    const tag = document.createElement("span");
    tag.textContent = role === "user" ? "YOU" : "TRINITE · " + classification;
    const body = document.createElement("p");
    body.textContent = content; // Never interpret teacher/user text as HTML.
    article.append(tag, body);
    if (fingerprint) {
      const meta = document.createElement("small");
      meta.textContent = "request SHA-256: " + fingerprint;
      article.append(meta);
    }
    messages.append(article);
    article.scrollIntoView({block: "nearest"});
  }
  function pruneHistory(nextMessage) {
    while (history.length &&
           (history.length > MAX_HISTORY ||
            history.some((entry) => byteLength(entry.content) > MAX_MESSAGE_BYTES) ||
            byteLength(JSON.stringify(payload(nextMessage))) > MAX_REQUEST_BYTES)) {
      history.splice(0, 2);
    }
  }
  function setBusy(value) {
    busy = value;
    send.disabled = value;
    loadButton.disabled = value;
    forgetButton.disabled = value;
    remember.disabled = value;
  }
  async function sessionCommand(action, id) {
    const response = await fetch("/api/session", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        schema: "trinite.chat.session.request.v1", action, session_id: id
      })
    });
    const result = await response.json();
    if (!response.ok || result.schema !== "trinite.chat.session.result.v1" ||
        result.status !== "ok") {
      throw new Error("Session request rejected");
    }
    return result;
  }
  async function loadSaved() {
    if (busy) return;
    const id = savedId();
    if (!id) {
      status.textContent = "No saved session token on this device.";
      remember.checked = false;
      return;
    }
    setBusy(true);
    try {
      const result = await sessionCommand("load", id);
      if (!Array.isArray(result.messages) || result.session_id !== id) {
        throw new Error("Invalid saved session");
      }
      messages.replaceChildren();
      history.length = 0;
      for (const item of result.messages) {
        if (!["user", "assistant"].includes(item.role) ||
            typeof item.content !== "string" ||
            typeof item.classification !== "string") {
          throw new Error("Invalid saved message");
        }
        line(item.role, item.content, item.classification, null);
      }
      sessionId = id;
      remember.checked = true;
      status.textContent = "Saved SQLite session loaded (latest eight exchanges).";
    } catch (_) {
      sessionId = null;
      remember.checked = false;
      status.textContent = "Saved session unavailable; no changes made to the database.";
    } finally {
      setBusy(false);
    }
  }
  remember.addEventListener("change", async () => {
    if (remember.checked) {
      if (!canSaveToken()) {
        remember.checked = false;
        status.textContent = "Browser localStorage unavailable; SQLite opt-in disabled.";
      } else if (savedId()) {
        await loadSaved();
      } else {
        sessionId = null;
        history.length = 0;
        messages.replaceChildren();
        status.textContent = "SQLite requested. A successful exchange will create a saved session.";
      }
    } else {
      sessionId = null;
      history.length = 0;
      messages.replaceChildren();
      status.textContent = "In-tab mode. Any previous saved session remains until explicitly forgotten.";
    }
  });
  loadButton.addEventListener("click", loadSaved);
  forgetButton.addEventListener("click", async () => {
    if (busy) return;
    const id = savedId();
    if (!id) {
      status.textContent = "No saved session token to forget.";
      return;
    }
    setBusy(true);
    try {
      await sessionCommand("delete", id);
      if (!clearToken()) throw new Error("Could not clear local session token");
      sessionId = null;
      remember.checked = false;
      history.length = 0;
      messages.replaceChildren();
      status.textContent = "Saved session logically deleted. Backups and filesystem remnants may remain.";
    } catch (_) {
      status.textContent = "Could not forget session; verify local SQLite is enabled.";
    } finally {
      setBusy(false);
    }
  });
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (busy) return;
    const text = prompt.value.trim();
    if (!text) return;
    if (byteLength(text) > MAX_MESSAGE_BYTES) {
      status.textContent = "Message exceeds the 8 KiB UTF-8 limit.";
      return;
    }
    const saved = remember.checked;
    if (saved && !canSaveToken()) {
      status.textContent = "No browser token storage; refusing to create an unreachable saved session.";
      return;
    }
    if (!saved) {
      pruneHistory(text);
      if (byteLength(JSON.stringify(payload(text))) > MAX_REQUEST_BYTES) {
        status.textContent = "Request exceeds the 16 KiB UTF-8 limit.";
        return;
      }
    }
    const request = saved
      ? {schema: "trinite.chat.persist.request.v1", message: text, session_id: sessionId}
      : payload(text);
    if (byteLength(JSON.stringify(request)) > MAX_REQUEST_BYTES) {
      status.textContent = "Request exceeds the 16 KiB limit.";
      return;
    }
    setBusy(true);
    status.textContent = saved ? "Processing with opt-in local SQLite..." : "Processing locally...";
    try {
      const reply = await fetch("/api/chat", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(request)
      });
      const result = await reply.json();
      if (!result || result.schema !== "trinite.chat.result.v1" ||
          !["verified_exact", "model_generated_unverified", "blocked", "failed"].includes(result.status)) {
        throw new Error("Invalid server response");
      }
      if (saved && ["verified_exact", "model_generated_unverified"].includes(result.status)) {
        if (result.persisted !== true || !/^[0-9a-f]{32}$/.test(result.session_id || "")) {
          throw new Error("Persistence not confirmed");
        }
        if (!rememberToken(result.session_id)) {
          throw new Error("Could not retain saved session token");
        }
        sessionId = result.session_id;
      }
      line("user", text, "", null);
      let answer = result.text || "No answer: " + (result.reason || "blocked");
      if (result.status === "verified_exact" && result.exact && result.exact.verified === true) {
        answer = result.text;
      }
      line("assistant", answer, result.status, result.receipt && result.receipt.request_sha256);
      if (!saved && (result.status === "verified_exact" || result.status === "model_generated_unverified")) {
        if (byteLength(answer) <= MAX_MESSAGE_BYTES) {
          history.push({role: "user", content: text}, {role: "assistant", content: answer});
          pruneHistory("");
        }
      }
      prompt.value = "";
      status.textContent = saved && result.persisted === true
        ? "Saved locally in SQLite. Unverified teacher prose is not a proof."
        : result.status === "verified_exact"
          ? "Exact arithmetic independently checked."
          : result.status === "model_generated_unverified"
            ? "Generated response. NOT independently verified."
            : "Request " + result.status + ": " + (result.reason || "unknown");
    } catch (_) {
      status.textContent = "Request failed. No automatic retry or cloud fallback; storage may need inspection.";
    } finally {
      setBusy(false);
      prompt.focus();
    }
  });
  document.getElementById("clear").addEventListener("click", () => {
    if (busy) return;
    history.length = 0;
    sessionId = null;
    remember.checked = false;
    messages.replaceChildren();
    status.textContent = "Display cleared. Previously saved SQLite sessions remain until Forget.";
  });
})();
