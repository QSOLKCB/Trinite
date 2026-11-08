"use strict";
(() => {
  const messages = document.getElementById("messages");
  const form = document.getElementById("chat-form");
  const prompt = document.getElementById("prompt");
  const status = document.getElementById("status");
  const send = document.getElementById("send");
  const history = [];
  let busy = false;

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

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (busy) return;
    const text = prompt.value.trim();
    if (!text) return;
    busy = true;
    send.disabled = true;
    status.textContent = "Processing locally...";
    try {
      const reply = await fetch("/api/chat", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({schema: "trinite.chat.request.v1", message: text, history}),
      });
      const result = await reply.json();
      if (!result || result.schema !== "trinite.chat.result.v1" ||
          !["verified_exact", "model_generated_unverified", "blocked", "failed"].includes(result.status)) {
        throw new Error("Invalid server response");
      }
      line("user", text, "", null);
      let answer = result.text || "No answer: " + (result.reason || "blocked");
      if (result.status === "verified_exact" && result.exact && result.exact.verified === true) {
        answer = result.text;
      }
      line("assistant", answer, result.status, result.receipt && result.receipt.request_sha256);
      if (result.status === "verified_exact" || result.status === "model_generated_unverified") {
        history.push({role: "user", content: text}, {role: "assistant", content: answer});
        if (history.length > 16) history.splice(0, 2);
      }
      prompt.value = "";
      status.textContent = result.status === "verified_exact"
        ? "Exact arithmetic independently checked."
        : result.status === "model_generated_unverified"
          ? "Generated response. NOT independently verified."
          : "Request " + result.status + ": " + (result.reason || "unknown");
    } catch (_) {
      status.textContent = "Request failed. No automatic retry or cloud fallback.";
    } finally {
      send.disabled = false;
      busy = false;
      prompt.focus();
    }
  });
  document.getElementById("clear").addEventListener("click", () => {
    if (busy) return;
    history.length = 0;
    messages.replaceChildren();
    status.textContent = "Conversation cleared from browser-tab memory.";
  });
})();
