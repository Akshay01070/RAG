/* ================================================================
   Indecimal RAG Assistant — Client Script
   Handles query submission, rendering, and source accordions.
   ================================================================ */

const API_URL = "/api/query";

const form         = document.getElementById("query-form");
const input        = document.getElementById("query-input");
const sendBtn      = document.getElementById("send-btn");
const messagesDiv  = document.getElementById("messages");
const welcomeArea  = document.getElementById("welcome-area");
const messagesArea = document.getElementById("messages-area");
const loadBarFill  = document.getElementById("load-bar-fill");
const newChatBtn   = document.getElementById("new-chat-btn");

// -------------------------------------------------------------------
// Bento card & sample query buttons
// -------------------------------------------------------------------
document.querySelectorAll(".bento-card").forEach((btn) => {
    btn.addEventListener("click", () => {
        input.value = btn.dataset.query;
        form.dispatchEvent(new Event("submit"));
    });
});

// -------------------------------------------------------------------
// New Chat — reset UI
// -------------------------------------------------------------------
newChatBtn.addEventListener("click", (e) => {
    e.preventDefault();
    messagesDiv.innerHTML = "";
    welcomeArea.classList.remove("hidden");
    messagesArea.classList.remove("active");
    if (loadBarFill) loadBarFill.style.width = "12%";
});

// -------------------------------------------------------------------
// Form submit
// -------------------------------------------------------------------
form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = input.value.trim();
    if (!query) return;

    // Switch to chat mode
    welcomeArea.classList.add("hidden");
    messagesArea.classList.add("active");

    // Add user message
    appendMessage("user", query);
    input.value = "";
    sendBtn.disabled = true;

    // Animate system load
    if (loadBarFill) loadBarFill.style.width = "70%";

    // Show typing indicator
    const typingEl = appendTyping();

    try {
        const res = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query }),
        });
        const data = await res.json();
        typingEl.remove();

        if (data.error) {
            appendMessage("bot", `⚠️ ${data.error}`, []);
        } else {
            appendMessage("bot", data.answer, data.sources || []);
        }
    } catch (err) {
        typingEl.remove();
        appendMessage("bot", `⚠️ Network error — is the server running?\n\n${err.message}`, []);
    } finally {
        sendBtn.disabled = false;
        if (loadBarFill) loadBarFill.style.width = "12%";
        input.focus();
    }
});

// -------------------------------------------------------------------
// Render helpers
// -------------------------------------------------------------------

function appendMessage(role, text, sources = []) {
    const wrapper = document.createElement("div");
    wrapper.classList.add("msg", role);

    const avatar = document.createElement("div");
    avatar.classList.add("msg-avatar");
    avatar.textContent = role === "user" ? "U" : "AI";

    const content = document.createElement("div");
    content.classList.add("msg-content");

    const bubble = document.createElement("div");
    bubble.classList.add("msg-bubble");
    bubble.innerHTML = renderMarkdown(text);

    content.appendChild(bubble);

    // Sources accordion for bot messages
    if (role === "bot" && sources.length > 0) {
        const toggleBtn = document.createElement("button");
        toggleBtn.classList.add("sources-toggle");
        toggleBtn.innerHTML = `
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="6 9 12 15 18 9"/>
            </svg>
            ${sources.length} source chunks retrieved
        `;

        const panel = document.createElement("ul");
        panel.classList.add("sources-panel");

        sources.forEach((s) => {
            const li = document.createElement("li");
            li.classList.add("source-chip");
            li.innerHTML = `
                <div class="source-chip-header">
                    <span class="source-label">${escapeHtml(s.source)}</span>
                    <span class="source-score">Score: ${s.score.toFixed(4)}</span>
                </div>
                <div>${escapeHtml(s.text)}</div>
            `;
            panel.appendChild(li);
        });

        toggleBtn.addEventListener("click", () => {
            toggleBtn.classList.toggle("open");
            panel.classList.toggle("open");
        });

        content.appendChild(toggleBtn);
        content.appendChild(panel);
    }

    wrapper.appendChild(avatar);
    wrapper.appendChild(content);
    messagesDiv.appendChild(wrapper);
    scrollToBottom();
}

function appendTyping() {
    const wrapper = document.createElement("div");
    wrapper.classList.add("msg", "bot");

    const avatar = document.createElement("div");
    avatar.classList.add("msg-avatar");
    avatar.textContent = "AI";

    const content = document.createElement("div");
    content.classList.add("msg-content");

    const bubble = document.createElement("div");
    bubble.classList.add("msg-bubble", "typing-indicator");
    bubble.innerHTML = `<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>`;

    content.appendChild(bubble);
    wrapper.appendChild(avatar);
    wrapper.appendChild(content);
    messagesDiv.appendChild(wrapper);
    scrollToBottom();
    return wrapper;
}

// -------------------------------------------------------------------
// Minimal Markdown → HTML
// -------------------------------------------------------------------
function renderMarkdown(text) {
    let html = escapeHtml(text);

    // Bold
    html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    // Italic
    html = html.replace(/\*(.*?)\*/g, "<em>$1</em>");
    // Inline code
    html = html.replace(/`(.*?)`/g, "<code>$1</code>");

    // Unordered lists (lines starting with - or •)
    html = html.replace(/^[\-•]\s+(.+)$/gm, "<li>$1</li>");
    html = html.replace(/((?:<li>.*<\/li>\n?)+)/g, "<ul>$1</ul>");

    // Ordered lists (lines starting with number.)
    html = html.replace(/^\d+\.\s+(.+)$/gm, "<li>$1</li>");

    // Paragraphs
    html = html.replace(/\n{2,}/g, "</p><p>");
    html = "<p>" + html + "</p>";
    html = html.replace(/<p>\s*<\/p>/g, "");

    return html;
}

function escapeHtml(str) {
    const el = document.createElement("span");
    el.textContent = str;
    return el.innerHTML;
}

function scrollToBottom() {
    requestAnimationFrame(() => {
        messagesArea.scrollTop = messagesArea.scrollHeight;
    });
}

// Allow Enter to submit
input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        form.dispatchEvent(new Event("submit"));
    }
});
