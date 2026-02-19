// Theme
function applyStoredTheme() {
    const theme = localStorage.getItem("theme") || "light";
    document.body.className = theme;
    document.getElementById("darkmode-toggle").textContent =
        theme === "dark" ? "☀️ Light Mode" : "🌙 Dark Mode";
}
applyStoredTheme();

document.getElementById("darkmode-toggle").onclick = function() {
    const isDark = document.body.classList.contains("dark");
    const newTheme = isDark ? "light" : "dark";
    document.body.className = newTheme;
    localStorage.setItem("theme", newTheme);
    document.getElementById("darkmode-toggle").textContent =
        newTheme === "dark" ? "☀️ Light Mode" : "🌙 Dark Mode";
};

// Wait until DOM is fully loaded before binding mode buttons
document.addEventListener("DOMContentLoaded", () => {
    const sendBtn = document.getElementById("send-btn");
    const promptInput = document.getElementById("prompt");
    const chat = document.getElementById("chat");

    // --- Mode buttons ---
    document.querySelectorAll(".mode-btn").forEach(btn => {
        btn.onclick = function() {
            const mode = this.dataset.mode;
            fetch("/set_mode", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ mode })
            })
            .then(res => res.json())
            .then(data => {
                document.getElementById("llm-mode").textContent = data.mode;
            })
            .catch(err => console.error("Error setting LLM mode:", err));
        };
    });

    // --- Clear chat ---
    document.getElementById("clear-chat").onclick = function() {
        fetch("/clear_chat", { method: "POST" }).then(() => location.reload());
    };

    // --- Helper functions ---
    function setSendingState(isSending) {
        sendBtn.disabled = isSending;
        promptInput.disabled = isSending;
        if (isSending) sendBtn.classList.add("disabled");
        else sendBtn.classList.remove("disabled");
    }

    function showTypingIndicator() {
        const typingEl = document.createElement("div");
        typingEl.id = "typing-indicator";
        typingEl.className = "message assistant typing";
        typingEl.innerHTML = "🤖 <span class='label'>AI Assistant:</span> <span class='dots'><span>.</span><span>.</span><span>.</span></span>";
        chat.appendChild(typingEl);
        chat.scrollTop = chat.scrollHeight;
    }

    function removeTypingIndicator() {
        const el = document.getElementById("typing-indicator");
        if (el) el.remove();
    }

    // --- Send message ---
    async function sendMessage() {
        const prompt = promptInput.value.trim();
        if (!prompt) return;

        // Add user message
        const userDiv = document.createElement("div");
        userDiv.className = "message user";
        userDiv.textContent = "🧑 " + prompt;
        chat.appendChild(userDiv);
        chat.scrollTop = chat.scrollHeight;

        promptInput.value = "";
        setSendingState(true);
        showTypingIndicator();

        try {
            const res = await fetch("/send_message", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ prompt })
            });

            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let assistantDiv = document.createElement("div");
            assistantDiv.className = "message assistant";
            assistantDiv.innerHTML = "🤖 <span class='label'>AI Assistant:</span> ";
            chat.appendChild(assistantDiv);
            chat.scrollTop = chat.scrollHeight;

            let responseText = "";
            removeTypingIndicator();

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                const chunk = decoder.decode(value, { stream: true });
                responseText += chunk;
                assistantDiv.innerHTML = "🤖 <span class='label'>AI Assistant:</span> " + responseText;
                chat.scrollTop = chat.scrollHeight;
            }
        } catch (err) {
            console.error("Error sending message:", err);
        } finally {
            setSendingState(false);
        }
    }

    sendBtn.addEventListener("click", sendMessage);

    promptInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
});
