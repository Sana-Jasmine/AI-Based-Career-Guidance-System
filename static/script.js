
function appendMessage(sender, text, cssClass) {
  const chatArea = document.getElementById("chatArea");

  const p = document.createElement("p");
  p.className = "chat-msg " + cssClass;

  
  p.textContent = sender + ": " + text;

  chatArea.appendChild(p);
  chatArea.scrollTop = chatArea.scrollHeight;
  return p; 
}


function sendMsg() {
  const input = document.getElementById("msg");
  const msg   = input.value.trim();

  if (msg === "") return;

  
  appendMessage("You", msg, "user-msg");

  
  input.value = "";

  
  const typingEl = appendMessage("Bot", "Typing...", "bot-msg typing-msg");

  
  fetch("/chat", {
    method:  "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body:    "message=" + encodeURIComponent(msg),
  })
    .then(function(res) {
      if (!res.ok) throw new Error("Server error: " + res.status);
      return res.json();
    })
    .then(function(data) {
      
      typingEl.remove();
      appendMessage("Bot", data.response, "bot-msg");
    })
    .catch(function() {
      
      typingEl.remove();
      appendMessage("Bot", "Something went wrong. Please try again.", "bot-msg");
    });
}


document.addEventListener("DOMContentLoaded", function () {
  const input = document.getElementById("msg");
  if (input) {
    input.addEventListener("keydown", function (e) {
      if (e.key === "Enter") sendMsg();
    });
  }
});