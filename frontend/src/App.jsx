import { useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./App.css";

const API_BASE = "http://127.0.0.1:8000";

export default function App() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:
        "Hi! Upload your Open Enrollment PDFs on the left or ask a question below to compare deductibles, copays, HSA matches, and out-of-pocket limits.",
      sources: [],
    },
  ]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [documents, setDocuments] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState("");
  const chatEndRef = useRef(null);

  useEffect(() => {
    fetchDocuments();
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function fetchDocuments() {
    try {
      const res = await fetch(`${API_BASE}/documents`);
      if (res.ok) {
        const data = await res.json();
        setDocuments(data.documents || []);
      }
    } catch {
      // Backend might still be starting up
    }
  }

  async function handleFileUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setUploadStatus(`Uploading & indexing ${file.name}...`);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Upload failed");

      setDocuments(data.documents || []);
      setUploadStatus(`Indexed ${file.name} successfully.`);
    } catch (err) {
      setUploadStatus(`Error: ${err.message}`);
    } finally {
      setUploading(false);
      e.target.value = "";
    }
  }

  async function handleSend(e) {
    e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || isStreaming) return;

    const historyPayload = messages
      .filter((m) => m.content)
      .map(({ role, content }) => ({ role, content }));

    setMessages((prev) => [
      ...prev,
      { role: "user", content: trimmed },
      { role: "assistant", content: "", sources: [] },
    ]);
    setInput("");
    setIsStreaming(true);

    try {
      const res = await fetch(`${API_BASE}/ask/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify({
          question: trimmed,
          chat_history: historyPayload,
        }),
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Request failed");
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split("\n\n");
        buffer = events.pop() || "";

        for (const rawEvent of events) {
          const line = rawEvent.trim();
          if (!line.startsWith("data: ")) continue;

          const parsed = JSON.parse(line.slice(6));

          if (parsed.type === "token") {
            setMessages((prev) => {
              const updated = [...prev];
              const last = updated[updated.length - 1];
              updated[updated.length - 1] = {
                ...last,
                content: last.content + parsed.content,
              };
              return updated;
            });
          } else if (parsed.type === "sources") {
            setMessages((prev) => {
              const updated = [...prev];
              updated[updated.length - 1] = {
                ...updated[updated.length - 1],
                sources: parsed.sources,
              };
              return updated;
            });
          } else if (parsed.type === "error") {
            throw new Error(parsed.message);
          }
        }
      }
    } catch (err) {
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: "assistant",
          content: `Error: ${err.message}`,
          sources: [],
        };
        return updated;
      });
    } finally {
      setIsStreaming(false);
    }
  }

  return (
    <div className="layout">
      <aside className="sidebar">
        <h2>OE Documents</h2>
        <label className={`upload-btn ${uploading ? "disabled" : ""}`}>
          {uploading ? "Indexing PDF..." : "+ Upload OE PDF"}
          <input
            type="file"
            accept=".pdf"
            onChange={handleFileUpload}
            disabled={uploading}
            hidden
          />
        </label>
        {uploadStatus && <p className="status-text">{uploadStatus}</p>}

        <ul className="doc-list">
          {documents.length === 0 ? (
            <li className="empty">No PDFs indexed yet.</li>
          ) : (
            documents.map((doc) => <li key={doc}>📄 {doc}</li>)
          )}
        </ul>

        <button
          className="clear-btn"
          onClick={() =>
            setMessages([
              {
                role: "assistant",
                content:
                  "Chat cleared. Ask another question about your medical plans.",
                sources: [],
              },
            ])
          }
        >
          Clear Chat
        </button>
      </aside>

      <main className="chat-area">
        <header className="chat-header">
          <h1>Open Enrollment Plan Advisor</h1>
        </header>

        <div className="messages">
          {messages.map((msg, idx) => (
            <div key={idx} className={`bubble-row ${msg.role}`}>
              <div className="bubble">
                <div className="content">
                  {msg.role === "assistant" ? (
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {msg.content || "..."}
                    </ReactMarkdown>
                  ) : (
                    msg.content
                  )}
                </div>
                {msg.sources && msg.sources.length > 0 && (
                  <div className="sources">
                    Sources: {msg.sources.join(" • ")}
                  </div>
                )}
              </div>
            </div>
          ))}
          <div ref={chatEndRef} />
        </div>

        <form className="input-bar" onSubmit={handleSend}>
          <input
            type="text"
            placeholder="e.g., Compare HDHP vs PPO if I take 1 generic med and rarely see a doctor..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={isStreaming}
          />
          <button type="submit" disabled={isStreaming || !input.trim()}>
            {isStreaming ? "..." : "Send"}
          </button>
        </form>
      </main>
    </div>
  );
}
