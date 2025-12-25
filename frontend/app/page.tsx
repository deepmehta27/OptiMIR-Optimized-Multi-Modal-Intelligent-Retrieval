"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "./globals.css";
import "katex/dist/katex.min.css";
import { OpenAI, Anthropic } from "@lobehub/icons";

type ModelId = "gpt4o" | "gpt-nano" | "claude" | "claude-sonnet";

const MODEL_OPTIONS: { id: ModelId; label: string; provider: string }[] = [
  { id: "gpt4o", label: "GPT-4o Mini", provider: "OpenAI" },
  { id: "gpt-nano", label: "GPT-4.1 Nano", provider: "OpenAI" },
  { id: "claude", label: "Claude Haiku 4.5", provider: "Anthropic" },
  { id: "claude-sonnet", label: "Claude Sonnet 4.5", provider: "Anthropic" },
];

const BACKEND_MODEL_MAP: Record<ModelId, "gpt4o" | "claude"> = {
  gpt4o: "gpt4o",
  "gpt-nano": "gpt4o",
  claude: "claude",
  "claude-sonnet": "claude",
};


type ChatMessage = {
  role: "user" | "assistant";
  text: string;
};

export default function HomePage() {
  const [message, setMessage] = useState("");
  const [model, setModel] = useState<ModelId>("gpt4o");
  const [showModelPicker, setShowModelPicker] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isSending, setIsSending] = useState(false);
  const currentModel = MODEL_OPTIONS.find((m) => m.id === model);

  async function ingestFile(selectedFile: File) {
    const form = new FormData();
    form.append("file", selectedFile);

    const res = await fetch("http://localhost:8000/ingest", {
      method: "POST",
      body: form,
    });

    if (!res.ok) {
      throw new Error("Failed to ingest PDF");
    }
    return res.json();
  }

  async function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0] || null;
    if (!f) return;

    try {
      await ingestFile(f);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: `Document "${f.name}" ingested successfully. You can now ask questions about it.`,
        },
      ]);
    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: "Failed to ingest PDF. Please try again.",
        },
      ]);
    }
  }

  async function sendMessage() {
    if (!message.trim() || isSending) return;

    const userText = message.trim();
    setMessage("");
    setMessages((prev) => [...prev, { role: "user", text: userText }]);
    setIsSending(true);

    const backendModel = BACKEND_MODEL_MAP[model];

const body = JSON.stringify({
  question: userText,
  model: backendModel,
  use_context: true,
});

const endpoint = "http://localhost:8000/chat/stream";

    let assistantText = "";
    setMessages((prev) => [...prev, { role: "assistant", text: "" }]);

    try {
      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
      });

      if (!res.ok || !res.body) {
        throw new Error("Stream error");
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });

        const lines = chunk
          .split("\n")
          .map((l) => l.trim())
          .filter((l) => l.startsWith("data: "));

        for (const line of lines) {
          const dataStr = line.slice(6).trim();
          if (!dataStr || dataStr === "[DONE]") continue;

          try {
            const payload = JSON.parse(dataStr);
            if (payload.type === "token") {
              assistantText += payload.token;
              setMessages((prev) => {
                const copy = [...prev];
                const last = copy[copy.length - 1];
                if (last && last.role === "assistant") {
                  last.text = assistantText;
                }
                return copy;
              });
            }
          } catch {
            // ignore parse errors
          }
        }
      }
    } catch (err) {
      console.error(err);
      setMessages((prev) => {
        const copy = [...prev];
        const last = copy[copy.length - 1];
        if (last && last.role === "assistant") {
          last.text =
            last.text ||
            "Something went wrong while contacting the backend. Please try again.";
        }
        return copy;
      });
    } finally {
      setIsSending(false);
    }
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    void sendMessage();
  }

  return (
    <main className="min-h-screen flex flex-col bg-white text-gray-900 font-sans">
      {/* Header */}
      <header className="px-8 pt-16 max-w-4xl mx-auto w-full">
        <h1 className="text-4xl font-semibold tracking-tight">Hello there!</h1>
        <p className="text-xl text-gray-400 mt-2 font-medium">
          How can I help you today?
        </p>
      </header>

      {/* Messages */}
      <section className="flex-1 max-w-4xl mx-auto w-full px-8 py-10 overflow-y-auto space-y-4">
        {messages.map((m, i) => (
          <div
            key={i}
            className={m.role === "user" ? "text-right" : "text-left"}
          >
            <div
              className={
                "inline-block rounded-2xl px-4 py-2 text-sm " +
                (m.role === "user"
                  ? "bg-gray-900 text-white"
                  : "bg-gray-100 text-gray-800")
              }
            >
          <ReactMarkdown
            remarkPlugins={[remarkMath]}
            rehypePlugins={[rehypeKatex]}
          >
            {m.text}
          </ReactMarkdown>
        </div>
          </div>
        ))}
      </section>

      {/* Bottom */}
      <section className="max-w-4xl mx-auto w-full px-6 pb-8 space-y-4">
        {/* Input box */}
        <div className="rounded-4xl border border-gray-200 bg-white shadow-sm flex flex-col overflow-hidden focus-within:border-gray-300 transition-colors">
          {/* Controls row */}
          <div className="flex items-center justify-between px-5 pt-4 pb-2">
            <div className="flex items-center gap-3">
              {/* Upload */}
              <label className="flex items-center gap-2 px-3 py-1.5 rounded-xl border border-dashed border-gray-300 text-xs text-gray-500 cursor-pointer hover:bg-gray-50">
                <input
                  type="file"
                  accept="application/pdf"
                  className="hidden"
                  onChange={handleFileChange}
                />
                {/* paperclip icon */}
  <svg
    width="14"
    height="14"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    className="text-gray-500"
  >
    <path d="M21.44 11.05l-9.19 9.19a5 5 0 0 1-7.07-7.07L14.38 4.99a3 3 0 0 1 4.24 4.24L9.88 17.98a1 1 0 0 1-1.41-1.41L16.5 8.54"/>
  </svg>
  <span>Upload PDF</span>
</label>

              {/* Model chip */}
              <button
                onClick={() => setShowModelPicker(true)}
                className="flex items-center gap-2 px-3 py-1.5 rounded-xl hover:bg-gray-50 transition-colors"
              >
                {/* model icon */}
             {currentModel?.provider === "OpenAI" ? (
              <OpenAI size={16} />
            ) : (
              <Anthropic size={16} />
            )}
            <span className="text-xs font-medium text-gray-500">
              {currentModel?.label}
            </span>
          </button>
            </div>

            <div />
          </div>

          {/* Text area + send */}
          <form className="flex items-end px-5 pb-4 gap-4" onSubmit={handleSubmit}>
            <textarea
              rows={1}
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              placeholder="Send a message..."
              className="flex-1 py-2 resize-none border-none outline-none text-[15px] placeholder:text-gray-400"
            onKeyDown={(e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (message.trim() && !isSending) {
        void sendMessage();
      }
    }
  }}
/>
            <button
              type="submit"
              disabled={!message.trim() || isSending}
              className="h-10 w-10 rounded-full bg-gray-900 text-white flex items-center justify-center disabled:bg-gray-100 disabled:text-gray-300 transition-all shrink-0"
            >
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M12 19V5M5 12l7-7 7 7" />
              </svg>
            </button>
          </form>
        </div>
      </section>

      {/* Model picker modal */}
      {showModelPicker && (
        <div
          className="fixed inset-0 bg-black/20 backdrop-blur-sm flex items-center justify-center z-50 p-4"
          onClick={() => setShowModelPicker(false)}
        >
          <div
            className="bg-white rounded-3xl shadow-2xl w-full max-w-sm overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="px-6 py-5 border-b border-gray-50">
              <h3 className="text-base font-semibold">Select model</h3>
            </div>

            <div className="p-2">
              {MODEL_OPTIONS.map((m) => (
  <button
    key={m.id}
    onClick={() => {
      setModel(m.id);
      setShowModelPicker(false);
    }}
    className={`w-full flex items-center justify-between px-4 py-3 rounded-2xl transition-colors ${
      model === m.id ? "bg-gray-50" : "hover:bg-gray-50/50"
    }`}
  >
    <div className="flex items-center gap-2">
      {m.provider === "OpenAI" ? (
        <OpenAI size={16} />
      ) : (
        <Anthropic size={16} />
      )}
      <div className="flex flex-col items-start">
        <span className="text-sm font-semibold">{m.label}</span>
        <span className="text-[10px] uppercase tracking-wider text-gray-400 font-bold">
          {m.provider}
        </span>
      </div>
    </div>
    {model === m.id && (
      <svg
        width="18"
        height="18"
        viewBox="0 0 24 24"
        fill="none"
        stroke="black"
        strokeWidth="3"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <polyline points="20 6 9 17 4 12" />
      </svg>
    )}
  </button>
))}
            </div>
          </div>
        </div>
      )}
    </main>
  );
}

