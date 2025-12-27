"use client";

import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "./globals.css";
import "katex/dist/katex.min.css";
import { OpenAI, Anthropic } from "@lobehub/icons";

type ModelId = "gpt4o" | "gpt-nano" | "claude" | "claude-sonnet";

type RagasResult = {
  status: string;
  count: number;
  scores?: { faithfulness?: number };
};

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

type SourceChunk = {
  source: string;
  page?: number;
  score?: number;
  text: string;
};

type ChatMessage = {
  role: "user" | "assistant";
  text: string;
  sources?: SourceChunk[];
};

const REFUSAL_TEXT =
  "I am sorry, but the provided documents do not contain information to answer this question.";

export default function HomePage() {
  const [message, setMessage] = useState("");
  const [model, setModel] = useState<ModelId>("gpt4o");
  const [showModelPicker, setShowModelPicker] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isSending, setIsSending] = useState(false);
  const [pdfs, setPdfs] = useState<string[]>([]);
  const currentModel = MODEL_OPTIONS.find((m) => m.id === model);
  const [ragasResult, setRagasResult] = useState<RagasResult | null>(null);
  const [isEvaluating, setIsEvaluating] = useState(false);

  async function runRagasEvalFrontend() {
  setIsEvaluating(true);
  try {
    const res = await fetch("http://localhost:8000/evaluate/ragas", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ limit: 5 }),
    });
    const data = (await res.json()) as RagasResult;
    setRagasResult(data);
  } catch (e) {
    console.error("Ragas eval failed", e);
  } finally {
    setIsEvaluating(false);
  }
}

  async function refreshPdfs() {
    try {
      const res = await fetch("http://localhost:8000/pdfs");
      const data = await res.json();
      setPdfs(data.sources || []);
    } catch (e) {
      console.error("Failed to load PDFs", e);
    }
  }

  async function deletePdf(source: string) {
    try {
      await fetch("http://localhost:8000/delete_pdf", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source }),
      });

      setPdfs((prev) => prev.filter((s) => s !== source));
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: `Removed "${source}" from the OptiMIR index. Future answers will no longer use it.`,
        },
      ]);
    } catch (e) {
      console.error("Failed to delete PDF", e);
    }
  }

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

  useEffect(() => {
    void refreshPdfs();
  }, []);

  async function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0] || null;
    if (!f) return;

    try {
      await ingestFile(f);
      await refreshPdfs();
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
      history: messages.slice(-4),
    });

    const endpoint = "http://localhost:8000/chat/stream";

    let assistantText = "";

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

            if (payload.type === "meta") {
              const chunks = (payload.chunks ?? []) as SourceChunk[];
              const sources: SourceChunk[] = chunks.map((c: SourceChunk) => ({
                source: c.source,
                page: c.page,
                score: c.score,
                text: c.text,
              }));

              setMessages((prev) => {
                const copy = [...prev];
                const last = copy[copy.length - 1];
                if (last && last.role === "assistant") {
                  last.sources = sources;
                  return copy;
                }
                // no assistant yet for this turn → create it
                return [...copy, { role: "assistant", text: "", sources }];
              });
            }

            if (payload.type === "token") {
              assistantText += payload.token;

              setMessages((prev) => {
                const copy = [...prev];
                let last = copy[copy.length - 1];

                if (!last || last.role !== "assistant") {
                  last = { role: "assistant", text: "" };
                  copy.push(last);
                }

                last.text = assistantText;
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
    <main className="min-h-screen flex bg-white text-gray-900 font-sans">
      {/* Sidebar */}
      <aside className="h-screen w-64 border-r border-gray-200 bg-white px-5 py-6 hidden md:flex flex-col shrink-0">
        <div className="mb-4 rounded-2xl border border-gray-200 bg-gray-50 px-3 py-2.5">
  <div className="flex items-center justify-between mb-2">
    <p className="text-[11px] font-semibold text-gray-500">
      RAG quality (last {ragasResult?.count ?? 0} runs)
    </p>
    <button
      onClick={runRagasEvalFrontend}
      disabled={isEvaluating}
      className="text-[10px] px-2 py-0.5 rounded-full border border-gray-300 text-gray-600 hover:bg-gray-100 disabled:opacity-50"
    >
      {isEvaluating ? "Evaluating..." : "Run eval"}
    </button>
  </div>
  <div className="flex justify-between text-xs text-gray-700">
  <span>Faithfulness</span>
  <span>
    {/* Check if it's a valid number before calling toFixed */}
    {typeof ragasResult?.scores?.faithfulness === "number"
      ? ragasResult.scores.faithfulness.toFixed(2)
      : "-"}
  </span>
</div>
</div>
        <h2 className="text-sm font-semibold text-gray-700 mb-4 tracking-tight">
          Documents
        </h2>
        <div className="flex-1 overflow-y-auto">
          {pdfs.length === 0 ? (
            <p className="text-sm text-gray-400">No PDFs uploaded yet.</p>
          ) : (
            <ul className="space-y-2">
              {pdfs.map((src) => (
                <li
                  key={src}
                  className="group flex items-center justify-between gap-2 text-sm text-gray-800 bg-gray-50 hover:bg-gray-100 rounded-xl px-3 py-2.5 transition-colors"
                >
                  <span className="truncate" title={src}>
                    {src}
                  </span>
                  <button
                    onClick={() => void deletePdf(src)}
                    className="text-gray-400 hover:text-red-500 opacity-0 group-hover:opacity-100 transition-all"
                    title="Remove from index"
                  >
                    <svg
                      width="14"
                      height="14"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <polyline points="3 6 5 6 21 6" />
                      <path d="M19 6l-1 14H6L5 6" />
                      <path d="M10 11v6" />
                      <path d="M14 11v6" />
                      <path d="M9 6V4h6v2" />
                    </svg>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      </aside>

      {/* Main chat area */}
      <div className="flex-1 flex flex-col h-screen relative overflow-hidden">
        {/* Scrollable messages */}
        <section className="flex-1 overflow-y-auto">
          <div className="max-w-4xl mx-auto w-full px-8 pt-16 pb-40">
            {/* Greeting when empty */}
            {messages.length === 0 && (
              <header className="mb-12">
                <h1 className="text-3xl font-semibold tracking-tight text-gray-900">
                  Hello there!
                </h1>
                <p className="text-lg text-gray-400 mt-2 font-medium">
                  How can I help you today?
                </p>
              </header>
            )}

            {/* Messages */}
            <div className="space-y-6">
              {messages.map((m, i) => (
                <div
                  key={i}
                  className={`flex ${
                    m.role === "user" ? "justify-end" : "justify-start"
                  }`}
                >
                  <div
                    className={`max-w-[85%] rounded-2xl px-5 py-3 text-sm leading-relaxed ${
                      m.role === "user"
                        ? "bg-gray-900 text-white shadow-sm"
                        : "bg-gray-100 text-gray-800"
                    }`}
                  >
                    <div className="prose prose-sm max-w-none prose-p:leading-relaxed">
                      <ReactMarkdown
                        remarkPlugins={[remarkMath]}
                        rehypePlugins={[rehypeKatex]}
                      >
                        {m.text}
                      </ReactMarkdown>
                    </div>

                    {m.role === "assistant" &&
                      m.sources &&
                      m.sources.length > 0 &&
                      m.text.trim() !== REFUSAL_TEXT && (
                        <details className="mt-3 pt-3 border-t border-gray-200/50 text-xs text-gray-500">
                          <summary className="cursor-pointer select-none font-medium hover:text-gray-700">
                            Sources ({m.sources.length})
                          </summary>
                          <ul className="mt-3 space-y-2">
                            {m.sources.slice(0, 4).map((s, idx2) => (
                              <li
                                key={idx2}
                                className="flex flex-col gap-1"
                              >
                                <span className="font-medium text-gray-700">
                                  {s.source}
                                  {s.page ? ` — page ${s.page}` : ""}
                                </span>
                                <span className="line-clamp-2 text-[11px] text-gray-400">
                                  {s.text}
                                </span>
                              </li>
                            ))}
                          </ul>
                        </details>
                      )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Floating input */}
        <section className="absolute bottom-0 left-0 right-0 bg-linear-to-t from-white via-white to-transparent pt-10 pb-6">
          <div className="max-w-4xl mx-auto w-full px-6">
            <div className="rounded-4xl border border-gray-100 bg-white shadow-sm flex flex-col overflow-hidden focus-within:border-gray-400 transition-all">
              <div className="flex items-center justify-between px-5 pt-4 pb-1">
                <div className="flex items-center gap-3">
                  <label className="flex items-center gap-2 px-3 py-1.5 rounded-xl border border-dashed border-gray-300 text-[11px] font-bold text-gray-500 cursor-pointer hover:bg-gray-50">
                    <input
                      type="file"
                      accept="application/pdf"
                      className="hidden"
                      onChange={handleFileChange}
                    />
                    <svg
                      width="14"
                      height="14"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <path d="M21.44 11.05l-9.19 9.19a5 5 0 0 1-7.07-7.07L14.38 4.99a3 3 0 0 1 4.24 4.24L9.88 17.98a1 1 0 0 1-1.41-1.41L16.5 8.54" />
                    </svg>
                    <span>Upload PDF</span>
                  </label>

                  <button
                    onClick={() => setShowModelPicker(true)}
                    className="flex items-center gap-2 px-3 py-1.5 rounded-xl hover:bg-gray-50 transition-colors"
                  >
                    {currentModel?.provider === "OpenAI" ? (
                      <OpenAI size={14} />
                    ) : (
                      <Anthropic size={14} />
                    )}
                    <span className="text-[11px] font-bold text-gray-500">
                      {currentModel?.label}
                    </span>
                  </button>
                </div>
              </div>

              <form
                className="flex items-end px-5 pb-4 gap-4"
                onSubmit={handleSubmit}
              >
                <textarea
                  rows={1}
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  placeholder="Send a message..."
                  className="flex-1 py-3 resize-none border-none outline-none text-[15px] placeholder:text-gray-400 max-h-32"
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      if (message.trim() && !isSending) void sendMessage();
                    }
                  }}
                />
                <button
                  type="submit"
                  disabled={!message.trim() || isSending}
                  className="h-10 w-10 rounded-full bg-gray-900 text-white flex items-center justify-center disabled:bg-gray-100 disabled:text-gray-300 transition-all shrink-0"
                >
                  <svg
                    width="18"
                    height="18"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="3"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <path d="M12 19V5M5 12l7-7 7 7" />
                  </svg>
                </button>
              </form>
            </div>
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
                        <span className="text-sm font-semibold">
                          {m.label}
                        </span>
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
      </div>
    </main>
  );
}
