"use client";

import { useEffect, useState,useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "./globals.css";
import "katex/dist/katex.min.css";
import { OpenAI, Anthropic } from "@lobehub/icons";

type ModelId = "gpt4o-mini" | "gpt4o" | "claude-haiku" | "claude-sonnet";

type Model = {
  id: ModelId;
  name: string;
  provider: "OPENAI" | "ANTHROPIC";
  tier: "Budget" | "Premium";       
};

const MAX_FILES = 10;

type RagasResult = {
  status: string;
  count: number;
  scores?: {
    faithfulness?: number;
    answer_relevancy?: number;
    context_precision?: number;
    context_recall?: number;
  };
  interpretation?: {
    faithfulness?: string;
    answer_relevancy?: string;
    context_precision?: string;
    context_recall?: string;
  };
};

const models: Model[] = [
  { 
    id: "gpt4o-mini", 
    name: "GPT-4o Mini", 
    provider: "OPENAI",
    tier: "Budget",
  },
  { 
    id: "gpt4o", 
    name: "GPT-4o", 
    provider: "OPENAI",
    tier: "Premium",
  },
  { 
    id: "claude-haiku", 
    name: "Claude Haiku 4.5", 
    provider: "ANTHROPIC",
    tier: "Budget",
  },
  { 
    id: "claude-sonnet", 
    name: "Claude Sonnet 4.5", 
    provider: "ANTHROPIC",
    tier: "Premium",
  },
];

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
  responseTime?: number;
};

const REFUSAL_TEXT =
  "I am sorry, but the provided documents do not contain information to answer this question.";

export default function HomePage() {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [message, setMessage] = useState("");
  const [model, setModel] = useState<ModelId>("gpt4o");
  const [showModelPicker, setShowModelPicker] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isSending, setIsSending] = useState(false);
  const [pdfs, setPdfs] = useState<string[]>([]);
  const currentModel = models.find((m) => m.id === model);
  const [ragasResult, setRagasResult] = useState<RagasResult | null>(null);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [isBatchUploading, setIsBatchUploading] = useState(false);
  const [images, setImages] = useState<string[]>([]);
  const [selectedEvalModel, setSelectedEvalModel] = useState<ModelId | "all">("all");
  const [showEvalModal, setShowEvalModal] = useState(false);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function runRagasEvalFrontend() {
  setIsEvaluating(true);
  try {
    const body: { limit: number; model_filter?: string } = { limit: 10 };
    
    // Add model filter if not "all"
    if (selectedEvalModel !== "all") {
      // Map frontend model to backend model name
      const modelMap: Record<ModelId, string> = {
        "gpt4o-mini": "gpt-4o-mini",
        "gpt4o": "gpt-4o",
        "claude-haiku": "claude-haiku-4-5-20251001",
        "claude-sonnet": "claude-sonnet-4-5-20250929",
      };
      body.model_filter = modelMap[selectedEvalModel as ModelId];
    }
    
    const res = await fetch("http://localhost:8000/evaluate/ragas", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    
    const data = (await res.json()) as RagasResult;
    setRagasResult(data);
    
    // Show success message
    setMessages((prev) => [
      ...prev,
      {
        role: "assistant",
        text: `✅ **Evaluation Complete**\n\nAnalyzed ${data.count} interactions${
          selectedEvalModel !== "all" ? ` (${models.find(m => m.id === selectedEvalModel)?.name})` : ""
        }.\n\nCheck the sidebar for detailed scores.`,
      },
    ]);
  } catch (e) {
    console.error("Ragas eval failed", e);
    setMessages((prev) => [
      ...prev,
      {
        role: "assistant",
        text: "❌ Evaluation failed. Make sure you have some conversations logged first.",
      },
    ]);
  } finally {
    setIsEvaluating(false);
  }
}

  async function refreshPdfs() {
  try {
    const res = await fetch("http://localhost:8000/documents");
    if (!res.ok) throw new Error("Failed to fetch");
    const data = await res.json();
    
    // data.all contains both PDFs and images
    setPdfs(data.pdfs || []);
    setImages(data.images || []);
  } catch (err) {
    console.error(err);
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

async function ingestImage(selectedFile: File) {
  const form = new FormData();
  form.append("file", selectedFile);

  const res = await fetch("http://localhost:8000/ingest/image", {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const errorData = await res.json();
    const detail = errorData.detail || "Unknown error";
    throw new Error(detail);
  }

  return res.json();
}

  async function ingestFile(selectedFile: File) {
  const form = new FormData();
  form.append("file", selectedFile);

  const res = await fetch("http://localhost:8000/ingest", {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    // Parse error details from backend
    const errorData = await res.json();
    let reason = "Unknown error";
    
    if (errorData.detail) {
      if (typeof errorData.detail === "string") {
        reason = errorData.detail;
      } else if (errorData.detail.reason) {
        reason = errorData.detail.reason;
      } else if (errorData.detail.error) {
        reason = errorData.detail.error;
      }
    }
    
    throw new Error(`Reason: ${reason}`);
  }

  return res.json();
}

  useEffect(() => {
    void refreshPdfs();
  }, []);

   async function sendMessage() {
  if (!message.trim() && selectedFiles.length === 0) return;
  if (isSending || isBatchUploading) return;

  const userText = message.trim();
  
  // Clear message immediately for better UX
  setMessage("");
  
  // Add user message to chat
  if (userText) {
    setMessages((prev) => [...prev, { role: "user", text: userText }]);
  }

  // ✅ STEP 1: If files are attached, upload them first
  if (selectedFiles.length > 0) {
    const uploadSuccess = await uploadAllFiles();
    
    if (!uploadSuccess) {
      // If upload failed, don't proceed with question
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: "⚠️ Some files failed to upload. Please try again or ask your question without those files.",
        },
      ]);
      return;
    }
    
    // Small delay to let the UI update
    await new Promise(resolve => setTimeout(resolve, 500));
  }

  // ✅ STEP 2: If user asked a question, answer it now
  if (!userText) return; // No question asked, just uploaded files

  setIsSending(true);
    const startTime = Date.now();

  const body = JSON.stringify({
    question: userText,
    model: model,
    use_context: true,
    history: messages,
  });

  const endpoint = "http://localhost:8000/chat/stream";

  let assistantText = "";

  try {
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
    });

    if (!res.ok || !res.body) throw new Error("Stream error");

    const reader = res.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk
        .split("\n")
        .map((l) => l.trim())
        .filter((l) => l.startsWith("data:"));

      for (const line of lines) {
        const dataStr = line.slice(6).trim();
        if (!dataStr || dataStr === "DONE") continue;

        try {
      const payload = JSON.parse(dataStr);

      // ✅ Handle metadata (includes chunks/sources)
      if (payload.type === "meta") {
              const chunks = (payload.chunks ?? []) as SourceChunk[];
              if (chunks.length > 0) {
                const sources: SourceChunk[] = chunks.map((c) => ({
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
            }
            return copy;
          });
        }
      }

      // ✅ Handle streaming tokens
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
          (last.text || "") +
          "\n\nSomething went wrong while contacting the backend. Please try again.";
      }
      return copy;
    });
  } finally {
    setIsSending(false);
    // Calculate response time and add to last assistant message
    const endTime = Date.now();
    const responseTime = endTime - startTime;
    setMessages((prev) => {
      const copy = [...prev];
      const lastMsg = copy[copy.length - 1];
      if (lastMsg && lastMsg.role === "assistant") {
        lastMsg.responseTime = responseTime;
      }
      return copy;
    });
  }
}

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    void sendMessage();
  }
// Add files to queue (removed unused 'type' parameter)
function addFilesToQueue(files: FileList | null) {
  if (!files) return;
  
  const newFiles = Array.from(files);
  const currentCount = selectedFiles.length;
  const remainingSlots = MAX_FILES - currentCount;
  
  if (remainingSlots <= 0) {
    alert(`Maximum ${MAX_FILES} files allowed. Please upload current batch first.`);
    return;
  }
  
  const filesToAdd = newFiles.slice(0, remainingSlots);
  
  if (newFiles.length > remainingSlots) {
    alert(`Only adding ${remainingSlots} files. Maximum is ${MAX_FILES} files per upload.`);
  }
  
  // ✅ Just add to queue - don't auto-upload
  setSelectedFiles(prev => [...prev, ...filesToAdd]);
}

// Remove file from queue
function removeFileFromQueue(index: number) {
  setSelectedFiles(prev => prev.filter((_, i) => i !== index));
}

// Upload all queued files
async function uploadAllFiles(): Promise<boolean> {
  if (selectedFiles.length === 0) return true;
  
  setIsBatchUploading(true);
  
  // Store files before clearing (important!)
  const filesToUpload = [...selectedFiles];
  
  // ✅ Clear chips BEFORE starting upload
  setSelectedFiles([]);
  
  setMessages((prev) => [
    ...prev,
    {
      role: "assistant",
      text: `🔄 Processing ${filesToUpload.length} document(s) before answering...\n\nPlease wait...`,
    },
  ]);
  
  const results = {
    successful: [] as string[],
    failed: [] as string[],
  };
  
  // Use filesToUpload instead of selectedFiles
  for (const file of filesToUpload) {
    try {
      const isPdf = file.type === "application/pdf";
      if (isPdf) {
        await ingestFile(file);
      } else {
        await ingestImage(file);
      }
      results.successful.push(file.name);
    } catch (err: unknown) {
      console.error(`Failed to upload ${file.name}:`, err);
      results.failed.push(file.name);
    }
  }
  
  await refreshPdfs();
  
  // Show results
  let resultText = "";
  if (results.successful.length > 0) {
    resultText += `✅ ${results.successful.length} document(s) processed successfully.\n`;
  }
  
  if (results.failed.length > 0) {
    resultText += `❌ ${results.failed.length} document(s) failed:\n`;
    results.failed.forEach(name => {
      resultText += `  • ${name}\n`;
    });
  }
  
  setMessages((prev) => {
    const withoutProcessing = prev.slice(0, -1);
    return [
      ...withoutProcessing,
      {
        role: "assistant",
        text: resultText,
      },
    ];
  });
  
  setIsBatchUploading(false);
  
  return results.failed.length === 0;
}

// For PDF uploads - simplified
// Unified file selector - accepts both PDFs and images
function handleFileSelect(e: React.ChangeEvent<HTMLInputElement>) {
  addFilesToQueue(e.target.files);
  e.target.value = ''; // Reset input
}

 return (
  <main className="min-h-screen flex bg-white text-gray-900 font-sans">
    {/* Sidebar */}
    <aside className="h-screen w-64 border-r border-gray-200 bg-white px-5 py-6 hidden md:flex flex-col shrink-0">
      {/* Metrics Explanation - appears BEFORE scorecard */}
        <div className="mb-3 p-3 bg-linear-to-br from-blue-50 to-indigo-50 rounded-xl border border-blue-100">
          <div className="flex items-start gap-2">
            <svg className="w-4 h-4 text-blue-600 mt-0.5 shrink-0" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
            </svg>
            <div>
              <p className="text-xs font-semibold text-blue-900 mb-1">Understanding Your Scores</p>
              <p className="text-10px text-blue-700 leading-relaxed">
                Quality scores measure how well OptiMIR answers your questions. <strong>Have several conversations first</strong> before evaluating for meaningful results.
              </p>
            </div>
          </div>
        </div>
<div className="mb-4 rounded-2xl border border-gray-200 bg-white px-4 py-3 shadow-sm">
  <div className="flex items-center justify-between mb-3">
    <div>
      <p className="text-xs font-semibold text-gray-700">RAG Quality Score</p>
      <p className="text-[10px] text-gray-400 mt-0.5">
        Last {ragasResult?.count ?? 0} interactions
      </p>
    </div>
    <button
      onClick={() => setShowEvalModal(true)}
      disabled={isEvaluating}
      className="text-[10px] px-3 py-1.5 rounded-full bg-blue-50 text-blue-600 hover:bg-blue-100 disabled:opacity-50 disabled:cursor-not-allowed font-medium transition-colors"
    >
      {isEvaluating ? (
        <span className="flex items-center gap-1">
          <svg className="animate-spin h-3 w-3" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
          </svg>
          Evaluating...
        </span>
      ) : "Evaluate"}
    </button>
  </div>

  {/* Score Grid */}
<div className="grid grid-cols-2 gap-2">
  {[
            { 
              label: "Faithfulness", 
              key: "faithfulness" as const, 
              desc: "Answers based only on your documents"
            },
            { 
              label: "Relevancy", 
              key: "answer_relevancy" as const, 
              desc: "Directly answers your question"
            },
            { 
              label: "Precision", 
              key: "context_precision" as const, 
              desc: "Finds the right information"
            },
            { 
              label: "Recall", 
              key: "context_recall" as const, 
              desc: "Finds all needed information"
            }
  ].map((metric) => {
    const score = ragasResult?.scores?.[metric.key];
    const value = typeof score === "number" ? score : 0;
    const percentage = (value * 100).toFixed(0);
    const color = value >= 0.7 ? "text-green-600" : value >= 0.5 ? "text-yellow-600" : "text-red-600";
    
    return (
      <div key={metric.key} className="bg-gray-50 rounded-lg px-2 py-2">
        <div className="flex items-center justify-between mb-1">
          <span className="text-[10px] font-medium text-gray-600">
            {metric.label}
          </span>
          <span className={`text-xs font-bold ${color}`}>
            {ragasResult ? `${percentage}%` : "-"}
          </span>
        </div>
        <p className="text-[9px] text-gray-400">{metric.desc}</p>
      </div>
    );
  })}
</div>
</div>
      
      <h2 className="text-sm font-semibold text-gray-700 mb-4 tracking-tight">
  Documents
</h2>

<div className="flex-1 overflow-y-auto">
  {pdfs.length === 0 && images.length === 0 ? (
    <p className="text-sm text-gray-400">No documents uploaded yet.</p>
  ) : (
    <div className="space-y-4">
      {/* PDFs Section */}
      {pdfs.length > 0 && (
        <div>
          <p className="text-xs font-medium text-gray-500 mb-2 px-1">
            PDFs ({pdfs.length})
          </p>
          <ul className="space-y-2">
            {pdfs.map((src) => (
              <li
                key={src}
                className="group flex items-center justify-between gap-2 text-sm text-gray-800 bg-gray-50 hover:bg-gray-100 rounded-xl px-3 py-2.5 transition-colors"
              >
                <span className="flex items-center gap-2 truncate">
                  <svg
                    className="w-4 h-4 text-red-500 shrink-0"
                    fill="currentColor"
                    viewBox="0 0 20 20"
                  >
                    <path
                      fillRule="evenodd"
                      d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4z"
                      clipRule="evenodd"
                    />
                  </svg>
                  <span className="truncate" title={src}>
                    {src}
                  </span>
                </span>
                <button
                  onClick={() => void deletePdf(src)}
                  className="text-gray-400 hover:text-red-500 opacity-0 group-hover:opacity-100 transition-all shrink-0"
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
        </div>
      )}
      
      {/* Images Section */}
      {images.length > 0 && (
        <div>
          <p className="text-xs font-medium text-gray-500 mb-2 px-1">
            Images ({images.length})
          </p>
          <ul className="space-y-2">
            {images.map((src) => (
              <li
                key={src}
                className="group flex items-center justify-between gap-2 text-sm text-gray-800 bg-gray-50 hover:bg-gray-100 rounded-xl px-3 py-2.5 transition-colors"
              >
                <span className="flex items-center gap-2 truncate">
                  <svg
                    className="w-4 h-4 text-blue-500 shrink-0"
                    fill="currentColor"
                    viewBox="0 0 20 20"
                  >
                    <path
                      fillRule="evenodd"
                      d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l4-8 3 6 2-4 3 6z"
                      clipRule="evenodd"
                    />
                  </svg>
                  <span className="truncate" title={src}>
                    {src}
                  </span>
                </span>
                <button
                  onClick={() => void deletePdf(src)}
                  className="text-gray-400 hover:text-red-500 opacity-0 group-hover:opacity-100 transition-all shrink-0"
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
        </div>
      )}
    </div>
  )}
</div>
    </aside>

    {/* Main chat area */}
    <div className="flex-1 flex flex-col h-screen relative overflow-hidden">
      {/* Scrollable messages */}
        <section className="flex-1 overflow-y-auto pb-48">
          <div className="max-w-4xl mx-auto w-full px-8 pt-16">
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
                            <li key={idx2} className="flex flex-col gap-1">
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
                    {/* ✅ Response time badge - shown BELOW assistant messages */}
                  {m.role === "assistant" && m.responseTime && (
                    <div className="mt-2">
                      <span className="text-xs bg-green-50 text-green-700 px-3 py-1 rounded-full font-medium">
                        ⚡ Answered in {m.responseTime}ms
                      </span>
                    </div>
                  )}
              
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        </div>
      </section>

      {/* Floating input */}
<section 
  className="fixed bottom-0 left-0 md:left-64 right-0 bg-linear-to-t from-white via-white to-transparent pt-10 pb-6 z-50"
  style={{ pointerEvents: 'none' }}
>
  <div className="max-w-4xl mx-auto w-full px-6" style={{ pointerEvents: 'auto' }}>
    {/* File attachment chips */}
    {selectedFiles.length > 0 && (
      <div className="mb-3 flex flex-wrap gap-2">
        {selectedFiles.map((file, idx) => (
          <div
            key={idx}
            className="flex items-center gap-2 bg-white border border-gray-200 rounded-2xl px-4 py-2 shadow-sm"
          >
            {/* File icon */}
            <div className="w-10 h-10 bg-red-100 rounded-lg flex items-center justify-center shrink-0">
              {file.type === "application/pdf" ? (
                <svg
                  width="20"
                  height="20"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                >
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                  <polyline points="14 2 14 8 20 8" />
                  <line x1="16" y1="13" x2="8" y2="13" />
                  <line x1="16" y1="17" x2="8" y2="17" />
                  <polyline points="10 9 9 9 8 9" />
                </svg>
              ) : (
                <svg
                  width="20"
                  height="20"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                >
                  <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                  <circle cx="8.5" cy="8.5" r="1.5" />
                  <polyline points="21 15 16 10 5 21" />
                </svg>
              )}
            </div>

            {/* File info */}
            <div className="flex flex-col min-w-0">
              <p className="text-sm font-medium text-gray-900 truncate max-w-200">
                {file.name}
              </p>
              <p className="text-xs text-gray-500">
                {file.type === "application/pdf" ? "PDF" : "Image"}
              </p>
            </div>

            {/* Remove button */}
            <button
              onClick={() => removeFileFromQueue(idx)}
              className="ml-2 w-6 h-6 rounded-full hover:bg-gray-100 flex items-center justify-center shrink-0"
            >
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </div>
        ))}
      </div>
    )}

    <div className="rounded-4xl border border-gray-200 bg-white shadow-sm flex flex-col overflow-hidden focus-within:border-gray-400 transition-all">
      <div className="flex items-center justify-between px-5 pt-4 pb-1">
        <div className="flex items-center gap-2">
          {/* Add Files Button */}
          <label
            className={`flex items-center gap-2 px-3 py-1.5 rounded-xl text-[11px] font-bold transition-all cursor-pointer ${
              isBatchUploading
                ? "bg-gray-200 text-gray-400 cursor-not-allowed"
                : "hover:bg-gray-50 text-gray-600"
            }`}
          >
            <input
              type="file"
              accept="application/pdf,image/jpeg,image/jpg,image/png,image/webp"
              multiple
              className="hidden"
              onChange={handleFileSelect}
              disabled={isBatchUploading}
            />
            <svg
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-7.07-7.07l9.19-9.19a4 4 0 1 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" />
            </svg>
            <span>Add Files</span>
          </label>
        </div>

        <button
      onClick={() => setShowModelPicker(true)}
      className="flex items-center gap-2 px-3 py-1.5 rounded-xl hover:bg-gray-50 transition-colors"
    >
      {currentModel?.provider === "OPENAI" ? (
        <OpenAI size={14} />
      ) : (
        <Anthropic size={14} />
      )}
      <span className="text-[11px] font-bold text-gray-500">
        {currentModel?.name}
      </span>
    </button>
      </div>
      <form
  className="flex items-end px-5 pb-4 gap-4"
  onSubmit={handleSubmit}
>
  <textarea
  rows={1}
  value={message}
  onChange={(e) => setMessage(e.target.value)}
  placeholder={
    isBatchUploading 
      ? "Processing files..." 
      : selectedFiles.length > 0 
        ? `Ask a question about ${selectedFiles.length} file(s)...`
        : "Send a message..."
  }
  disabled={isBatchUploading}
  className="flex-1 py-3 resize-none border-none outline-none text-[15px] placeholder:text-gray-400 max-h-32 disabled:bg-gray-50 disabled:cursor-not-allowed"
  onKeyDown={(e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if ((message.trim() || selectedFiles.length > 0) && !isSending && !isBatchUploading) {
        void sendMessage();
      }
    }
  }}
/>
  <button
  type="submit"
  disabled={
    (selectedFiles.length === 0 && !message.trim()) || // Disabled if BOTH empty
    isSending || 
    isBatchUploading
  }
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
              {models.map((m) => (
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
              {m.provider === "OPENAI" ? (
                <OpenAI size={16} />
              ) : (
                <Anthropic size={16} />
              )}
              <div className="flex flex-col items-start">
            <div className="flex items-center gap-2">
              <span className="text-sm font-semibold">{m.name}</span>
              <span 
                className={`text-[10px] px-1.5 py-0.5 rounded-full font-semibold ${
                  m.tier === "Budget" 
                    ? "bg-gray-100 text-gray-700" 
                    : "bg-black text-white"
                }`}
              >
                {m.tier}
              </span>
            </div>
            <span className="text-[10px] text-gray-400">
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
    {/* Evaluation Modal */}
{showEvalModal && (
  <div
    className="fixed inset-0 bg-black/40 backdrop-blur-sm flex items-center justify-center z-50 p-4"
    onClick={() => setShowEvalModal(false)}
  >
    <div
      className="bg-white rounded-3xl shadow-2xl w-full max-w-lg overflow-hidden"
      onClick={(e) => e.stopPropagation()}
    >
      {/* Header */}
      <div className="px-6 py-5 border-b border-gray-100 bg-linear-to-r from-blue-50 to-white">
        <h3 className="text-lg font-semibold text-gray-900">
          Evaluate RAG Quality
        </h3>
        <p className="text-xs text-gray-500 mt-1">
          Measure how well your AI is performing using RAGAS metrics
        </p>
      </div>

      {/* Content */}
      <div className="px-6 py-5">
        {/* What is RAGAS */}
        <div className="mb-5 p-3 bg-blue-50 rounded-xl border border-blue-100">
          <div className="flex items-start gap-2">
            <svg className="w-4 h-4 text-blue-600 mt-0.5 shrink-0" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd"/>
            </svg>
            <div>
              <p className="text-xs font-medium text-blue-900 mb-1">
                What is RAGAS?
              </p>
              <p className="text-[11px] text-blue-700 leading-relaxed">
                RAGAS evaluates your RAG system across 4 dimensions: faithfulness (no hallucinations), 
                answer relevancy (on-topic), context precision (quality retrieval), and context recall 
                (completeness). Higher scores mean better performance.
              </p>
            </div>
          </div>
        </div>

        {/* Model Selection */}
        <div className="mb-5">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Select Model to Evaluate
          </label>
          <div className="space-y-2">
            {[
              { id: "all", name: "All Models", desc: "Evaluate all conversations" },
              ...models.map(m => ({ 
                id: m.id, 
                name: m.name, 
                desc: `Only ${m.name} conversations` 
              }))
            ].map((option) => (
              <button
                key={option.id}
                onClick={() => setSelectedEvalModel(option.id as ModelId | "all")}
                className={`w-full text-left px-4 py-3 rounded-xl border-2 transition-all ${
                  selectedEvalModel === option.id
                    ? "border-blue-500 bg-blue-50"
                    : "border-gray-200 hover:border-gray-300 bg-white"
                }`}
              >
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-900">
                      {option.name}
                    </p>
                    <p className="text-xs text-gray-500 mt-0.5">
                      {option.desc}
                    </p>
                  </div>
                  {selectedEvalModel === option.id && (
                    <svg className="w-5 h-5 text-blue-600" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd"/>
                    </svg>
                  )}
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Warning */}
        <div className="mb-5 p-3 bg-amber-50 rounded-xl border border-amber-100">
          <div className="flex items-start gap-2">
            <svg className="w-4 h-4 text-amber-600 mt-0.5 shrink-0" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd"/>
            </svg>
            <p className="text-[11px] text-amber-800 leading-relaxed">
              This evaluation uses GPT-4o-mini to judge quality and may take 30-60 seconds. 
              Make sure you have active conversations logged before running.
            </p>
          </div>
        </div>
      </div>

      {/* Actions */}
      <div className="px-6 py-4 bg-gray-50 flex items-center justify-end gap-3">
        <button
          onClick={() => setShowEvalModal(false)}
          className="px-4 py-2 text-sm font-medium text-gray-700 hover:text-gray-900 transition-colors"
        >
          Cancel
        </button>
        <button
          onClick={() => {
            setShowEvalModal(false);
            void runRagasEvalFrontend();
          }}
          disabled={isEvaluating}
          className="px-5 py-2 text-sm font-semibold text-white bg-blue-600 hover:bg-blue-700 rounded-xl transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isEvaluating ? "Evaluating..." : "Start Evaluation"}
        </button>
      </div>
    </div>
  </div>
)}
  </main>
);
}