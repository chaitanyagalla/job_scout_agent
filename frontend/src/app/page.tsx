"use client";

import {
  BriefcaseBusiness,
  ClipboardList,
  FileText,
  LoaderCircle,
  RotateCcw,
  Search,
  Send,
  Trash2,
  Upload,
  Wifi,
  WifiOff,
} from "lucide-react";
import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import {
  AdkConfig,
  checkBackendHealth,
  defaultAdkConfig,
  ensureSession,
  extractAssistantText,
  extractRunError,
  extractToolNames,
  runAgent,
  uploadArtifact,
} from "@/lib/adkClient";

type ChatMessage = {
  id: string;
  role: "assistant" | "user" | "system";
  content: string;
  detail?: string;
  tools?: string[];
};

type BackendStatus = "checking" | "online" | "offline";

const quickPrompts = [
  {
    icon: Search,
    label: "Remote frontend roles",
    prompt: "Find 10 remote frontend developer jobs for 0-1 years in India.",
  },
  {
    icon: BriefcaseBusiness,
    label: "Full stack in Bengaluru",
    prompt: "Search full stack developer roles in Bengaluru and show the best matches.",
  },
  {
    icon: FileText,
    label: "Use resume",
    prompt: "Use my uploaded resume to find matching MERN stack jobs.",
  },
  {
    icon: ClipboardList,
    label: "Score saved jobs",
    prompt: "Score all jobs you found against my resume.",
  },
];

const fileLimitBytes = 8 * 1024 * 1024;

function createId(prefix: string) {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return `${prefix}-${crypto.randomUUID()}`;
  }

  return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function getStoredSessionId() {
  if (typeof window === "undefined") {
    return "browser-session";
  }

  const current = window.localStorage.getItem("job-scout-session-id");
  if (current) {
    return current;
  }

  const next = createId("session");
  window.localStorage.setItem("job-scout-session-id", next);
  return next;
}

function renderTextWithLinks(text: string) {
  const parts = text.split(/(https?:\/\/[^\s)]+)/g);

  return parts.map((part, index) => {
    if (!part.match(/^https?:\/\//)) {
      return <span key={`${part}-${index}`}>{part}</span>;
    }

    const cleanHref = part.replace(/[.,;:]+$/, "");
    const suffix = part.slice(cleanHref.length);

    return (
      <span key={`${part}-${index}`}>
        <a href={cleanHref} target="_blank" rel="noreferrer">
          {cleanHref}
        </a>
        {suffix}
      </span>
    );
  });
}

export default function Home() {
  const [sessionId, setSessionId] = useState("browser-session");
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "welcome",
      role: "assistant",
      content:
        "Tell me the role, location, experience range, or attach a resume. I will search jobs, fetch details, and rank matches when resume context is available.",
    },
  ]);
  const [pending, setPending] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [backendStatus, setBackendStatus] = useState<BackendStatus>("checking");
  const [statusDetail, setStatusDetail] = useState("Checking backend");
  const [attachedFile, setAttachedFile] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  const config: AdkConfig = useMemo(
    () => ({
      ...defaultAdkConfig,
      sessionId,
    }),
    [sessionId],
  );

  useEffect(() => {
    const storedSessionId = getStoredSessionId();
    setSessionId(storedSessionId);
  }, []);

  useEffect(() => {
    let active = true;

    async function bootSession() {
      setBackendStatus("checking");
      setStatusDetail("Checking backend");

      try {
        await checkBackendHealth(defaultAdkConfig.apiBaseUrl);
        await ensureSession({
          ...defaultAdkConfig,
          sessionId,
        });

        if (active) {
          setBackendStatus("online");
          setStatusDetail("Backend connected");
        }
      } catch (error) {
        if (active) {
          setBackendStatus("offline");
          setStatusDetail(error instanceof Error ? error.message : "Backend unavailable");
        }
      }
    }

    if (sessionId !== "browser-session") {
      void bootSession();
    }

    return () => {
      active = false;
    };
  }, [sessionId]);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, pending]);

  async function handleSubmit(event?: FormEvent<HTMLFormElement>, promptOverride?: string) {
    event?.preventDefault();
    const prompt = (promptOverride || input).trim();

    if (!prompt || pending) {
      return;
    }

    setInput("");
    setPending(true);

    const userMessage: ChatMessage = {
      id: createId("user"),
      role: "user",
      content: prompt,
    };
    setMessages((current) => [...current, userMessage]);

    try {
      await ensureSession(config);
      const events = await runAgent(config, prompt);
      const runError = extractRunError(events);

      if (runError) {
        throw new Error(runError);
      }

      const answer = extractAssistantText(events);
      const tools = extractToolNames(events);

      setMessages((current) => [
        ...current,
        {
          id: createId("assistant"),
          role: "assistant",
          content: answer || "Done. The backend completed the request without a text response.",
          tools,
        },
      ]);
      setBackendStatus("online");
      setStatusDetail("Backend connected");
    } catch (error) {
      setBackendStatus("offline");
      setStatusDetail(error instanceof Error ? error.message : "Request failed");
      setMessages((current) => [
        ...current,
        {
          id: createId("error"),
          role: "system",
          content: "The request could not be completed.",
          detail: error instanceof Error ? error.message : "Unknown error",
        },
      ]);
    } finally {
      setPending(false);
    }
  }

  async function handleFileUpload(file: File | null) {
    if (!file || uploading) {
      return;
    }

    if (file.size > fileLimitBytes) {
      setMessages((current) => [
        ...current,
        {
          id: createId("file-too-large"),
          role: "system",
          content: "The selected file is too large.",
          detail: "Use a resume under 8 MB.",
        },
      ]);
      return;
    }

    setUploading(true);

    try {
      await ensureSession(config);
      const result = await uploadArtifact(config, file);
      setAttachedFile(file.name);
      setBackendStatus("online");
      setStatusDetail("Resume attached");
      setMessages((current) => [
        ...current,
        {
          id: createId("upload"),
          role: "system",
          content: `${file.name} is attached to this session.`,
          detail: result.canonicalUri,
        },
      ]);
    } catch (error) {
      setBackendStatus("offline");
      setStatusDetail(error instanceof Error ? error.message : "Upload failed");
      setMessages((current) => [
        ...current,
        {
          id: createId("upload-error"),
          role: "system",
          content: "The resume could not be attached.",
          detail: error instanceof Error ? error.message : "Unknown error",
        },
      ]);
    } finally {
      setUploading(false);
    }
  }

  function startNewSession() {
    const next = createId("session");
    window.localStorage.setItem("job-scout-session-id", next);
    setSessionId(next);
    setAttachedFile(null);
    setMessages([
      {
        id: "welcome-new",
        role: "assistant",
        content:
          "New session ready. Ask for jobs by role and location, or attach a resume before requesting resume-aware matching.",
      },
    ]);
  }

  function clearChat() {
    setMessages([]);
    setAttachedFile(null);
  }

  const StatusIcon =
    backendStatus === "online" ? Wifi : backendStatus === "offline" ? WifiOff : LoaderCircle;

  return (
    <main className="app-shell">
      <aside className="sidebar" aria-label="Job Scout controls">
        <div className="brand">
          <div className="brand-mark">
            <BriefcaseBusiness size={22} aria-hidden />
          </div>
          <div>
            <p className="eyebrow">Job Scout</p>
            <h1>Agent workspace</h1>
          </div>
        </div>

        <div className={`status status-${backendStatus}`}>
          <StatusIcon size={18} className={backendStatus === "checking" ? "spin" : ""} />
          <div>
            <span>{backendStatus === "online" ? "Online" : backendStatus === "offline" ? "Offline" : "Checking"}</span>
            <small>{statusDetail}</small>
          </div>
        </div>

        <section className="control-group" aria-labelledby="session-heading">
          <h2 id="session-heading">Session</h2>
          <div className="session-id" title={sessionId}>
            {sessionId}
          </div>
          <div className="button-row">
            <button type="button" className="icon-button" onClick={startNewSession} title="New session">
              <RotateCcw size={17} aria-hidden />
              <span>New</span>
            </button>
            <button type="button" className="icon-button subtle" onClick={clearChat} title="Clear chat">
              <Trash2 size={17} aria-hidden />
              <span>Clear</span>
            </button>
          </div>
        </section>

        <section className="control-group" aria-labelledby="resume-heading">
          <h2 id="resume-heading">Resume</h2>
          <label className="upload-zone">
            <Upload size={20} aria-hidden />
            <span>{uploading ? "Attaching..." : attachedFile || "Attach PDF, DOCX, or TXT"}</span>
            <input
              type="file"
              accept=".pdf,.doc,.docx,.txt,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/msword,text/plain"
              onChange={(event) => {
                void handleFileUpload(event.target.files?.[0] || null);
                event.currentTarget.value = "";
              }}
              disabled={uploading}
            />
          </label>
        </section>

        <section className="control-group" aria-labelledby="prompts-heading">
          <h2 id="prompts-heading">Prompts</h2>
          <div className="quick-grid">
            {quickPrompts.map((item) => {
              const Icon = item.icon;
              return (
                <button
                  type="button"
                  key={item.label}
                  className="quick-button"
                  onClick={() => void handleSubmit(undefined, item.prompt)}
                  disabled={pending}
                >
                  <Icon size={17} aria-hidden />
                  <span>{item.label}</span>
                </button>
              );
            })}
          </div>
        </section>
      </aside>

      <section className="chat-panel" aria-label="Job Scout chat">
        <div className="chat-header">
          <div>
            <p className="eyebrow">FastAPI + ADK</p>
            <h2>Job matching chat</h2>
          </div>
          <div className="api-pill">{defaultAdkConfig.apiBaseUrl}</div>
        </div>

        <div className="message-list">
          {messages.length === 0 ? (
            <div className="empty-state">
              <ClipboardList size={28} aria-hidden />
              <p>No messages in this session.</p>
            </div>
          ) : (
            messages.map((message) => (
              <article key={message.id} className={`message message-${message.role}`}>
                <div className="message-meta">
                  <span>{message.role === "assistant" ? "Job Scout" : message.role === "user" ? "You" : "System"}</span>
                  {message.tools && message.tools.length > 0 ? (
                    <small>{message.tools.join(", ")}</small>
                  ) : null}
                </div>
                <div className="message-body">{renderTextWithLinks(message.content)}</div>
                {message.detail ? <div className="message-detail">{message.detail}</div> : null}
              </article>
            ))
          )}

          {pending ? (
            <article className="message message-assistant loading-message">
              <div className="message-meta">
                <span>Job Scout</span>
              </div>
              <div className="message-body">
                <LoaderCircle size={18} className="spin" aria-hidden />
                Searching and ranking...
              </div>
            </article>
          ) : null}
          <div ref={scrollRef} />
        </div>

        <form className="composer" onSubmit={(event) => void handleSubmit(event)}>
          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="Find 15 backend developer jobs in Pune for 0-2 years..."
            rows={3}
            disabled={pending}
          />
          <button type="submit" className="send-button" disabled={pending || !input.trim()} title="Send">
            {pending ? <LoaderCircle size={19} className="spin" aria-hidden /> : <Send size={19} aria-hidden />}
            <span>Send</span>
          </button>
        </form>
      </section>
    </main>
  );
}
