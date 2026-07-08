export type AdkConfig = {
  apiBaseUrl: string;
  appName: string;
  userId: string;
  sessionId: string;
};

export type AdkFunctionCall = {
  name?: string | null;
  args?: Record<string, unknown> | null;
};

export type AdkFunctionResponse = {
  name?: string | null;
  response?: Record<string, unknown> | null;
};

export type AdkPart = {
  text?: string | null;
  functionCall?: AdkFunctionCall | null;
  functionResponse?: AdkFunctionResponse | null;
};

export type AdkEvent = {
  id?: string;
  author?: string;
  timestamp?: number;
  errorCode?: string | null;
  errorMessage?: string | null;
  content?: {
    role?: string | null;
    parts?: AdkPart[] | null;
  } | null;
};

export type UploadResult = {
  version: number;
  canonicalUri: string;
  mimeType?: string | null;
};

const trimTrailingSlash = (value: string) => value.replace(/\/+$/, "");

export const defaultAdkConfig = {
  apiBaseUrl: trimTrailingSlash(
    process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8080",
  ),
  appName: process.env.NEXT_PUBLIC_ADK_APP_NAME || "job_scout",
  userId: process.env.NEXT_PUBLIC_ADK_USER_ID || "local-user",
};

async function requestJson<T>(
  url: string,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(url, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `${response.status} ${response.statusText}`);
  }

  return response.json() as Promise<T>;
}

export async function checkBackendHealth(apiBaseUrl: string) {
  const response = await fetch(`${trimTrailingSlash(apiBaseUrl)}/health`, {
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }

  return response.json() as Promise<Record<string, unknown>>;
}

export async function ensureSession(config: AdkConfig) {
  const base = trimTrailingSlash(config.apiBaseUrl);
  const sessionUrl = `${base}/apps/${config.appName}/users/${config.userId}/sessions/${config.sessionId}`;

  const createResponse = await fetch(sessionUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });

  if (createResponse.ok) {
    return createResponse.json() as Promise<Record<string, unknown>>;
  }

  const existingResponse = await fetch(sessionUrl, { cache: "no-store" });
  if (existingResponse.ok) {
    return existingResponse.json() as Promise<Record<string, unknown>>;
  }

  const detail = await createResponse.text();
  throw new Error(detail || "Could not create or load an ADK session.");
}

export async function runAgent(
  config: AdkConfig,
  message: string,
): Promise<AdkEvent[]> {
  const base = trimTrailingSlash(config.apiBaseUrl);

  return requestJson<AdkEvent[]>(`${base}/run`, {
    method: "POST",
    body: JSON.stringify({
      appName: config.appName,
      userId: config.userId,
      sessionId: config.sessionId,
      newMessage: {
        role: "user",
        parts: [{ text: message }],
      },
      streaming: false,
    }),
  });
}

export async function uploadArtifact(
  config: AdkConfig,
  file: File,
): Promise<UploadResult> {
  const inlineData = await fileToInlineData(file);
  const base = trimTrailingSlash(config.apiBaseUrl);
  const url = `${base}/apps/${config.appName}/users/${config.userId}/sessions/${config.sessionId}/artifacts`;

  return requestJson<UploadResult>(url, {
    method: "POST",
    body: JSON.stringify({
      filename: file.name,
      artifact: {
        inlineData,
      },
      customMetadata: {
        originalName: file.name,
        uploadedFrom: "job-scout-frontend",
      },
    }),
  });
}

export function extractAssistantText(events: AdkEvent[]): string {
  const textParts = events
    .filter((event) => event.author !== "user")
    .flatMap((event) => event.content?.parts || [])
    .map((part) => part.text?.trim())
    .filter((text): text is string => Boolean(text));

  return textParts.join("\n\n").trim();
}

export function extractToolNames(events: AdkEvent[]): string[] {
  const names = events.flatMap((event) =>
    (event.content?.parts || [])
      .map((part) => part.functionCall?.name || part.functionResponse?.name)
      .filter((name): name is string => Boolean(name)),
  );

  return Array.from(new Set(names));
}

export function extractRunError(events: AdkEvent[]): string | null {
  const failed = events.find((event) => event.errorMessage || event.errorCode);
  if (!failed) {
    return null;
  }

  return failed.errorMessage || failed.errorCode || "The backend returned an error.";
}

function fileToInlineData(file: File): Promise<{
  data: string;
  mimeType: string;
  displayName: string;
}> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = () => {
      const result = String(reader.result || "");
      const [, base64 = ""] = result.split(",");

      if (!base64) {
        reject(new Error("Could not read file contents."));
        return;
      }

      resolve({
        data: base64,
        mimeType: file.type || "application/octet-stream",
        displayName: file.name,
      });
    };

    reader.onerror = () => reject(reader.error || new Error("Could not read file."));
    reader.readAsDataURL(file);
  });
}
